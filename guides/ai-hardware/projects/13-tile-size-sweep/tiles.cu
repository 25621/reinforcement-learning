// Project 13 - tiling a matmul, and what tile size actually buys.
//
// A tile is a square block of the matrices held in fast memory so that each
// element loaded is USED many times instead of once. That reuse is the entire
// source of arithmetic intensity in a matmul, and arithmetic intensity is what
// decides whether you are allowed anywhere near peak FLOPs.
//
// The arithmetic, once, for a TxT shared tile (one output per thread):
//   per TxT output tile:  FLOPs                 = 2*T*T*N
//                         bytes loaded from DRAM = 2*T*N*4
//                         arithmetic intensity   = T/4  FLOP/byte
//
// The ridge point of this GPU (project 2) is 31.9 FLOP/byte, so a single-level
// shared tile would need T = 128 to reach compute-bound - and two 128x128
// float tiles are 128 KB, against a 48 KB budget. The conclusion is forced:
// ONE level of tiling cannot make this matmul compute-bound. You need a second
// level, in registers, where the tile costs no shared memory at all.
//
//   register tiling: each thread computes a TMxTN patch of C, so the BMxBN
//   block tile gives   AI = BM*BN / (2*(BM+BN))  at the DRAM level,
//   while shared memory only has to hold (BM+BN)*BK elements.
//
// Compiled configurations sweep both levels, plus naive and cuBLAS.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

static const int N = 2048;                 // square matrices, 16 MB each

// ---------------------------------------------------------------------------
// 0. No tiling at all. Every element of A and B is re-read from memory for
//    every output it contributes to. AI = 2 flops / 8 bytes = 0.25.
// ---------------------------------------------------------------------------
__global__ void k_naive(const float *A, const float *B, float *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float acc = 0.f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// 1. One level: a TxT shared tile, one output per thread. AI = T/4.
// ---------------------------------------------------------------------------
template <int T>
__global__ void k_smem(const float *A, const float *B, float *C, int n) {
    __shared__ float As[T][T], Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < n; k0 += T) {
        As[ty][tx] = A[row * n + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// 2. Two levels: a BMxBN block tile in shared memory, and a TMxTN patch per
//    thread in registers. The register patch is the trick - it multiplies
//    arithmetic intensity without spending any shared memory, which is the
//    resource that ran out above.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void k_reg(const float *__restrict__ A, const float *__restrict__ B,
                      float *__restrict__ C, int n) {
    constexpr int NT = (BM / TM) * (BN / TN);        // threads per block
    __shared__ float As[BK][BM];                     // stored transposed
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int nn = 0; nn < TN; ++nn) acc[m][nn] = 0.f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        // A tile: consecutive threads take consecutive k, so each group of BK
        // threads reads one contiguous run - full 32-byte sectors.
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            As[c][r] = A[(row0 + r) * n + k0 + c];
        }
        // B tile: consecutive threads take consecutive columns - coalesced.
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = B[(k0 + r) * n + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[k][ty * TM + m];
#pragma unroll
            for (int nn = 0; nn < TN; ++nn) b[nn] = Bs[k][tx * TN + nn];
#pragma unroll
            for (int m = 0; m < TM; ++m)
#pragma unroll
                for (int nn = 0; nn < TN; ++nn) acc[m][nn] = fmaf(a[m], b[nn], acc[m][nn]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int nn = 0; nn < TN; ++nn)
            C[(row0 + ty * TM + m) * n + col0 + tx * TN + nn] = acc[m][nn];
}

__global__ void k_spin(float *out, int iters) {
    float a = threadIdx.x * 1e-4f, b = 1.0000001f, c = 1e-7f;
    for (int i = 0; i < iters; ++i) a = fmaf(a, b, c);
    if (a == 1.2345e30f) out[0] = a;
}

// ---------------------------------------------------------------------------
static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

template <class F>
static float time_ms(F launch, int iters = 5, int reps = 5) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(); CHK(cudaDeviceSynchronize());
    float t[8];
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(b); CHK(cudaEventSynchronize(b));
        cudaEventElapsedTime(&t[r], a, b);
        t[r] /= iters;
    }
    qsort(t, reps, sizeof(float), cmpf);
    cudaEventDestroy(a); cudaEventDestroy(b);
    CHK(cudaGetLastError());
    return t[reps / 2];
}

static float *g_ref = nullptr;              // cuBLAS answer, for correctness
static float *g_hC = nullptr;

static float max_err(const float *dC) {
    CHK(cudaMemcpy(g_hC, dC, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
    float e = 0.f;
    for (long i = 0; i < (long)N * N; ++i) {
        float d = fabsf(g_hC[i] - g_ref[i]);
        float s = fabsf(g_ref[i]) + 1e-6f;
        if (d / s > e) e = d / s;
    }
    return e;
}

// occupancy for a kernel, as a fraction of the SM's warp capacity
template <class K>
static double occupancy(K kern, int threads, size_t smem, const cudaDeviceProp &p) {
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, threads, smem);
    return (double)nb * threads / p.maxThreadsPerMultiProcessor;
}

static void report(const char *name, int tile, int tm, int tn, size_t smem,
                   int threads, double occ, double ai, float ms, float err) {
    double flops = 2.0 * N * N * (double)N;
    printf("cfg,%s,%d,%d,%d,%zu,%d,%.4f,%.3f,%.6f,%.2f,%.3e\n",
           name, tile, tm, tn, smem, threads, occ, ai, ms,
           flops / (ms * 1e-3) / 1e12, err);
}

#define RUN_REG(BM, BN, BK, TM, TN)                                            \
    do {                                                                       \
        constexpr int NT = (BM / TM) * (BN / TN);                              \
        constexpr size_t SM = ((size_t)BK * BM + (size_t)BK * BN) * sizeof(float); \
        dim3 g(N / BN, N / BM);                                                \
        auto kk = k_reg<BM, BN, BK, TM, TN>;                                   \
        CHK(cudaMemset(dC, 0, (size_t)N * N * sizeof(float)));                 \
        float ms = time_ms([&] { kk<<<g, NT>>>(dA, dB, dC, N); });             \
        char nm[64];                                                           \
        snprintf(nm, sizeof nm, "reg_%dx%d_k%d_%dx%d", BM, BN, BK, TM, TN);    \
        report(nm, BM, TM, TN, SM, NT,                                         \
               occupancy(kk, NT, SM, p),                                       \
               (double)BM * BN / (2.0 * (BM + BN)), ms, max_err(dC));          \
    } while (0)

#define RUN_SMEM(T)                                                            \
    do {                                                                       \
        dim3 b(T, T), g(N / T, N / T);                                         \
        auto kk = k_smem<T>;                                                   \
        constexpr size_t SM = 2ull * T * T * sizeof(float);                    \
        CHK(cudaMemset(dC, 0, (size_t)N * N * sizeof(float)));                 \
        float ms = time_ms([&] { kk<<<g, b>>>(dA, dB, dC, N); });              \
        char nm[64];                                                           \
        snprintf(nm, sizeof nm, "smem_%dx%d", T, T);                           \
        report(nm, T, 1, 1, SM, T * T, occupancy(kk, T * T, SM, p),            \
               T / 4.0, ms, max_err(dC));                                      \
    } while (0)

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("#device,%s,%d.%d,%d,%zu,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, (size_t)p.sharedMemPerBlock, N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC, *dRef, *out;
    CHK(cudaMalloc(&dA, bytes)); CHK(cudaMalloc(&dB, bytes));
    CHK(cudaMalloc(&dC, bytes)); CHK(cudaMalloc(&dRef, bytes));
    CHK(cudaMalloc(&out, 1024 * sizeof(float)));
    float *hA = (float *)malloc(bytes);
    g_hC = (float *)malloc(bytes);
    g_ref = (float *)malloc(bytes);
    unsigned st = 12345u;
    auto rnd = [&] { st = st * 1103515245u + 12345u; return (float)((st >> 9) & 0xFFFF) / 65536.f - 0.5f; };
    for (long i = 0; i < (long)N * N; ++i) hA[i] = rnd();
    CHK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    for (long i = 0; i < (long)N * N; ++i) hA[i] = rnd();
    CHK(cudaMemcpy(dB, hA, bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < 250; ++i) k_spin<<<19 * 8, 256>>>(out, 200000);
    CHK(cudaDeviceSynchronize());

    // ---- cuBLAS: both the reference answer and the speed to beat ----
    cublasHandle_t h; cublasCreate(&h);
    float al = 1.f, be = 0.f;
    // cuBLAS is column-major; computing B^T * A^T in its view gives our C.
    auto blas = [&] {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, dB, N, dA, N,
                    &be, dRef, N);
    };
    float ms_blas = time_ms(blas);
    CHK(cudaMemcpy(g_ref, dRef, bytes, cudaMemcpyDeviceToHost));
    double flops = 2.0 * N * N * (double)N;
    printf("cfg,cublas,0,0,0,0,0,0.0000,0.000,%.6f,%.2f,%.3e\n",
           ms_blas, flops / (ms_blas * 1e-3) / 1e12, 0.0);

    // ---- naive ----
    {
        dim3 b(32, 8), g(N / 32, N / 8);
        CHK(cudaMemset(dC, 0, bytes));
        float ms = time_ms([&] { k_naive<<<g, b>>>(dA, dB, dC, N); });
        report("naive", 1, 1, 1, 0, 256, occupancy(k_naive, 256, 0, p), 0.25,
               ms, max_err(dC));
    }

    // ---- one level: shared tile only ----
    RUN_SMEM(8);
    RUN_SMEM(16);
    RUN_SMEM(32);

    // ---- the wall: how far can ONE level of tiling go? ----
    // T=64 needs 32 KB of shared memory (fine) but 64x64 = 4096 threads per
    // block, and the hardware maximum is 1024. T=128 would need 128 KB of
    // shared memory against a 48 KB budget and does not even compile. This is
    // a hard architectural ceiling, not a tuning problem.
    {
        dim3 b(64, 64), g(N / 64, N / 64);
        k_smem<64><<<g, b>>>(dA, dB, dC, N);
        cudaError_t e = cudaGetLastError();
        printf("wall,64,%d,%d,%d,%s\n", 2 * 64 * 64 * 4, 64 * 64,
               p.maxThreadsPerBlock, cudaGetErrorString(e));
        cudaGetLastError();
    }

    // ---- two levels: shared tile + register patch ----
    RUN_REG(32, 32, 32, 1, 1);       // same tile as smem_32, written the other way
    RUN_REG(64, 64, 16, 2, 2);
    RUN_REG(64, 64, 16, 4, 4);
    RUN_REG(64, 64, 8, 4, 4);
    RUN_REG(128, 128, 8, 4, 4);
    RUN_REG(128, 128, 8, 8, 8);
    RUN_REG(128, 128, 16, 8, 8);
    RUN_REG(128, 64, 8, 8, 4);
    RUN_REG(256, 128, 8, 8, 8);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(out);
    free(hA); free(g_hC); free(g_ref);
    return 0;
}
