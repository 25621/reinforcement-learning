// Project 17 - climbing from a naive matmul to something that can stand next
// to cuBLAS, one idea per rung.
//
//   1 naive        one output per thread, every operand read from global
//   2 smem         32x32 shared-memory tile, still one output per thread
//   3 tile1d       64x64 block tile, 8 outputs per thread (one column)
//   4 tile2d       128x128 block tile, 8x8 outputs per thread
//   5 vec          same as 4 plus float4 loads and a transposed A tile
//   6 cuBLAS       the reference
//
// Prints CSV on stdout; run.py parses, tabulates, plots.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CB(call)                                                              \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s,            \
                    __FILE__, __LINE__);                                      \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ------------------------------------------------------------ 1. naive
// Thread (x, y) owns C[y][x]. threadIdx.x runs along the columns, so the 32
// threads of a warp read 32 adjacent floats of B and write 32 adjacent floats
// of C - the loads are coalesced. A is broadcast (all lanes read the same
// element), which the hardware also serves in one transaction.
__global__ void sgemm_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------ 2. smem tile
// Stage a T x T tile of A and of B in shared memory; every element loaded is
// then used T times instead of once.
template <int T>
__global__ void sgemm_smem(int M, int N, int K, const float *A,
                           const float *B, float *C) {
    __shared__ float As[T][T], Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int t = 0; t < K; t += T) {
        As[ty][tx] = A[row * K + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ------------------------------------------------------------ 3. 1D tiling
// Each thread now owns TM outputs stacked in a column. One value of B loaded
// into a register is reused for all TM of them.
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_tile1d(int M, int N, int K, const float *A,
                             const float *B, float *C) {
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;
    const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM] = {0.0f};
    for (int bk = 0; bk < K; bk += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            float b = Bs[d * BN + threadCol];
            for (int i = 0; i < TM; ++i)
                acc[i] += As[(threadRow * TM + i) * BK + d] * b;
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        C[(threadRow * TM + i) * N + threadCol] = acc[i];
}

// ------------------------------------------------------------ 4. 2D tiling
// Each thread owns a TM x TN square of C. TM + TN loads from shared memory
// feed TM * TN multiply-adds - the second level of tiling, held in registers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tile2d(int M, int N, int K, const float *A,
                             const float *B, float *C) {
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int nThreads = (BM * BN) / (TM * TN);
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
    const int strideA = nThreads / BK;
    const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;
    const int strideB = nThreads / BN;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        for (int o = 0; o < BM; o += strideA)
            As[(innerRowA + o) * BK + innerColA] =
                A[(innerRowA + o) * K + innerColA];
        for (int o = 0; o < BK; o += strideB)
            Bs[(innerRowB + o) * BN + innerColB] =
                B[(innerRowB + o) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + d];
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[d * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}

// ------------------------------------------------------------ 5. vectorised
// Same shape as 4. Two changes: loads move 16 bytes at a time (float4), and
// the A tile is stored transposed so the inner loop reads it contiguously.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vec(int M, int N, int K, const float *A, const float *B,
                          float *C) {
    __shared__ float As[BK * BM];          // transposed: [k][m]
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
    const int nThreads = (BM * BN) / (TM * TN);
    const int strideB = nThreads / (BN / 4);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        float4 a = *reinterpret_cast<const float4 *>(
            &A[innerRowA * K + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;
        for (int o = 0; o < BK; o += strideB)
            *reinterpret_cast<float4 *>(
                &Bs[(innerRowB + o) * BN + innerColB * 4]) =
                *reinterpret_cast<const float4 *>(
                    &B[(innerRowB + o) * N + innerColB * 4]);
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int d = 0; d < BK; ++d) {
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4 *>(&regM[i]) =
                    *reinterpret_cast<float4 *>(&As[d * BM + threadRow * TM + i]);
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4 *>(&regN[j]) =
                    *reinterpret_cast<float4 *>(&Bs[d * BN + threadCol * TN + j]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4)
            *reinterpret_cast<float4 *>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j]) =
                *reinterpret_cast<float4 *>(&acc[i * TN + j]);
}

// ------------------------------------------------------------ warm-up
__global__ void k_spin(float *sink, int iters) {
    float x = threadIdx.x * 1e-6f;
    for (int i = 0; i < iters; ++i) x = fmaf(x, 1.0000001f, 1e-7f);
    if (x == 12345.0f) sink[0] = x;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

template <typename F>
static float time_ms(F f, int reps) {
    Timer t;
    f();
    CK(cudaDeviceSynchronize());
    t.start();
    for (int i = 0; i < reps; ++i) f();
    float ms = t.stop() / reps;
    CK(cudaGetLastError());
    return ms;
}

// ------------------------------------------------------------ driver

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("#device,%s,%d.%d,%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.l2CacheSize, p.sharedMemPerBlock);

    const int sizes[] = {1024, 2048, 4096};
    const int NS = 3;
    const int MAXN = 4096;

    float *hA = (float *)malloc((size_t)MAXN * MAXN * 4);
    float *hC = (float *)malloc((size_t)MAXN * MAXN * 4);
    float *hR = (float *)malloc((size_t)MAXN * MAXN * 4);
    for (size_t i = 0; i < (size_t)MAXN * MAXN; ++i)
        hA[i] = (float)((i * 1103515245u + 12345u) % 1000) * 0.001f - 0.5f;

    float *dA, *dB, *dC, *dR, *sink;
    CK(cudaMalloc(&dA, (size_t)MAXN * MAXN * 4));
    CK(cudaMalloc(&dB, (size_t)MAXN * MAXN * 4));
    CK(cudaMalloc(&dC, (size_t)MAXN * MAXN * 4));
    CK(cudaMalloc(&dR, (size_t)MAXN * MAXN * 4));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemcpy(dA, hA, (size_t)MAXN * MAXN * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hA, (size_t)MAXN * MAXN * 4, cudaMemcpyHostToDevice));

    cublasHandle_t h;
    CB(cublasCreate(&h));
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < 250; ++i) k_spin<<<152, 256>>>(sink, 200000);
    CK(cudaDeviceSynchronize());

    for (int s = 0; s < NS; ++s) {
        int n = sizes[s];
        double flop = 2.0 * n * n * n;
        int reps = (n <= 1024) ? 20 : (n <= 2048 ? 8 : 3);
        int reps_slow = (n <= 1024) ? 5 : (n <= 2048 ? 3 : 2);

        // cuBLAS reference (row-major C = A*B via a column-major B*A call)
        float ms_cublas = time_ms([&] {
            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n,
                           dA, n, &beta, dR, n));
        }, reps);
        printf("k,%d,cublas,%.4f,%.2f,%.3e\n", n, ms_cublas,
               flop / (ms_cublas * 1e6), 0.0);
        CK(cudaMemcpy(hR, dR, (size_t)n * n * 4, cudaMemcpyDeviceToHost));

        struct Run { const char *name; float ms; };
        Run runs[5];

        {   // 1. naive
            dim3 blk(32, 32), grd(n / 32, n / 32);
            runs[0] = {"naive",
                       time_ms([&] { sgemm_naive<<<grd, blk>>>(n, n, n, dA, dB, dC); },
                               reps_slow)};
        }
        CK(cudaMemcpy(hC, dC, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
        double err0 = 0;
        for (size_t i = 0; i < (size_t)n * n; ++i)
            err0 = fmax(err0, fabs((double)hC[i] - hR[i]));
        printf("k,%d,naive,%.4f,%.2f,%.3e\n", n, runs[0].ms,
               flop / (runs[0].ms * 1e6), err0);

        {   // 2. shared tile
            dim3 blk(32, 32), grd(n / 32, n / 32);
            runs[1] = {"smem",
                       time_ms([&] { sgemm_smem<32><<<grd, blk>>>(n, n, n, dA, dB, dC); },
                               reps_slow)};
        }
        CK(cudaMemcpy(hC, dC, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
        double err1 = 0;
        for (size_t i = 0; i < (size_t)n * n; ++i)
            err1 = fmax(err1, fabs((double)hC[i] - hR[i]));
        printf("k,%d,smem,%.4f,%.2f,%.3e\n", n, runs[1].ms,
               flop / (runs[1].ms * 1e6), err1);

        {   // 3. 1D thread tiling
            dim3 grd(n / 64, n / 64);
            runs[2] = {"tile1d",
                       time_ms([&] {
                           sgemm_tile1d<64, 64, 8, 8><<<grd, 512>>>(n, n, n, dA, dB, dC);
                       }, reps)};
        }
        CK(cudaMemcpy(hC, dC, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
        double err2 = 0;
        for (size_t i = 0; i < (size_t)n * n; ++i)
            err2 = fmax(err2, fabs((double)hC[i] - hR[i]));
        printf("k,%d,tile1d,%.4f,%.2f,%.3e\n", n, runs[2].ms,
               flop / (runs[2].ms * 1e6), err2);

        {   // 4. 2D thread tiling
            dim3 grd(n / 128, n / 128);
            runs[3] = {"tile2d",
                       time_ms([&] {
                           sgemm_tile2d<128, 128, 8, 8, 8><<<grd, 256>>>(n, n, n, dA, dB, dC);
                       }, reps)};
        }
        CK(cudaMemcpy(hC, dC, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
        double err3 = 0;
        for (size_t i = 0; i < (size_t)n * n; ++i)
            err3 = fmax(err3, fabs((double)hC[i] - hR[i]));
        printf("k,%d,tile2d,%.4f,%.2f,%.3e\n", n, runs[3].ms,
               flop / (runs[3].ms * 1e6), err3);

        {   // 5. vectorised
            dim3 grd(n / 128, n / 128);
            runs[4] = {"vec",
                       time_ms([&] {
                           sgemm_vec<128, 128, 8, 8, 8><<<grd, 256>>>(n, n, n, dA, dB, dC);
                       }, reps)};
        }
        CK(cudaMemcpy(hC, dC, (size_t)n * n * 4, cudaMemcpyDeviceToHost));
        double err4 = 0;
        for (size_t i = 0; i < (size_t)n * n; ++i)
            err4 = fmax(err4, fabs((double)hC[i] - hR[i]));
        printf("k,%d,vec,%.4f,%.2f,%.3e\n", n, runs[4].ms,
               flop / (runs[4].ms * 1e6), err4);
        fflush(stdout);
    }

    // register and shared-memory cost of each kernel, straight from the driver
    struct Att { const char *name; const void *fn; };
    Att atts[] = {
        {"naive", (const void *)sgemm_naive},
        {"smem", (const void *)sgemm_smem<32>},
        {"tile1d", (const void *)sgemm_tile1d<64, 64, 8, 8>},
        {"tile2d", (const void *)sgemm_tile2d<128, 128, 8, 8, 8>},
        {"vec", (const void *)sgemm_vec<128, 128, 8, 8, 8>},
    };
    int threads[] = {1024, 1024, 512, 256, 256};
    for (int i = 0; i < 5; ++i) {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, atts[i].fn));
        int blocks = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, atts[i].fn,
                                                         threads[i], 0));
        double occ = (double)blocks * threads[i] / 32.0 /
                     (p.maxThreadsPerMultiProcessor / 32.0);
        printf("attr,%s,%d,%d,%zu,%d,%.4f\n", atts[i].name, threads[i],
               fa.numRegs, fa.sharedSizeBytes, blocks, occ);
    }

    CB(cublasDestroy(h));
    return 0;
}
