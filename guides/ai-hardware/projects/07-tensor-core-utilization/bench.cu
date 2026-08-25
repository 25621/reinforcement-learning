// Project 07 - how much of a GPU's math pipeline a matmul actually uses.
//
// The guide asks for `ncu` + `nsys` on a matmul. On this machine both are
// unavailable (see the README), and this GPU predates Tensor Cores anyway.
// So we measure the same quantity from first principles instead:
//
//     pipe utilization = achieved ops/sec / peak ops/sec
//
// which is exactly what Nsight Compute's `pct_of_peak_sustained` counters
// report, just computed rather than read off a hardware counter.
//
// Four matmuls of the same shape:
//   naive fp32   - one thread per output, everything from global memory
//   tiled fp32   - shared-memory tiles (the classic optimisation)
//   cuBLAS fp32  - NVIDIA's own SGEMM
//   dp4a int8    - the sm_61 4-way integer dot-product instruction, which is
//                  the direct ancestor of the Tensor Core: one instruction
//                  that does a whole little dot product instead of one MAC
//   cuBLAS int8  - NVIDIA's integer GEMM (uses dp4a under the hood)
//
// Plus a timeline breakdown (copy in / compute / copy out) at several sizes,
// which is the question `nsys` answers.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 32
#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------- fp32 naive
__global__ void mm_naive(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < N; ++k) acc = fmaf(A[row * N + k], B[k * N + col], acc);
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------- fp32 tiled
__global__ void mm_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int kt = 0; kt < N; kt += TILE) {
        As[ty][tx] = A[row * N + kt + tx];
        Bs[ty][tx] = B[(kt + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------- int8 dp4a
// B is passed already transposed (Bt[col][k]) so that the four k-values a
// single dp4a consumes are adjacent in memory for BOTH operands. This layout
// requirement is not an accident of our code - every matrix instruction, dp4a
// and Tensor Core alike, dictates how its operands must be laid out.
__global__ void mm_dp4a(const char *A, const char *Bt, int *C, int N) {
    __shared__ char As[TILE][TILE + 4];
    __shared__ char Bs[TILE][TILE + 4];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    int acc = 0;
    for (int kt = 0; kt < N; kt += TILE) {
        // As[i][j] = A[(blockRow+i)][kt+j];  Bs[i][j] = Bt[(blockCol+i)][kt+j]
        As[ty][tx] = A[row * N + kt + tx];
        Bs[ty][tx] = Bt[(blockIdx.x * TILE + ty) * N + kt + tx];
        __syncthreads();
        // 32 k-values, 4 at a time -> 8 dp4a instructions instead of 32 FMAs
        #pragma unroll
        for (int j = 0; j < TILE / 4; ++j) {
            int a4 = *(const int *)&As[ty][j * 4];
            int b4 = *(const int *)&Bs[tx][j * 4];
            acc = __dp4a(a4, b4, acc);
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// --------------------------------------------------------------------------
static float time_ms(void (*launch)(void *), void *ctx, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(ctx); CHK(cudaDeviceSynchronize());          // warm up
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch(ctx);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

struct Ctx {
    int N; float *dA, *dB, *dC; char *dAi, *dBt; int *dCi;
    cublasHandle_t h;
};
static Ctx g;

static void l_naive(void *) {
    dim3 t(16, 16), b((g.N + 15) / 16, (g.N + 15) / 16);
    mm_naive<<<b, t>>>(g.dA, g.dB, g.dC, g.N);
}
static void l_tiled(void *) {
    dim3 t(TILE, TILE), b(g.N / TILE, g.N / TILE);
    mm_tiled<<<b, t>>>(g.dA, g.dB, g.dC, g.N);
}
static void l_dp4a(void *) {
    dim3 t(TILE, TILE), b(g.N / TILE, g.N / TILE);
    mm_dp4a<<<b, t>>>(g.dAi, g.dBt, g.dCi, g.N);
}
static void l_sgemm(void *) {
    float al = 1.f, be = 0.f;
    cublasSgemm(g.h, CUBLAS_OP_N, CUBLAS_OP_N, g.N, g.N, g.N,
                &al, g.dB, g.N, g.dA, g.N, &be, g.dC, g.N);
}
static bool igemm_ok = true;
static void l_igemm(void *) {
    int al = 1, be = 0;
    cublasStatus_t s = cublasGemmEx(
        g.h, CUBLAS_OP_T, CUBLAS_OP_N, g.N, g.N, g.N,
        &al, g.dBt, CUDA_R_8I, g.N, g.dAi, CUDA_R_8I, g.N,
        &be, g.dCi, CUDA_R_32I, g.N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) igemm_ok = false;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 2048;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);

    printf("#device,%s,%d.%d,%d,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate);
    printf("#tensor_cores,%d\n", p.major >= 7 ? 1 : 0);

    size_t nf = (size_t)N * N;
    float *hA = (float *)malloc(nf * 4), *hB = (float *)malloc(nf * 4);
    char *hAi = (char *)malloc(nf), *hBt = (char *)malloc(nf);
    for (size_t i = 0; i < nf; ++i) {
        hA[i] = (float)((i * 1103515245u + 12345u) % 17) / 17.f - 0.5f;
        hB[i] = (float)((i * 22695477u + 1u) % 13) / 13.f - 0.5f;
        hAi[i] = (char)((i % 7) - 3);
    }
    // transposed int8 copy of B: hBt[col*N + k] = B[k*N + col]
    for (int k = 0; k < N; ++k)
        for (int c = 0; c < N; ++c)
            hBt[(size_t)c * N + k] = (char)((int)(((size_t)k * N + c) % 5) - 2);

    g.N = N;
    CHK(cudaMalloc(&g.dA, nf * 4)); CHK(cudaMalloc(&g.dB, nf * 4));
    CHK(cudaMalloc(&g.dC, nf * 4));
    CHK(cudaMalloc(&g.dAi, nf)); CHK(cudaMalloc(&g.dBt, nf));
    CHK(cudaMalloc(&g.dCi, nf * 4));
    CHK(cudaMemcpy(g.dA, hA, nf * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(g.dB, hB, nf * 4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(g.dAi, hAi, nf, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(g.dBt, hBt, nf, cudaMemcpyHostToDevice));
    cublasCreate(&g.h);

    int it = N <= 1024 ? 20 : 5;
    printf("mm,naive_fp32,%d,%.6f\n", N, time_ms(l_naive, 0, it));
    printf("mm,tiled_fp32,%d,%.6f\n", N, time_ms(l_tiled, 0, it));
    printf("mm,cublas_fp32,%d,%.6f\n", N, time_ms(l_sgemm, 0, it));
    printf("mm,dp4a_int8,%d,%.6f\n", N, time_ms(l_dp4a, 0, it));
    float ig = time_ms(l_igemm, 0, it);
    CHK(cudaDeviceSynchronize());
    if (igemm_ok) printf("mm,cublas_int8,%d,%.6f\n", N, ig);
    else printf("#cublas_int8_unsupported\n");

    // ---- correctness: dp4a kernel vs a CPU reference on a small corner ----
    int *hC = (int *)malloc(nf * 4);
    l_dp4a(0); CHK(cudaDeviceSynchronize());
    CHK(cudaMemcpy(hC, g.dCi, nf * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) {
        long ref = 0;
        for (int k = 0; k < N; ++k) ref += (long)hAi[(size_t)r * N + k] *
                                          (long)hBt[(size_t)c * N + k];
        if (ref != hC[(size_t)r * N + c]) ++bad;
    }
    printf("#dp4a_mismatches_in_8x8,%ld\n", bad);

    // ---- correctness: tiled and cuBLAS fp32 agree with the naive kernel ----
    float *hRef = (float *)malloc(nf * 4), *hAlt = (float *)malloc(nf * 4);
    l_naive(0); CHK(cudaDeviceSynchronize());
    CHK(cudaMemcpy(hRef, g.dC, nf * 4, cudaMemcpyDeviceToHost));
    const char *nm[2] = {"tiled_fp32", "cublas_fp32"};
    for (int v = 0; v < 2; ++v) {
        if (v == 0) l_tiled(0); else l_sgemm(0);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(hAlt, g.dC, nf * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (size_t i = 0; i < nf; ++i) {
            double d = fabs((double)hAlt[i] - hRef[i]);
            if (d > worst) worst = d;
        }
        printf("#maxabsdiff_vs_naive,%s,%.3e\n", nm[v], worst);
    }

    // ---- timeline breakdown: copy in / compute / copy out ----
    for (int n = 128; n <= 4096; n *= 2) {
        size_t b = (size_t)n * n * 4;
        float *a1, *b1, *c1;
        CHK(cudaMalloc(&a1, b)); CHK(cudaMalloc(&b1, b)); CHK(cudaMalloc(&c1, b));
        float *ha; CHK(cudaMallocHost(&ha, b));
        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventCreate(&e2); cudaEventCreate(&e3);
        float al = 1.f, be = 0.f;
        // warm up
        cublasSgemm(g.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, b1, n, a1, n,
                    &be, c1, n);
        CHK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        CHK(cudaMemcpy(a1, ha, b, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(b1, ha, b, cudaMemcpyHostToDevice));
        cudaEventRecord(e1);
        cublasSgemm(g.h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, b1, n, a1, n,
                    &be, c1, n);
        cudaEventRecord(e2);
        CHK(cudaMemcpy(ha, c1, b, cudaMemcpyDeviceToHost));
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);
        float h2d, comp, d2h;
        cudaEventElapsedTime(&h2d, e0, e1);
        cudaEventElapsedTime(&comp, e1, e2);
        cudaEventElapsedTime(&d2h, e2, e3);
        printf("timeline,%d,%.6f,%.6f,%.6f\n", n, h2d, comp, d2h);
        cudaFree(a1); cudaFree(b1); cudaFree(c1); cudaFreeHost(ha);
    }

    cublasDestroy(g.h);
    return 0;
}
