// Project 02 - measure the five operations the roofline model predicts about.
//
// Compiled by run.py with:
//   nvcc -O3 -arch=sm_61 bench_ops.cu -o bench_ops -lcublas
//
// Prints one CSV line per operation:
//   name,seconds,flops,bytes
// run.py does all the arithmetic and the plotting; this file only measures.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("#CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

// ---------------------------------------------------------------- kernels
// LayerNorm over the last dimension. One block per row; the row lives in
// shared memory so it is read from global memory exactly once.
__global__ void layernorm_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 int D, float eps) {
    extern __shared__ float smem[];
    const float* row = x + (size_t)blockIdx.x * D;
    float* orow = y + (size_t)blockIdx.x * D;

    float local = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = row[i]; smem[i] = v; local += v; }
    __shared__ float red[32];
    // block reduction for the sum
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int i = 0; i < blockDim.x / 32; ++i) s += red[i]; red[31] = s / D; }
    __syncthreads();
    const float mean = red[31];

    float lv = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float d = smem[i] - mean; lv += d * d; }
    for (int off = 16; off > 0; off >>= 1) lv += __shfl_down_sync(0xffffffff, lv, off);
    __shared__ float red2[32];
    if ((threadIdx.x & 31) == 0) red2[threadIdx.x >> 5] = lv;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int i = 0; i < blockDim.x / 32; ++i) s += red2[i]; red2[31] = rsqrtf(s / D + eps); }
    __syncthreads();
    const float rstd = red2[31];

    for (int i = threadIdx.x; i < D; i += blockDim.x) orow[i] = (smem[i] - mean) * rstd;
}

// Softmax over the last dimension, same block-per-row layout.
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
    extern __shared__ float smem[];
    const float* row = x + (size_t)blockIdx.x * D;
    float* orow = y + (size_t)blockIdx.x * D;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = row[i]; smem[i] = v; m = fmaxf(m, v); }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float s = -INFINITY; for (int i = 0; i < blockDim.x / 32; ++i) s = fmaxf(s, red[i]); red[31] = s; }
    __syncthreads();
    const float mx = red[31];

    float ls = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float e = __expf(smem[i] - mx); smem[i] = e; ls += e; }
    for (int off = 16; off > 0; off >>= 1) ls += __shfl_down_sync(0xffffffff, ls, off);
    __shared__ float red2[32];
    if ((threadIdx.x & 31) == 0) red2[threadIdx.x >> 5] = ls;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0.f; for (int i = 0; i < blockDim.x / 32; ++i) s += red2[i]; red2[31] = 1.f / s; }
    __syncthreads();
    const float inv = red2[31];

    for (int i = threadIdx.x; i < D; i += blockDim.x) orow[i] = smem[i] * inv;
}

// GELU, tanh approximation. Pure elementwise, grid-stride loop.
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, size_t n) {
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + tanhf(k0 * (v + k1 * v * v * v)));
    }
}

// Naive transpose: reads are coalesced, writes are strided by N.
__global__ void transpose_kernel(const float* __restrict__ x, float* __restrict__ y, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < N && row < N) y[(size_t)col * N + row] = x[(size_t)row * N + col];
}

// ---------------------------------------------------------------- harness
struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("#device,%s,%d.%d,%d,%d,%d,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate, p.memoryClockRate, p.memoryBusWidth);

    Timer t;
    const int ITERS = 20, WARM = 5;

    // ---- 1. matmul via cuBLAS ----
    {
        const int N = 4096;
        float *A, *B, *C;
        CK(cudaMalloc(&A, (size_t)N * N * 4)); CK(cudaMalloc(&B, (size_t)N * N * 4));
        CK(cudaMalloc(&C, (size_t)N * N * 4));
        CK(cudaMemset(A, 1, (size_t)N * N * 4)); CK(cudaMemset(B, 1, (size_t)N * N * 4));
        cublasHandle_t h; cublasCreate(&h);
        const float al = 1.f, be = 0.f;
        for (int i = 0; i < WARM; ++i)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, A, N, B, N, &be, C, N);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < ITERS; ++i)
                cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &al, A, N, B, N, &be, C, N);
            float ms = t.stop() / ITERS; if (ms < best) best = ms;
        }
        printf("matmul,%d,%.6e,%.0f,%.0f\n", N, best / 1e3,
               2.0 * N * N * N, 3.0 * N * N * 4);
        cublasDestroy(h); cudaFree(A); cudaFree(B); cudaFree(C);
    }

    // ---- shared buffers for the row-wise ops ----
    const int M = 8192, D = 4096;           // 8192 rows of 4096 floats = 128 MB
    const size_t nrow = (size_t)M * D;
    float *X, *Y;
    CK(cudaMalloc(&X, nrow * 4)); CK(cudaMalloc(&Y, nrow * 4));
    {   // fill with something non-degenerate
        float* host = (float*)malloc(nrow * 4);
        for (size_t i = 0; i < nrow; ++i) host[i] = (float)((i % 101) - 50) * 0.03f;
        CK(cudaMemcpy(X, host, nrow * 4, cudaMemcpyHostToDevice));
        free(host);
    }
    const int TPB = 256;
    const size_t shm = (size_t)D * 4;

    // ---- 2. layernorm ----
    {
        for (int i = 0; i < WARM; ++i) layernorm_kernel<<<M, TPB, shm>>>(X, Y, D, 1e-5f);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < ITERS; ++i) layernorm_kernel<<<M, TPB, shm>>>(X, Y, D, 1e-5f);
            float ms = t.stop() / ITERS; if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("layernorm,%d,%.6e,%.0f,%.0f\n", D, best / 1e3,
               8.0 * nrow, 2.0 * nrow * 4);
    }

    // ---- 3. softmax ----
    {
        for (int i = 0; i < WARM; ++i) softmax_kernel<<<M, TPB, shm>>>(X, Y, D);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < ITERS; ++i) softmax_kernel<<<M, TPB, shm>>>(X, Y, D);
            float ms = t.stop() / ITERS; if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("softmax,%d,%.6e,%.0f,%.0f\n", D, best / 1e3,
               5.0 * nrow, 2.0 * nrow * 4);
    }

    // ---- 4. gelu ----
    {
        int blocks = p.multiProcessorCount * 32;
        for (int i = 0; i < WARM; ++i) gelu_kernel<<<blocks, TPB>>>(X, Y, nrow);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < ITERS; ++i) gelu_kernel<<<blocks, TPB>>>(X, Y, nrow);
            float ms = t.stop() / ITERS; if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("gelu,%zu,%.6e,%.0f,%.0f\n", nrow, best / 1e3,
               8.0 * nrow, 2.0 * nrow * 4);
    }

    // ---- 5. transpose ----
    {
        const int N = 8192;                 // 8192^2 floats = 256 MB in, 256 MB out
        float *A, *B;
        CK(cudaMalloc(&A, (size_t)N * N * 4)); CK(cudaMalloc(&B, (size_t)N * N * 4));
        CK(cudaMemset(A, 1, (size_t)N * N * 4));
        dim3 blk(32, 8), grd((N + 31) / 32, (N + 7) / 8);
        for (int i = 0; i < WARM; ++i) transpose_kernel<<<grd, blk>>>(A, B, N);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < ITERS; ++i) transpose_kernel<<<grd, blk>>>(A, B, N);
            float ms = t.stop() / ITERS; if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("transpose,%d,%.6e,%.0f,%.0f\n", N, best / 1e3,
               0.0, 2.0 * (double)N * N * 4);
        cudaFree(A); cudaFree(B);
    }

    cudaFree(X); cudaFree(Y);
    return 0;
}
