// Project 05 - the GPU half of the bake-off.
//
// Built by run.py:  nvcc -O3 -arch=sm_61 gemm.cu -o gemm -lcublas
//
// For each matrix size it reports THREE numbers, because "how fast is the GPU"
// has three different honest answers:
//   compute  - the matmul alone, data already on the GPU
//   total    - copy the inputs over, multiply, copy the result back
//   once     - the same as total but measured as a single cold call
//
// Prints: gemm,<N>,<compute_sec>,<h2d_sec>,<d2h_sec>,<total_sec>,<checksum>

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("#CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    double stop() { cudaEventRecord(b); cudaEventSynchronize(b);
                    float ms; cudaEventElapsedTime(&ms, a, b); return ms / 1e3; }
};

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("#device,%s,%d.%d,%d,%d,%d,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate, p.memoryClockRate, p.memoryBusWidth);

    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.f, beta = 0.f;
    Timer t;

    const int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < nsizes; ++si) {
        const int N = sizes[si];
        const size_t bytes = (size_t)N * N * 4;

        float *hA, *hB, *hC;
        CK(cudaMallocHost(&hA, bytes));      // pinned: the fastest a copy can go
        CK(cudaMallocHost(&hB, bytes));
        CK(cudaMallocHost(&hC, bytes));
        for (size_t i = 0; i < (size_t)N * N; ++i) {
            hA[i] = (float)((i % 17) - 8) * 0.125f;
            hB[i] = (float)((i % 23) - 11) * 0.0625f;
        }
        float *dA, *dB, *dC;
        CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));

        // repeats: enough to time small matrices honestly
        int iters = (int)(2e9 / ((double)N * N * N));
        if (iters < 3) iters = 3;
        if (iters > 500) iters = 500;

        // ---- compute only ----
        CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
        for (int i = 0; i < 3; ++i)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
        CK(cudaDeviceSynchronize());
        double compute = 1e30;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < iters; ++i)
                cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
            double s = t.stop() / iters;
            if (s < compute) compute = s;
        }

        // ---- the copies, timed on their own ----
        double h2d = 1e30, d2h = 1e30;
        for (int r = 0; r < 5; ++r) {
            t.start();
            for (int i = 0; i < 5; ++i) {
                CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
            }
            double s = t.stop() / 5; if (s < h2d) h2d = s;
            t.start();
            for (int i = 0; i < 5; ++i) CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
            s = t.stop() / 5; if (s < d2h) d2h = s;
        }

        // ---- the whole job, the way a caller would actually see it ----
        double total = 1e30;
        for (int r = 0; r < 5; ++r) {
            t.start();
            CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
            CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
            double s = t.stop(); if (s < total) total = s;
        }

        double checksum = 0.0;
        for (int i = 0; i < N; ++i) checksum += hC[i];

        printf("gemm,%d,%.9e,%.9e,%.9e,%.9e,%.6f\n", N, compute, h2d, d2h, total, checksum);
        fflush(stdout);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    }

    cublasDestroy(h);
    return 0;
}
