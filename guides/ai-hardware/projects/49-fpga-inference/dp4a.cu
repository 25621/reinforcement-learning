// dp4a.cu - what this GPU does with int8, measured, so the FPGA comparison is
// against a real number instead of a spec sheet.
//
// __dp4a(a, b, c) takes two packed vectors of 4 int8 values, multiplies them
// element-wise, and adds the four products plus c into an int32. That is 8
// integer operations in one instruction. Pascal (sm_61) added it precisely
// because neural-network inference wanted int8 throughput without tensor
// cores.
//
// Prints: dp4a,<GOPS>   and   fp32,<GFLOPS>
// Build: nvcc -O3 -arch=sm_61 dp4a.cu -o dp4a

#include <cstdio>
#include <cuda_runtime.h>

__global__ void dp4a_bench(int* sink, int trips) {
    int a = 0x01020304 + threadIdx.x, b = 0x04030201, acc0 = 0, acc1 = 0,
        acc2 = 0, acc3 = 0;
    for (int t = 0; t < trips; t++) {
#pragma unroll
        for (int k = 0; k < 32; k++) {
            acc0 = __dp4a(a, b, acc0);
            acc1 = __dp4a(a, b, acc1);
            acc2 = __dp4a(a, b, acc2);
            acc3 = __dp4a(a, b, acc3);
        }
    }
    if (acc0 + acc1 + acc2 + acc3 == 12345) sink[0] = acc0;
}

__global__ void fma_bench(float* sink, int trips) {
    float a = threadIdx.x * 1e-3f + 1.f, b = 1.0000001f, c = 0.9999999f;
    float x0 = a, x1 = a + 1.f, x2 = a + 2.f, x3 = a + 3.f;
    for (int t = 0; t < trips; t++) {
#pragma unroll
        for (int k = 0; k < 32; k++) {
            x0 = fmaf(x0, b, c); x1 = fmaf(x1, b, c);
            x2 = fmaf(x2, b, c); x3 = fmaf(x3, b, c);
        }
    }
    if (x0 + x1 + x2 + x3 == 1.2345f) sink[0] = x0;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    const int threads = 256, blocks = p.multiProcessorCount * 8, trips = 4000;
    void* sink; cudaMalloc(&sink, 16);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;

    // int8: 4 multiplies + 4 adds per __dp4a = 8 ops; fp32: 2 flops per fma.
    // The two are timed in ALTERNATING rounds, not one after the other: this
    // card's boost clock drifts with temperature, so measuring all of A then
    // all of B would charge B for whatever the clock did in between.
    double ops = (double)blocks * threads * trips * 32.0 * 4.0 * 8.0;
    double flops = (double)blocks * threads * trips * 32.0 * 4.0 * 2.0;
    dp4a_bench<<<blocks, threads>>>((int*)sink, 10);
    fma_bench<<<blocks, threads>>>((float*)sink, 10);
    cudaDeviceSynchronize();

    double best_i8 = 0, best_f32 = 0;
    for (int r = 0; r < 5; r++) {
        cudaEventRecord(t0);
        dp4a_bench<<<blocks, threads>>>((int*)sink, trips);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        best_i8 = fmax(best_i8, ops / (ms * 1e-3) / 1e9);

        cudaEventRecord(t0);
        fma_bench<<<blocks, threads>>>((float*)sink, trips);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        best_f32 = fmax(best_f32, flops / (ms * 1e-3) / 1e9);
    }
    printf("dp4a,%.2f\n", best_i8);
    printf("fp32,%.2f\n", best_f32);
    return 0;
}
