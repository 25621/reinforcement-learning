// Project 24 - what a warp size is worth, and why porting it hurts.
//
// A branch costs nothing as long as every thread in the same warp takes the
// same side of it. The moment one warp contains threads going both ways, the
// hardware runs BOTH sides and masks off the threads that should not be
// executing -- so the warp pays for both.
//
// This measures that: threads are split into alternating groups of G, group 0
// takes branch A, group 1 takes branch B, and so on. G is a RUNTIME argument
// so the compiler cannot specialise the branch away.
//
//   G  < warpSize  ->  every warp is mixed        -> ~2x
//   G >= warpSize  ->  every warp is pure         -> ~1x
//
// The port hazard: warpSize is 32 on NVIDIA and 64 on AMD. A kernel tuned so
// that G = 32 is divergence-free here becomes fully divergent on an MI300X,
// and nothing in the source changed.
//
// This file is also the subject of the hipify round-trip test in run.py: it is
// ported to HIP by hipify.py, compiled back through hipshim/, and its output
// must match this one exactly.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define CK(call)                                                              \
    do {                                                                      \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    hipGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void k_div(float *out, int n, int G, int iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 1e-6f;
    if (((tid / G) & 1) == 0) {
        for (int i = 0; i < iters; ++i) x = x * 1.0000001f + 1e-7f;
    } else {
        for (int i = 0; i < iters; ++i) x = x * 0.9999999f - 1e-7f;
    }
    if (tid < n) out[tid] = x;
}

// the control: identical arithmetic, but every thread takes the same side
__global__ void k_pure(float *out, int n, int iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = tid * 1e-6f;
    for (int i = 0; i < iters; ++i) x = x * 1.0000001f + 1e-7f;
    if (tid < n) out[tid] = x;
}

__global__ void k_spin(float *out, int iters)
{
    float x = threadIdx.x * 1e-6f;
    for (int i = 0; i < iters; ++i) x = x * 1.0000001f + 1e-7f;
    out[threadIdx.x] = x;
}

static void warm(float *d)
{
    for (int i = 0; i < 250; ++i) k_spin<<<152, 128>>>(d, 20000);
    CK(hipDeviceSynchronize());
}

static float bench_div(float *d, int n, int blocks, int threads, int G,
                       int iters, int reps)
{
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) k_div<<<blocks, threads>>>(d, n, G, iters);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) k_div<<<blocks, threads>>>(d, n, G, iters);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return ms / reps;
}

static float bench_pure(float *d, int n, int blocks, int threads, int iters,
                        int reps)
{
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) k_pure<<<blocks, threads>>>(d, n, iters);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) k_pure<<<blocks, threads>>>(d, n, iters);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return ms / reps;
}

int main(void)
{
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));

    const int threads = 256;
    const int blocks = p.multiProcessorCount * 8;
    const int n = blocks * threads;
    const int iters = 4000;
    const int reps = 20;

    float *d = NULL;
    CK(hipMalloc(&d, (size_t)n * sizeof(float)));

    printf("prop,name,%s\n", p.name);
    printf("prop,warpSize,%d\n", p.warpSize);
    printf("prop,sms,%d\n", p.multiProcessorCount);

    warm(d);
    float base = bench_pure(d, n, blocks, threads, iters, reps);
    printf("pure,0,%.6f\n", base);

    int Gs[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 9; ++i) {
        warm(d);
        float ms = bench_div(d, n, blocks, threads, Gs[i], iters, reps);
        printf("div,%d,%.6f\n", Gs[i], ms);
    }

    CK(hipFree(d));
    return 0;
}
