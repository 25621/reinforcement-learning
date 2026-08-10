// Project 03 - how many bytes per second can this GPU really move?
//
// Compiled by run.py:  nvcc -O3 -arch=sm_61 bandwidth.cu -o bandwidth
//
// Prints CSV rows. run.py turns them into GB/s, plots, and findings.
//   #device,name,cc,sms,clock_khz,mem_khz,bus_bits,l2_bytes,total_mem
//   kernel,<name>,<bytes_moved>,<seconds>
//   sweep,<name>,<buffer_bytes>,<bytes_moved>,<seconds>
//   pcie,<name>,<bytes_moved>,<seconds>

#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("#CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

// ------------------------------------------------------------------ kernels
// One float per thread, grid-stride loop. The plain, obvious version.
__global__ void copy_scalar(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

// Four floats per thread in ONE 16-byte instruction. Same bytes, fewer
// instructions, fewer memory requests in flight per byte.
__global__ void copy_float4(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

// Read only: sum the array. The write is one float per block, so the traffic
// is essentially all reads.
__global__ void read_only(const float4* __restrict__ in, float* __restrict__ sink, size_t n4) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5678f) sink[blockIdx.x] = acc;   // never true; stops the
                                                     // compiler deleting the loop
}

// Write only: fill the array, read nothing.
__global__ void write_only(float4* __restrict__ out, size_t n4, float v) {
    float4 val = make_float4(v, v, v, v);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = val;
}

// STREAM triad: a = b + s*c. Two reads, one write, one FMA.
__global__ void triad(const float4* __restrict__ b, const float4* __restrict__ c,
                      float4* __restrict__ a, size_t n4, float s) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 bb = b[i], cc = c[i];
        a[i] = make_float4(bb.x + s * cc.x, bb.y + s * cc.y,
                           bb.z + s * cc.z, bb.w + s * cc.w);
    }
}

// ------------------------------------------------------------------ harness
struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop_ms() { cudaEventRecord(b); cudaEventSynchronize(b);
                      float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static Timer T;

// Run `body` iters times, 4 rounds, keep the fastest round.
template <class F>
double best_seconds(F body, int iters) {
    for (int i = 0; i < 3; ++i) body();
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < 4; ++r) {
        T.start();
        for (int i = 0; i < iters; ++i) body();
        double s = T.stop_ms() / 1e3 / iters;
        if (s < best) best = s;
    }
    CK(cudaGetLastError());
    return best;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("#device,%s,%d.%d,%d,%d,%d,%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate, p.memoryClockRate, p.memoryBusWidth,
           p.l2CacheSize, p.totalGlobalMem);

    const int TPB = 256;
    const int BLOCKS = p.multiProcessorCount * 32;

    // ---- big buffers: 256 MB each, far larger than the 2 MB L2 ----
    const size_t N = 64u * 1024 * 1024;           // 64M floats = 256 MB
    const size_t N4 = N / 4;
    float *A, *B, *C;
    CK(cudaMalloc(&A, N * 4)); CK(cudaMalloc(&B, N * 4)); CK(cudaMalloc(&C, N * 4));
    CK(cudaMemset(A, 1, N * 4)); CK(cudaMemset(B, 1, N * 4)); CK(cudaMemset(C, 1, N * 4));

    double s;
    s = best_seconds([&]{ copy_scalar<<<BLOCKS, TPB>>>(A, B, N); }, 10);
    printf("kernel,copy_scalar,%zu,%.6e\n", 2 * N * 4, s);

    s = best_seconds([&]{ copy_float4<<<BLOCKS, TPB>>>((float4*)A, (float4*)B, N4); }, 10);
    printf("kernel,copy_float4,%zu,%.6e\n", 2 * N * 4, s);

    s = best_seconds([&]{ read_only<<<BLOCKS, TPB>>>((float4*)A, C, N4); }, 10);
    printf("kernel,read_only,%zu,%.6e\n", N * 4, s);

    s = best_seconds([&]{ write_only<<<BLOCKS, TPB>>>((float4*)B, N4, 1.5f); }, 10);
    printf("kernel,write_only,%zu,%.6e\n", N * 4, s);

    s = best_seconds([&]{ triad<<<BLOCKS, TPB>>>((float4*)A, (float4*)B, (float4*)C, N4, 2.f); }, 10);
    printf("kernel,triad,%zu,%.6e\n", 3 * N * 4, s);

    s = best_seconds([&]{ CK(cudaMemcpyAsync(B, A, N * 4, cudaMemcpyDeviceToDevice)); }, 10);
    printf("kernel,cudaMemcpy_D2D,%zu,%.6e\n", 2 * N * 4, s);

    // ---- block-size sweep for the scalar copy ----
    for (int tpb = 32; tpb <= 1024; tpb *= 2) {
        int blocks = p.multiProcessorCount * (2048 / tpb) * 2;
        s = best_seconds([&]{ copy_float4<<<blocks, tpb>>>((float4*)A, (float4*)B, N4); }, 10);
        printf("blocksize,%d,%zu,%.6e\n", tpb, 2 * N * 4, s);
    }

    // ---- size sweep: from inside L2 to far outside it ----
    for (size_t bytes = 4096; bytes <= (size_t)256 * 1024 * 1024; bytes *= 2) {
        size_t n = bytes / 4, n4 = n / 4;
        if (n4 == 0) continue;
        int blocks = (int)((n4 + TPB - 1) / TPB);
        if (blocks > BLOCKS) blocks = BLOCKS;
        // aim for at least ~1 ms of work per timed iteration
        int iters = (int)(200 * 1024 * 1024 / bytes);
        if (iters < 20) iters = 20;
        if (iters > 4000) iters = 4000;
        s = best_seconds([&]{ copy_float4<<<blocks, TPB>>>((float4*)A, (float4*)B, n4); }, iters);
        printf("sweep,copy_float4,%zu,%zu,%.6e\n", bytes, 2 * bytes, s);
    }

    // ---- an empty kernel: how much of a small measurement is pure overhead ----
    s = best_seconds([&]{ copy_float4<<<1, 32>>>((float4*)A, (float4*)B, 0); }, 1000);
    printf("kernel,empty_launch,0,%.6e\n", s);

    // ---- PCIe: host to device and back, pageable vs pinned ----
    const size_t H = 64u * 1024 * 1024;           // 64 MB
    void *pageable = malloc(H), *pinned = nullptr;
    CK(cudaMallocHost(&pinned, H));
    s = best_seconds([&]{ CK(cudaMemcpy(A, pageable, H, cudaMemcpyHostToDevice)); }, 5);
    printf("pcie,H2D_pageable,%zu,%.6e\n", H, s);
    s = best_seconds([&]{ CK(cudaMemcpy(A, pinned, H, cudaMemcpyHostToDevice)); }, 5);
    printf("pcie,H2D_pinned,%zu,%.6e\n", H, s);
    s = best_seconds([&]{ CK(cudaMemcpy(pageable, A, H, cudaMemcpyDeviceToHost)); }, 5);
    printf("pcie,D2H_pageable,%zu,%.6e\n", H, s);
    s = best_seconds([&]{ CK(cudaMemcpy(pinned, A, H, cudaMemcpyDeviceToHost)); }, 5);
    printf("pcie,D2H_pinned,%zu,%.6e\n", H, s);

    cudaFreeHost(pinned); free(pageable);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
