// linkprobe.cu - the acceptance test for a freshly built GPU box.
//
// Everything a bring-up checklist needs to measure about the *link* between
// host and device, plus the device-memory ceiling the link is compared with.
// Prints CSV lines on stdout; run.py parses them.
//
//   dev,<name>,<cc>,<sms>,<clockMHz>,<memClockMHz>,<busWidth>,<gmemMB>
//   h2d,<bytes>,<pageable_GBs>,<pinned_GBs>
//   d2h,<bytes>,<pageable_GBs>,<pinned_GBs>
//   duplex,<bytes>,<h2d_only_GBs>,<d2h_only_GBs>,<both_GBs>
//   zerocopy,<bytes>,<device_GBs>,<mapped_host_GBs>
//   dram,<bytes>,<GBs>
//   lat,<h2d_us>,<launch_us>
//
// Build: nvcc -O3 -arch=sm_61 linkprobe.cu -o linkprobe

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    return 1; } } while (0)

// Pure streaming read+write: 2 bytes moved per float touched.
__global__ void stream_copy(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i] * 1.0009f;
}

// Reads through whatever pointer it is handed (device or mapped host).
__global__ void stream_sum(const float* __restrict__ in, float* __restrict__ acc, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0.f;
    for (; i < n; i += stride) s += in[i];
    if (s == 1234.5678f) acc[0] = s;   // never true; keeps the loads alive
}

__global__ void empty_kernel() {}

static float time_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b); return ms;
}

int main() {
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    printf("dev,%s,%d.%d,%d,%d,%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.clockRate / 1000, p.memoryClockRate / 1000,
           p.memoryBusWidth, (size_t)(p.totalGlobalMem >> 20));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    const size_t sizes[] = {1u << 12, 1u << 16, 1u << 20, 1u << 22,
                            1u << 24, 1u << 26, 1u << 27, 1u << 28};
    const int nsz = sizeof(sizes) / sizeof(sizes[0]);
    const size_t maxb = sizes[nsz - 1];

    void *d_a, *d_b;
    CK(cudaMalloc(&d_a, maxb));
    CK(cudaMalloc(&d_b, maxb));
    void *h_pin;  CK(cudaHostAlloc(&h_pin, maxb, cudaHostAllocDefault));
    void *h_page = malloc(maxb);
    memset(h_page, 1, maxb);
    memset(h_pin, 1, maxb);

    // ---- A. transfer bandwidth, pageable vs pinned, both directions -------
    for (int s = 0; s < nsz; s++) {
        size_t b = sizes[s];
        int reps = b < (1u << 22) ? 200 : 20;
        float best_pg = 0, best_pn = 0, best_pg_d = 0, best_pn_d = 0;
        for (int trial = 0; trial < 3; trial++) {
            // H2D pageable
            CK(cudaMemcpy(d_a, h_page, b, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++) CK(cudaMemcpy(d_a, h_page, b, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > best_pg) best_pg = g;
            // H2D pinned
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++) CK(cudaMemcpy(d_a, h_pin, b, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > best_pn) best_pn = g;
            // D2H pageable
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++) CK(cudaMemcpy(h_page, d_a, b, cudaMemcpyDeviceToHost));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > best_pg_d) best_pg_d = g;
            // D2H pinned
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++) CK(cudaMemcpy(h_pin, d_a, b, cudaMemcpyDeviceToHost));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > best_pn_d) best_pn_d = g;
        }
        printf("h2d,%zu,%.3f,%.3f\n", b, best_pg, best_pn);
        printf("d2h,%zu,%.3f,%.3f\n", b, best_pg_d, best_pn_d);
    }

    // ---- B. full duplex: is the link symmetric and can both run at once? --
    {
        size_t b = 1u << 26;
        int reps = 20;
        cudaStream_t s1, s2;
        CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
        void *h_pin2; CK(cudaHostAlloc(&h_pin2, b, cudaHostAllocDefault));
        float up = 0, down = 0, both = 0;
        for (int trial = 0; trial < 3; trial++) {
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++)
                CK(cudaMemcpyAsync(d_a, h_pin, b, cudaMemcpyHostToDevice, s1));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > up) up = g;

            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++)
                CK(cudaMemcpyAsync(h_pin2, d_b, b, cudaMemcpyDeviceToHost, s2));
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > down) down = g;

            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++) {
                CK(cudaMemcpyAsync(d_a, h_pin, b, cudaMemcpyHostToDevice, s1));
                CK(cudaMemcpyAsync(h_pin2, d_b, b, cudaMemcpyDeviceToHost, s2));
            }
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = 2.0 * b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > both) both = g;
        }
        printf("duplex,%zu,%.3f,%.3f,%.3f\n", b, up, down, both);
        CK(cudaFreeHost(h_pin2));
        CK(cudaStreamDestroy(s1)); CK(cudaStreamDestroy(s2));
    }

    // ---- C. zero-copy: kernel reading host memory straight over the link --
    {
        size_t b = 1u << 26;               // 64 MB
        size_t n = b / sizeof(float);
        float *h_map, *d_map;
        CK(cudaHostAlloc((void**)&h_map, b, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**)&d_map, h_map, 0));
        for (size_t i = 0; i < n; i++) h_map[i] = 1.0f;
        float dev_gbs = 0, map_gbs = 0;
        int reps = 5;
        for (int trial = 0; trial < 2; trial++) {
            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++)
                stream_sum<<<256, 256>>>((const float*)d_a, (float*)d_b, n);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > dev_gbs) dev_gbs = g;

            CK(cudaEventRecord(t0));
            for (int r = 0; r < reps; r++)
                stream_sum<<<256, 256>>>((const float*)d_map, (float*)d_b, n);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            g = b * (double)reps / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > map_gbs) map_gbs = g;
        }
        printf("zerocopy,%zu,%.3f,%.3f\n", b, dev_gbs, map_gbs);
        CK(cudaFreeHost(h_map));
    }

    // ---- D. device DRAM ceiling (what the link is being compared to) ------
    {
        size_t b = 1u << 27;
        size_t n = b / sizeof(float);
        float best = 0;
        for (int trial = 0; trial < 3; trial++) {
            CK(cudaEventRecord(t0));
            for (int r = 0; r < 10; r++)
                stream_copy<<<512, 256>>>((const float*)d_a, (float*)d_b, n);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float g = 2.0 * b * 10.0 / (time_ms(t0, t1) * 1e-3) / 1e9;
            if (g > best) best = g;
        }
        printf("dram,%zu,%.3f\n", b, best);
    }

    // ---- E. small-transfer latency and launch overhead --------------------
    {
        int reps = 2000;
        CK(cudaMemcpy(d_a, h_pin, 4, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(t0));
        for (int r = 0; r < reps; r++) CK(cudaMemcpy(d_a, h_pin, 4, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float h2d_us = time_ms(t0, t1) * 1000.f / reps;

        empty_kernel<<<1, 1>>>(); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int r = 0; r < reps; r++) empty_kernel<<<1, 1>>>();
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float launch_us = time_ms(t0, t1) * 1000.f / reps;
        printf("lat,%.3f,%.3f\n", h2d_us, launch_us);
    }

    free(h_page);
    CK(cudaFreeHost(h_pin));
    CK(cudaFree(d_a)); CK(cudaFree(d_b));
    return 0;
}
