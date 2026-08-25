// Project 06 - everything the CUDA runtime knows about the GPU that
// `nvidia-smi` will not tell you, plus two load kernels used to show that
// `utilization.gpu` is not a measure of work.
//
// Modes:
//   ./devprops           print `key,value` device facts and exit
//   ./devprops lazy      run ONE warp doing a dependent FMA chain (~0 FLOP/s)
//   ./devprops busy      run every SM flat out (near peak FLOP/s)
//
// Both load modes print their achieved GFLOP/s so run.py can put a real number
// next to whatever nvidia-smi claimed the "utilization" was.

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define UNROLL 128

// ONE warp, ONE dependent chain. Each FMA must wait for the previous one, so
// this occupies 1 of 19 SMs and leaves the arithmetic pipeline ~empty.
__global__ void one_lazy_warp(long long iters, float *sink) {
    float a = 1e-6f * threadIdx.x, b = 1.0000001f, c = 1e-9f;
    for (long long i = 0; i < iters; ++i)
        #pragma unroll
        for (int j = 0; j < UNROLL; ++j) a = fmaf(a, b, c);
    if (a == 12345.678f) *sink = a;      // never true; stops the compiler
}

// Every SM, four independent chains per thread so the FMA pipeline stays full.
__global__ void everyone_works(long long iters, float *sink) {
    float a = 1e-6f * threadIdx.x, e = 2e-6f * threadIdx.x;
    float f = 3e-6f * threadIdx.x, g = 4e-6f * threadIdx.x;
    float b = 1.0000001f, c = 1e-9f;
    for (long long i = 0; i < iters; ++i)
        #pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            a = fmaf(a, b, c); e = fmaf(e, b, c);
            f = fmaf(f, b, c); g = fmaf(g, b, c);
        }
    if (a + e + f + g == 12345.678f) *sink = a;
}

static void kv(const char *k, const char *v) { printf("%s,%s\n", k, v); }
static void kvi(const char *k, long long v) { printf("%s,%lld\n", k, v); }

// Cores per SM is NOT reported by any API and NOT reported by nvidia-smi.
// It is a property of the architecture that you must look up from the compute
// capability. This is the same table that lives inside NVIDIA's helper_cuda.h.
static int cores_per_sm(int major, int minor) {
    switch (major * 10 + minor) {
        case 30: case 32: case 35: case 37: return 192;  // Kepler
        case 50: case 52: case 53: return 128;           // Maxwell
        case 60: return 64;                              // Pascal GP100
        case 61: case 62: return 128;                    // Pascal GP10x
        case 70: case 72: case 75: return 64;            // Volta, Turing
        case 80: return 64;                              // Ampere GA100
        case 86: case 87: case 89: return 128;           // Ampere GA10x, Ada
        case 90: return 128;                             // Hopper
        case 100: case 101: case 120: return 128;        // Blackwell
        default: return 0;
    }
}

int main(int argc, char **argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    if (argc > 1) {
        // ---------- load mode ----------
        bool busy = strcmp(argv[1], "busy") == 0;
        float *sink = nullptr;
        cudaMalloc(&sink, sizeof(float));

        // sized so each load runs ~4 s, long enough for nvidia-smi to sample it
        long long iters = busy ? 500000 : 8000000;
        int blocks = busy ? p.multiProcessorCount * 16 : 1;
        int tpb = busy ? 256 : 32;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        // warm up so the first-launch context cost is not in the timing
        if (busy) everyone_works<<<blocks, tpb>>>(10, sink);
        else      one_lazy_warp<<<blocks, tpb>>>(10, sink);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        if (busy) everyone_works<<<blocks, tpb>>>(iters, sink);
        else      one_lazy_warp<<<blocks, tpb>>>(iters, sink);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

        // 2 FLOP per FMA; `busy` issues 4 FMAs per unrolled step.
        double fmas = (double)iters * UNROLL * (busy ? 4.0 : 1.0)
                    * (double)blocks * (double)tpb;
        printf("load,%s,%.4f,%.4f\n", argv[1], ms / 1e3, 2.0 * fmas / (ms / 1e3) / 1e9);
        cudaFree(sink);
        return 0;
    }

    // ---------- properties mode ----------
    kv("name", p.name);
    char cc[16]; snprintf(cc, sizeof cc, "%d.%d", p.major, p.minor);
    kv("compute_capability", cc);
    kvi("sm_count", p.multiProcessorCount);
    kvi("cores_per_sm", cores_per_sm(p.major, p.minor));
    kvi("total_cores", (long long)p.multiProcessorCount * cores_per_sm(p.major, p.minor));
    kvi("clock_khz", p.clockRate);
    kvi("mem_clock_khz", p.memoryClockRate);
    kvi("mem_bus_bits", p.memoryBusWidth);
    kvi("l2_bytes", p.l2CacheSize);
    kvi("total_mem_bytes", (long long)p.totalGlobalMem);
    kvi("shared_mem_per_block", (long long)p.sharedMemPerBlock);
    kvi("shared_mem_per_sm", (long long)p.sharedMemPerMultiprocessor);
    kvi("regs_per_block", p.regsPerBlock);
    kvi("regs_per_sm", p.regsPerMultiprocessor);
    kvi("max_threads_per_sm", p.maxThreadsPerMultiProcessor);
    kvi("max_warps_per_sm", p.maxThreadsPerMultiProcessor / p.warpSize);
    kvi("max_threads_per_block", p.maxThreadsPerBlock);
    kvi("warp_size", p.warpSize);
    kvi("async_engine_count", p.asyncEngineCount);
    kvi("concurrent_kernels", p.concurrentKernels);
    kvi("ecc_enabled", p.ECCEnabled);
    kvi("unified_addressing", p.unifiedAddressing);
    kvi("managed_memory", p.managedMemory);
    kvi("cooperative_launch", p.cooperativeLaunch);
    kvi("global_l1_cache_supported", p.globalL1CacheSupported);
    kvi("pci_domain_id", p.pciDomainID);
    kvi("pci_bus_id", p.pciBusID);
    kvi("pci_device_id", p.pciDeviceID);

    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    kvi("runtime_free_bytes", (long long)freeb);
    kvi("runtime_total_bytes", (long long)totalb);

    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    kvi("cuda_runtime_version", rt);
    kvi("cuda_driver_version", drv);
    return 0;
}
