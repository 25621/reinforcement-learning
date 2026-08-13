// Project 14 - saturating off-chip memory bandwidth with a vector add.
//
// A vector add (c = a + b) has one addition per 12 bytes touched: arithmetic
// intensity 1/12 = 0.083 FLOP/byte, against a ridge point of 32. It is 400x
// below the point where compute could ever matter, so its runtime is a direct
// readout of memory bandwidth and nothing else. That makes it the standard
// instrument for asking "how much of the memory system can I actually get?"
//
// Experiments:
//   A. tuning       - work per thread, vector width, block size, grid size
//   B. what a byte costs - reads vs writes vs partial writes
//   C. size sweep   - the same kernel from 4 KiB to 256 MiB (finds the caches)
//
// Note on the name: this GPU has GDDR5, not HBM. The number is smaller
// (256 GB/s vs 3350 GB/s on an H100) but every technique and every failure
// mode below is identical - it is the same DRAM physics behind a narrower bus.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

static const long N = 1L << 26;        // 64 Mi floats per array = 256 MB each

// ---------------------------------------------------------------------------
// A. Vector add, five ways of dividing the work up.
// ---------------------------------------------------------------------------

// One element per thread. The textbook version.
__global__ void k_add1(const float *__restrict__ a, const float *__restrict__ b,
                       float *__restrict__ c, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Grid-stride loop: a fixed grid, each thread walks the array in strides of
// the whole grid. Consecutive threads stay adjacent, so it is still coalesced.
__global__ void k_add_gs(const float *__restrict__ a, const float *__restrict__ b,
                         float *__restrict__ c, long n) {
    long stride = gridDim.x * (long)blockDim.x;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// float2 / float4: one instruction moves 8 or 16 bytes instead of 4. Fewer
// instructions in flight for the same bytes, so less chance of running out of
// issue slots before running out of bandwidth.
__global__ void k_add2(const float2 *__restrict__ a, const float2 *__restrict__ b,
                       float2 *__restrict__ c, long n2) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n2) { float2 x = a[i], y = b[i]; c[i] = make_float2(x.x + y.x, x.y + y.y); }
}

__global__ void k_add4(const float4 *__restrict__ a, const float4 *__restrict__ b,
                       float4 *__restrict__ c, long n4) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// float4 in a grid-stride loop - the two ideas together.
__global__ void k_add4_gs(const float4 *__restrict__ a, const float4 *__restrict__ b,
                          float4 *__restrict__ c, long n4) {
    long stride = gridDim.x * (long)blockDim.x;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

// ---------------------------------------------------------------------------
// B. What does a byte cost? Reads, writes, and half-written sectors.
// ---------------------------------------------------------------------------
__global__ void k_read2(const float *__restrict__ a, const float *__restrict__ b,
                        float *__restrict__ c, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i] + b[i];
    if (v == 1.2345e30f) c[0] = v;                 // sink, never taken
}

__global__ void k_copy(const float *__restrict__ a, float *__restrict__ c, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}

__global__ void k_write(float *__restrict__ c, long n, float v) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) c[i] = v;
}

// Writes only every other float, so every 32-byte sector ends up HALF written.
// Two things the hardware could do: fetch the sector, merge, write it back
// (read-for-ownership, 2x traffic), or write the sector with byte enables so
// the untouched bytes are left alone (1x traffic). The measured time picks a
// winner - see the README; one of the two is arithmetically impossible here.
__global__ void k_write_half(float *__restrict__ c, long n, float v) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (2 * i < n) c[2 * i] = v;
}

// A control for the one above: writes the same NUMBER of floats, contiguously.
// Without it, "half-stride stores are slow" could just mean "half the threads".
__global__ void k_write_half_dense(float *__restrict__ c, long n, float v) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n / 2) c[i] = v;
}

// Sum R input arrays into one output, R = 1..4. Everything is perfectly
// coalesced in every case; the only thing that changes is the ratio of reads
// to writes, which is what decides how often the DRAM bus has to turn around.
template <int R>
__global__ void k_ratio(const float *__restrict__ a, const float *__restrict__ b,
                        const float *__restrict__ d, const float *__restrict__ e,
                        float *__restrict__ c, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    if (R > 1) v += b[i];
    if (R > 2) v += d[i];
    if (R > 3) v += e[i];
    c[i] = v;
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
static float time_ms(F launch, int iters = 10, int reps = 5) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(); CHK(cudaDeviceSynchronize());
    float t[16];
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

static long grid_for(long n, int tpb) { return (n + tpb - 1) / tpb; }

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    // spec peak = memory clock x 2 (DDR) x bus width / 8
    double peak = p.memoryClockRate * 1e3 * 2.0 * (p.memoryBusWidth / 8.0) / 1e9;
    printf("#device,%s,%d.%d,%d,%.2f,%zu,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, peak, (size_t)p.l2CacheSize, p.memoryBusWidth);

    float *a, *b, *c, *d, *e;
    for (float **q : {&a, &b, &c, &d, &e}) {
        CHK(cudaMalloc(q, N * sizeof(float)));
        CHK(cudaMemset(*q, 0, N * sizeof(float)));
    }

    for (int i = 0; i < 250; ++i) k_spin<<<19 * 8, 256>>>(c, 200000);
    CHK(cudaDeviceSynchronize());

    const double add_bytes = 3.0 * N * sizeof(float);      // 2 read + 1 write

    // ---------------- A1. variant sweep at a fixed block size ----------------
    {
        int tpb = 256;
        float ms;
        ms = time_ms([&] { k_add1<<<grid_for(N, tpb), tpb>>>(a, b, c, N); });
        printf("variant,scalar_1elem,%d,%ld,%.6f,%.2f\n", tpb, grid_for(N, tpb),
               ms, add_bytes / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_add2<<<grid_for(N / 2, tpb), tpb>>>((float2 *)a, (float2 *)b, (float2 *)c, N / 2); });
        printf("variant,float2,%d,%ld,%.6f,%.2f\n", tpb, grid_for(N / 2, tpb),
               ms, add_bytes / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_add4<<<grid_for(N / 4, tpb), tpb>>>((float4 *)a, (float4 *)b, (float4 *)c, N / 4); });
        printf("variant,float4,%d,%ld,%.6f,%.2f\n", tpb, grid_for(N / 4, tpb),
               ms, add_bytes / (ms * 1e-3) / 1e9);
        long g = 19 * 8;
        ms = time_ms([&] { k_add_gs<<<g, tpb>>>(a, b, c, N); });
        printf("variant,gridstride_scalar,%d,%ld,%.6f,%.2f\n", tpb, g, ms,
               add_bytes / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_add4_gs<<<g, tpb>>>((float4 *)a, (float4 *)b, (float4 *)c, N / 4); });
        printf("variant,gridstride_float4,%d,%ld,%.6f,%.2f\n", tpb, g, ms,
               add_bytes / (ms * 1e-3) / 1e9);
    }

    // ---------------- A2. block size, on the scalar kernel ----------------
    for (int tpb : {32, 64, 128, 256, 512, 1024}) {
        float ms = time_ms([&] { k_add1<<<grid_for(N, tpb), tpb>>>(a, b, c, N); });
        printf("block,%d,%ld,%.6f,%.2f\n", tpb, grid_for(N, tpb), ms,
               add_bytes / (ms * 1e-3) / 1e9);
    }

    // ---------------- A3. grid size, on the grid-stride float4 kernel ----------------
    for (int bps : {1, 2, 4, 8, 16, 32, 64, 128}) {
        long g = (long)bps * p.multiProcessorCount;
        float ms = time_ms([&] { k_add4_gs<<<g, 256>>>((float4 *)a, (float4 *)b, (float4 *)c, N / 4); });
        printf("grid,%d,%ld,%.6f,%.2f\n", bps, g, ms,
               add_bytes / (ms * 1e-3) / 1e9);
    }

    // ---------------- B. what a byte costs ----------------
    {
        int tpb = 256;
        float ms;
        ms = time_ms([&] { k_read2<<<grid_for(N, tpb), tpb>>>(a, b, c, N); });
        printf("cost,read2,%.6f,%.2f,%.2f\n", ms,
               2.0 * N * 4 / (ms * 1e-3) / 1e9, 2.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_copy<<<grid_for(N, tpb), tpb>>>(a, c, N); });
        printf("cost,copy_1r1w,%.6f,%.2f,%.2f\n", ms,
               2.0 * N * 4 / (ms * 1e-3) / 1e9, 2.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_write<<<grid_for(N, tpb), tpb>>>(c, N, 1.f); });
        printf("cost,write_only,%.6f,%.2f,%.2f\n", ms,
               1.0 * N * 4 / (ms * 1e-3) / 1e9, 1.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_add1<<<grid_for(N, tpb), tpb>>>(a, b, c, N); });
        printf("cost,add_2r1w,%.6f,%.2f,%.2f\n", ms,
               3.0 * N * 4 / (ms * 1e-3) / 1e9, 3.0 * N * 4 / (ms * 1e-3) / 1e9);
        // half the data written, two ways
        ms = time_ms([&] { k_write_half_dense<<<grid_for(N / 2, tpb), tpb>>>(c, N, 1.f); });
        printf("cost,write_half_dense,%.6f,%.2f,%.2f\n", ms,
               0.5 * N * 4 / (ms * 1e-3) / 1e9, 0.5 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_write_half<<<grid_for(N / 2, tpb), tpb>>>(c, N, 1.f); });
        printf("cost,write_half_strided,%.6f,%.2f,%.2f\n", ms,
               0.5 * N * 4 / (ms * 1e-3) / 1e9, 1.0 * N * 4 / (ms * 1e-3) / 1e9);
    }

    // ---------------- B2. read:write ratio ----------------
    {
        int tpb = 256;
        long g = grid_for(N, tpb);
        float ms;
        ms = time_ms([&] { k_ratio<1><<<g, tpb>>>(a, b, d, e, c, N); });
        printf("ratio,1,%.6f,%.2f\n", ms, 2.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_ratio<2><<<g, tpb>>>(a, b, d, e, c, N); });
        printf("ratio,2,%.6f,%.2f\n", ms, 3.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_ratio<3><<<g, tpb>>>(a, b, d, e, c, N); });
        printf("ratio,3,%.6f,%.2f\n", ms, 4.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_ratio<4><<<g, tpb>>>(a, b, d, e, c, N); });
        printf("ratio,4,%.6f,%.2f\n", ms, 5.0 * N * 4 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_read2<<<g, tpb>>>(a, b, c, N); });
        printf("ratio,99,%.6f,%.2f\n", ms, 2.0 * N * 4 / (ms * 1e-3) / 1e9);
    }

    // ---------------- C. size sweep, finding the caches ----------------
    for (int shift = 10; shift <= 26; ++shift) {
        long n = 1L << shift;                     // floats per array
        int tpb = 256;
        // enough repeats that even a tiny buffer takes a measurable time
        int iters = (int)(1 + (1L << 24) / n);
        if (iters > 2000) iters = 2000;
        float ms = time_ms([&] { k_copy<<<grid_for(n, tpb), tpb>>>(a, c, n); },
                           iters, 5);
        printf("size,%ld,%.6f,%.2f\n", n * 4, ms,
               2.0 * n * 4 / (ms * 1e-3) / 1e9);
    }

    cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d); cudaFree(e);
    return 0;
}
