// Project 12 - shared-memory bank conflicts.
//
// Shared memory is not one memory. It is 32 independent little memories
// ("banks"), each 4 bytes wide, interleaved:
//
//   bank of element i  =  i % 32          (for 4-byte elements)
//
//   element:  0  1  2  ... 31 | 32 33 ... 63 | 64 ...
//   bank:     0  1  2  ... 31 |  0  1 ... 31 |  0 ...
//
// A warp's 32 lanes can be served in ONE cycle if they hit 32 different banks.
// If k lanes want different addresses in the SAME bank, that bank has to serve
// them one after another: a k-way conflict, k times slower.
//
// The exception that trips everyone up: if lanes want the SAME address, there
// is no conflict at all - the hardware broadcasts one value to all of them.
// Conflict needs different rows of the same bank, not merely the same bank.
//
// Four experiments:
//   A. stride sweep      - conflict degree is gcd(stride, 32); does it show up?
//   B. broadcast         - stride 0: 32 lanes, one address. Conflict or not?
//   C. matrix transpose  - the classic, with three fixes measured end to end
//   D. cost in context   - the same conflict inside a DRAM-bound kernel

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

#define SMEM_ELEMS 2048            // 8 KB of shared memory per block
#define SMASK (SMEM_ELEMS - 1)

// ---------------------------------------------------------------------------
// A + B. Pure shared-memory throughput at a controlled bank pattern.
//
// Lane L reads element (L*stride + 32*iteration). Adding 32 per iteration walks
// to a new row of the SAME bank, so the conflict pattern is identical on every
// iteration while the address keeps changing (a constant address would just be
// hoisted out of the loop by the compiler).
//
// stride 0 is the broadcast case: every lane reads the same address.
// ---------------------------------------------------------------------------
__global__ void k_bank(float *out, int stride, int iters) {
    __shared__ float s[SMEM_ELEMS];
    for (int i = threadIdx.x; i < SMEM_ELEMS; i += blockDim.x) s[i] = i * 1e-3f;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int b0 = (lane * stride) & SMASK;
    // four independent streams so the loads overlap and we measure throughput,
    // not one load's latency
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int it = 0; it < iters; ++it) {
        int o = (it * 128) & SMASK;
        a0 += s[(b0 + o) & SMASK];
        a1 += s[(b0 + o + 32) & SMASK];
        a2 += s[(b0 + o + 64) & SMASK];
        a3 += s[(b0 + o + 96) & SMASK];
    }
    float acc = a0 + a1 + a2 + a3;
    if (acc == 1.2345e30f) out[0] = acc;
}

// ---------------------------------------------------------------------------
// C. 32x32 tiled matrix transpose. Four versions.
// ---------------------------------------------------------------------------
#define TW 32                       // tile width
#define TR 8                        // rows handled per thread block pass

// C0: no shared memory at all. Reads are coalesced, writes are stride-N.
__global__ void k_tr_naive(const float *__restrict__ in, float *out, int n) {
    int x = blockIdx.x * TW + threadIdx.x;
    int y = blockIdx.y * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR)
        if (x < n && y + j < n) out[x * (long)n + (y + j)] = in[(y + j) * (long)n + x];
}

// C1: shared-memory tile, no padding. The column read tile[tx][ty+j] puts all
// 32 lanes in bank (tx*32 + c) % 32 = c - the SAME bank. 32-way conflict.
__global__ void k_tr_smem(const float *__restrict__ in, float *out, int n) {
    __shared__ float tile[TW][TW];
    int x = blockIdx.x * TW + threadIdx.x;
    int y = blockIdx.y * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * (long)n + x];
    __syncthreads();
    x = blockIdx.y * TW + threadIdx.x;
    y = blockIdx.x * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) out[(y + j) * (long)n + x] = tile[threadIdx.x][threadIdx.y + j];
}

// C2: one padding column. Row length 33 is odd, so bank = (tx*33 + c) % 32 =
// (tx + c) % 32, which is different for every lane. Costs 128 extra bytes.
__global__ void k_tr_pad(const float *__restrict__ in, float *out, int n) {
    __shared__ float tile[TW][TW + 1];
    int x = blockIdx.x * TW + threadIdx.x;
    int y = blockIdx.y * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * (long)n + x];
    __syncthreads();
    x = blockIdx.y * TW + threadIdx.x;
    y = blockIdx.x * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) out[(y + j) * (long)n + x] = tile[threadIdx.x][threadIdx.y + j];
}

// C3: XOR swizzle. Element (r,c) is stored at column c^r. Both the row write
// and the column read then land on 32 different banks - and unlike padding it
// uses zero extra shared memory.
__global__ void k_tr_swizzle(const float *__restrict__ in, float *out, int n) {
    __shared__ float tile[TW][TW];
    int x = blockIdx.x * TW + threadIdx.x;
    int y = blockIdx.y * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) {
        int r = threadIdx.y + j;
        tile[r][threadIdx.x ^ r] = in[(y + j) * (long)n + x];
    }
    __syncthreads();
    x = blockIdx.y * TW + threadIdx.x;
    y = blockIdx.x * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR) {
        int c = threadIdx.y + j;
        out[(y + j) * (long)n + x] = tile[threadIdx.x][c ^ threadIdx.x];
    }
}

// C4: the speed limit. A plain copy - same bytes in, same bytes out, no
// transposition at all. Nothing that moves this much data can beat it.
__global__ void k_copy(const float *__restrict__ in, float *out, int n) {
    int x = blockIdx.x * TW + threadIdx.x;
    int y = blockIdx.y * TW + threadIdx.y;
    for (int j = 0; j < TW; j += TR)
        out[(y + j) * (long)n + x] = in[(y + j) * (long)n + x];
}

// ---------------------------------------------------------------------------
// D. The same 32-way conflict, but wrapped in a kernel that also has to stream
// data from DRAM. Does the conflict still cost what experiment A says?
//
// The knob that matters is the RATIO: DRAM bytes moved per conflicted shared
// read = 4*nload/work. Sweeping both `work` and `nload` sweeps that ratio, and
// the transpose in experiment C sits at 8 bytes per conflicted read (4 in,
// 4 out) - so this experiment can predict the transpose result.
// ---------------------------------------------------------------------------
__global__ void k_ctx(const float *__restrict__ in, float *out, long n,
                      int stride, int work, int nload) {
    __shared__ float s[SMEM_ELEMS];
    for (int i = threadIdx.x; i < SMEM_ELEMS; i += blockDim.x) s[i] = i * 1e-3f;
    __syncthreads();
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = gridDim.x * (long)blockDim.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < nload; ++k) acc += in[(i + k * total) % n];  // coalesced
    int lane = threadIdx.x & 31, b0 = (lane * stride) & SMASK;
    for (int k = 0; k < work; ++k) acc += s[(b0 + 32 * k) & SMASK];
    if (acc == 1.2345e30f) out[0] = acc;
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

static int gcd32(int s) {
    if (s == 0) return 0;                 // broadcast: not a conflict at all
    int a = s % 32, b = 32;
    if (a == 0) return 32;
    while (a) { int t = b % a; b = a; a = t; }
    return b;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("#device,%s,%d.%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, (size_t)p.sharedMemPerBlock);

    float *d_out, *d_a, *d_b;
    const int N = 4096;                                    // 4096^2 floats = 64 MB
    CHK(cudaMalloc(&d_out, 1024 * sizeof(float)));
    CHK(cudaMalloc(&d_a, (size_t)N * N * sizeof(float)));
    CHK(cudaMalloc(&d_b, (size_t)N * N * sizeof(float)));
    CHK(cudaMemset(d_out, 0, 1024 * sizeof(float)));
    CHK(cudaMemset(d_a, 0, (size_t)N * N * sizeof(float)));

    for (int i = 0; i < 250; ++i) k_spin<<<19 * 8, 256>>>(d_out, 200000);
    CHK(cudaDeviceSynchronize());

    // ---------------- A + B. stride sweep, including stride 0 ----------------
    {
        const int TPB = 256, BLOCKS = 19 * 8, ITERS = 4096;
        const int strides[] = {0, 1, 2, 3, 4, 5, 8, 16, 17, 32, 33};

        for (int s : strides) {
            float ms = time_ms([&] { k_bank<<<BLOCKS, TPB>>>(d_out, s, ITERS); });
            double loads = (double)BLOCKS * TPB * ITERS * 4;
            double gls = loads / (ms * 1e-3) / 1e9;

            printf("bank,%d,%d,%.6f,%.3f\n", s, gcd32(s), ms, gls);
        }

    }

    // ---------------- C. transpose ----------------
    {
        dim3 blk(TW, TR), grd(N / TW, N / TW);
        double bytes = 2.0 * (double)N * N * sizeof(float);
        struct { const char *name; void (*k)(const float *, float *, int); } v[] = {
            {"copy_limit",  k_copy},
            {"naive",       k_tr_naive},
            {"smem_nopad",  k_tr_smem},
            {"smem_pad1",   k_tr_pad},
            {"smem_swizzle", k_tr_swizzle},
        };
        for (auto &e : v) {
            auto kk = e.k;
            float ms = time_ms([&] { kk<<<grd, blk>>>(d_a, d_b, N); });
            printf("transpose,%s,%.6f,%.2f\n", e.name, ms, bytes / (ms * 1e-3) / 1e9);
        }
    }

    // ---------------- D. the same conflict, in context ----------------
    {
        const long n = 1L << 24;                       // 64 MB streamed
        const int TPB = 256;
        // (nload, work) pairs, ordered by DRAM bytes per conflicted read
        const int cfg[][2] = {{8, 1}, {4, 1}, {2, 1}, {1, 1}, {1, 2},
                              {1, 4}, {1, 8}, {1, 16}, {1, 32}};
        for (auto &c : cfg) {
            int nload = c[0], work = c[1];
            long grid = (n / nload + TPB - 1) / TPB;
            float t1 = time_ms([&] { k_ctx<<<grid, TPB>>>(d_a, d_out, n, 1, work, nload); });
            float t32 = time_ms([&] { k_ctx<<<grid, TPB>>>(d_a, d_out, n, 32, work, nload); });
            printf("context,%d,%d,%.3f,%.6f,%.6f,%.4f\n", nload, work,
                   4.0 * nload / work, t1, t32, t32 / t1);
        }
    }

    cudaFree(d_out); cudaFree(d_a); cudaFree(d_b);
    return 0;
}
