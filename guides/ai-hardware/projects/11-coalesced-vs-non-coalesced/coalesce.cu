// Project 11 - coalesced vs non-coalesced global-memory access.
//
// "Coalescing" = the memory system merging the 32 addresses a warp asks for
// into as few memory transactions as it can. The unit it merges into on this
// architecture is a 32-byte SECTOR, not the 128-byte cache line people usually
// quote. That one fact predicts every number this program prints:
//
//   sectors touched by one warp = min(32, 4 * stride_in_floats)
//
// so the penalty grows until stride 8 (32 sectors = one per lane) and then
// STOPS, because a warp cannot ask for more than 32 separate places at once.
//
// Four experiments:
//   A. stride sweep      - where the collapse starts and where it stops
//   B. alignment sweep   - shifting the base pointer by 1..31 floats
//   C. lane permutation  - does it matter WHICH lane reads WHICH address?
//   D. AoS vs SoA        - the same question wearing a data-structure hat
//
// Everything is read-only: each kernel loads and folds into an accumulator
// that is written back only on a value that never occurs, so nothing is
// dead-code-eliminated and no store traffic pollutes the measurement.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

static const long NBUF = 1L << 26;   // 64 Mi floats = 256 MB, 128x the 2 MB L2
static const int  TPB  = 256;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// A: thread i reads element i*stride. One 4-byte load per thread, always.
// Only the ADDRESS PATTERN changes, never the amount of useful data per thread.
__global__ void k_stride(const float *__restrict__ in, float *out,
                         long nread, long stride) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nread) return;
    float v = in[i * stride];
    if (v == 1.2345e30f) out[0] = v;      // sink: never taken, never removed
}

// A': same count of loads, addresses drawn from a precomputed random
// permutation of the whole buffer. Same number of sectors as a large stride,
// but no DRAM page locality at all.
__global__ void k_random(const float *__restrict__ in, const int *__restrict__ idx,
                         float *out, long nread) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nread) return;
    float v = in[idx[i]];
    if (v == 1.2345e30f) out[0] = v;
}

// B: contiguous, but the whole pattern shifted by `off` floats. Tests what a
// misaligned base pointer costs.
__global__ void k_offset(const float *__restrict__ in, float *out,
                         long nread, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nread) return;
    float v = in[i + off];
    if (v == 1.2345e30f) out[0] = v;
}

// C: contiguous 128-byte window per warp, but the lanes inside the warp are
// shuffled. mode 0 = identity, 1 = reversed, 2 = XOR with 0b10101, 3 = a fixed
// pseudo-random permutation read from a table, 4 = the CONTROL for mode 3: the
// same table lookup, but the table holds the identity. Mode 4 exists so that
// mode 3's cost can be split into "reordering the lanes" and "paying for one
// extra load to find out where to go". Without it the two are inseparable.
__global__ void k_permute(const float *__restrict__ in, float *out,
                          long nread, int mode, const int *__restrict__ perm) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nread) return;
    int lane = threadIdx.x & 31;
    int newlane;
    switch (mode) {
        case 1:  newlane = 31 - lane; break;
        case 2:  newlane = lane ^ 21; break;
        case 3:  newlane = perm[lane]; break;
        case 4:  newlane = perm[32 + lane]; break;
        default: newlane = lane;
    }
    long base = i - lane;
    float v = in[base + newlane];
    if (v == 1.2345e30f) out[0] = v;
}

struct Particle { float x, y, z, w; };   // 16 bytes - the classic AoS layout

// D1: array of structs, read ONE field. Consecutive threads are 16 bytes
// apart, i.e. exactly stride 4.
__global__ void k_aos1(const Particle *__restrict__ p, float *out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i].x;
    if (v == 1.2345e30f) out[0] = v;
}

// D2: array of structs, read ALL FOUR fields. Now the warp consumes every byte
// it pulls in.
__global__ void k_aos4(const Particle *__restrict__ p, float *out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle q = p[i];
    float v = q.x + q.y + q.z + q.w;
    if (v == 1.2345e30f) out[0] = v;
}

// D3: struct of arrays, read one field -> perfectly contiguous.
__global__ void k_soa1(const float *__restrict__ x, float *out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v == 1.2345e30f) out[0] = v;
}

// D4: struct of arrays, read all four fields from four separate arrays.
__global__ void k_soa4(const float *__restrict__ x, const float *__restrict__ y,
                       const float *__restrict__ z, const float *__restrict__ w,
                       float *out, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + y[i] + z[i] + w[i];
    if (v == 1.2345e30f) out[0] = v;
}

// A hot spin used only to pull the GPU off its 164 MHz idle clock. Without it
// the first measurement of the run is ~35% slow for no reason at all.
__global__ void k_spin(float *out, int iters) {
    float a = threadIdx.x * 1e-4f, b = 1.0000001f, c = 1e-7f;
    for (int i = 0; i < iters; ++i) a = fmaf(a, b, c);
    if (a == 1.2345e30f) out[0] = a;
}

// ---------------------------------------------------------------------------
// Timing: median of `reps` timed rounds, each round `iters` launches.
// ---------------------------------------------------------------------------
static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

template <class F>
static float time_ms(F launch, int iters = 20, int reps = 5) {
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

static long grid_for(long n) { return (n + TPB - 1) / TPB; }

// Sectors a warp touches for a given element stride, from the hardware rule.
static int sectors_per_warp(long stride) {
    long s = 4 * stride;               // 32 lanes * 4 B spread over 32 B sectors
    return (int)(s > 32 ? 32 : s);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("#device,%s,%d.%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, (size_t)p.l2CacheSize);

    float *d_in, *d_out; int *d_idx, *d_perm;
    CHK(cudaMalloc(&d_in, NBUF * sizeof(float)));
    CHK(cudaMalloc(&d_out, 1024 * sizeof(float)));
    CHK(cudaMemset(d_in, 0, NBUF * sizeof(float)));
    CHK(cudaMemset(d_out, 0, 1024 * sizeof(float)));

    for (int i = 0; i < 250; ++i) k_spin<<<19 * 8, 256>>>(d_out, 200000);   // ~1 s of heat, clocks go up
    CHK(cudaDeviceSynchronize());

    // ---------------- A. stride sweep ----------------
    const long strides[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (long s : strides) {
        long nread = NBUF / s;
        float ms = time_ms([&] { k_stride<<<grid_for(nread), TPB>>>(d_in, d_out, nread, s); });
        double useful = nread * 4.0;                            // bytes we asked for
        int sec = sectors_per_warp(s);
        double moved = nread / 32.0 * sec * 32.0;               // bytes DRAM/L2 moved
        printf("stride,%ld,%ld,%d,%.6f,%.2f,%.2f\n", s, nread, sec, ms,
               useful / (ms * 1e-3) / 1e9, moved / (ms * 1e-3) / 1e9);
    }
    // A': fully random gather, same load count as stride 256
    {
        long nread = NBUF / 256;
        int *h = (int *)malloc(nread * sizeof(int));
        unsigned long long r = 88172645463325252ULL;
        for (long i = 0; i < nread; ++i) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            h[i] = (int)(r % (unsigned long long)NBUF);
        }
        CHK(cudaMalloc(&d_idx, nread * sizeof(int)));
        CHK(cudaMemcpy(d_idx, h, nread * sizeof(int), cudaMemcpyHostToDevice));
        free(h);
        float ms = time_ms([&] { k_random<<<grid_for(nread), TPB>>>(d_in, d_idx, d_out, nread); });
        double useful = nread * 4.0, moved = nread * 32.0;
        printf("random,%ld,%ld,%d,%.6f,%.2f,%.2f\n", -1L, nread, 32, ms,
               useful / (ms * 1e-3) / 1e9, moved / (ms * 1e-3) / 1e9);
    }

    // ---------------- B. alignment sweep ----------------
    {
        long nread = NBUF - 64;
        for (int off : {0, 1, 2, 4, 8, 16, 31}) {
            float ms = time_ms([&] { k_offset<<<grid_for(nread), TPB>>>(d_in, d_out, nread, off); });
            printf("offset,%d,%ld,%.6f,%.2f\n", off, nread, ms,
                   nread * 4.0 / (ms * 1e-3) / 1e9);
        }
    }

    // ---------------- C. lane permutation ----------------
    {
        int hperm[64] = {17, 3, 28, 9, 22, 0, 14, 31, 6, 25, 11, 2, 19, 30, 8, 15,
                          1, 26, 12, 21, 5, 29, 10, 18, 24, 7, 13, 27, 4, 20, 16, 23};
        for (int i = 0; i < 32; ++i) hperm[32 + i] = i;   // the control table
        CHK(cudaMalloc(&d_perm, 64 * sizeof(int)));
        CHK(cudaMemcpy(d_perm, hperm, sizeof(hperm), cudaMemcpyHostToDevice));
        long nread = NBUF;
        for (int mode = 0; mode < 5; ++mode) {
            float ms = time_ms([&] { k_permute<<<grid_for(nread), TPB>>>(d_in, d_out, nread, mode, d_perm); });
            printf("permute,%d,%ld,%.6f,%.2f\n", mode, nread, ms,
                   nread * 4.0 / (ms * 1e-3) / 1e9);
        }
    }

    // ---------------- D. AoS vs SoA ----------------
    {
        long n = NBUF / 4;                       // 16 Mi particles = 256 MB as AoS
        Particle *d_p = (Particle *)d_in;        // reuse the same 256 MB
        float *x = d_in, *y = d_in + n, *z = d_in + 2 * n, *w = d_in + 3 * n;

        float ms;
        ms = time_ms([&] { k_aos1<<<grid_for(n), TPB>>>(d_p, d_out, n); });
        printf("layout,aos1,%ld,%.6f,%.2f,%.2f\n", n, ms,
               n * 4.0 / (ms * 1e-3) / 1e9, n * 16.0 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_soa1<<<grid_for(n), TPB>>>(x, d_out, n); });
        printf("layout,soa1,%ld,%.6f,%.2f,%.2f\n", n, ms,
               n * 4.0 / (ms * 1e-3) / 1e9, n * 4.0 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_aos4<<<grid_for(n), TPB>>>(d_p, d_out, n); });
        printf("layout,aos4,%ld,%.6f,%.2f,%.2f\n", n, ms,
               n * 16.0 / (ms * 1e-3) / 1e9, n * 16.0 / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { k_soa4<<<grid_for(n), TPB>>>(x, y, z, w, d_out, n); });
        printf("layout,soa4,%ld,%.6f,%.2f,%.2f\n", n, ms,
               n * 16.0 / (ms * 1e-3) / 1e9, n * 16.0 / (ms * 1e-3) / 1e9);
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_idx); cudaFree(d_perm);
    return 0;
}
