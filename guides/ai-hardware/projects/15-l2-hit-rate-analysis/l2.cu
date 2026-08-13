// Project 15 - measuring the L2 hit rate of an attention kernel WITHOUT a
// profiler.
//
// Nsight Compute cannot read this machine's hardware counters
// (ERR_NVGPUCTRPERM - see project 6), so `l2_tex_hit_rate` is unavailable.
// It turns out not to be needed. L2 hits and DRAM hits travel at very
// different speeds, so a kernel's achieved bandwidth is a thermometer for its
// hit rate, as long as the thermometer is calibrated:
//
//        1 / achieved  =  h / B_L2  +  (1 - h) / B_DRAM
//   =>          h      = (1/B_DRAM - 1/achieved) / (1/B_DRAM - 1/B_L2)
//
// A. calibrate  - measure B_L2 and B_DRAM with the same kernel
// B. validate   - a kernel whose TRUE hit rate is known by construction;
//                 does the thermometer recover it?
// C. attention  - hit rate vs sequence length (does K/V fit in L2?)
// D. attention  - hit rate vs BLOCK ORDER, with the footprint held fixed

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

#define D  64        // head dimension
#define BQ 64        // queries per block
#define BK 32        // keys per shared-memory tile

// ---------------------------------------------------------------------------
// A. Calibration: stream a working set of `n` floats, `rounds` times.
// ---------------------------------------------------------------------------
__global__ void k_stream(const float *__restrict__ in, float *out, long n,
                         int rounds) {
    long stride = gridDim.x * (long)blockDim.x;
    float acc = 0.f;
    for (int r = 0; r < rounds; ++r)
        for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride)
            acc += in[i];
    if (acc == 1.2345e30f) out[0] = acc;
}

// ---------------------------------------------------------------------------
// B. Known hit rate by construction. Each WARP decides, per iteration, to read
// either the small hot buffer (always resident in L2) or the huge cold one
// (never resident). The decision is warp-uniform and the 32 lanes read 32
// consecutive floats, so every access is perfectly coalesced either way - the
// ONLY thing that varies is locality. True hit rate = hot fraction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned hashu(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// hot_mask / cold_mask are (number of 32-float lines - 1), and both region
// sizes are powers of two, so the address maths is a shift and an AND. An
// earlier version used `%` on runtime values; integer division is ~20 cycles
// and made this kernel arithmetic-bound, which silently broke the whole
// measurement - the achieved bandwidth stopped depending on locality at all.
__global__ void k_mix(const float *__restrict__ hot, const float *__restrict__ cold,
                      float *out, unsigned hot_mask, unsigned cold_mask,
                      unsigned thresh, int iters) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned warp = gid >> 5, lane = gid & 31;
    float acc = 0.f;
    unsigned h = hashu(warp * 2654435761u + 12345u);
    for (int it = 0; it < iters; ++it) {
        h = hashu(h);
        const float *base = (h < thresh) ? hot : cold;
        unsigned mask = (h < thresh) ? hot_mask : cold_mask;
        acc += base[((h >> 8) & mask) * 32 + lane];
    }
    if (acc == 1.2345e30f) out[0] = acc;
}

// ---------------------------------------------------------------------------
// C + D. Attention. One block handles BQ queries and streams the whole of K
// and V past them, keeping a running softmax (the FlashAttention structure).
//
// COMPUTE=true  : the real thing.
// COMPUTE=false : the memory twin - identical global-memory address stream,
//                 trivial arithmetic. Times the memory system alone, so the
//                 hit-rate thermometer is not contaminated by FLOPs.
//
// `stagger` shifts each block's starting tile. Every block still reads all of
// K and V; only the ORDER changes. That holds the footprint and the byte count
// exactly fixed while destroying the synchrony between blocks.
// ---------------------------------------------------------------------------
template <bool COMPUTE>
__global__ void k_attn(const float *__restrict__ Q, const float *__restrict__ K,
                       const float *__restrict__ V, float *__restrict__ O,
                       int S, int stagger) {
    __shared__ float Ks[BK][D], Vs[BK][D];
    const int tid = threadIdx.x;                    // 0..BQ-1, one query each
    const int nqb = S / BQ;                         // query blocks per head
    const int head = blockIdx.x / nqb;
    const int qb = blockIdx.x % nqb;
    const long hoff = (long)head * S * D;           // this head's slice
    const int qrow = qb * BQ + tid;
    const int ntiles = S / BK;
    Q += hoff; K += hoff; V += hoff; O += hoff;

    float q[D], acc[D];
#pragma unroll
    for (int d = 0; d < D; ++d) { q[d] = Q[(long)qrow * D + d]; acc[d] = 0.f; }
    float m = -1e30f, l = 0.f;

    const int start = stagger ? (qb * BK) % ntiles : 0;

    for (int t = 0; t < ntiles; ++t) {
        int tile = (start + t) % ntiles;
        // cooperative load: 64 threads move BK*D = 2048 floats, coalesced
        for (int i = tid; i < BK * D; i += BQ) {
            int r = i / D, c = i % D;
            Ks[r][c] = K[((long)tile * BK + r) * D + c];
            Vs[r][c] = V[((long)tile * BK + r) * D + c];
        }
        __syncthreads();
        if (COMPUTE) {
#pragma unroll 1
            for (int j = 0; j < BK; ++j) {
                float s = 0.f;
#pragma unroll
                for (int d = 0; d < D; ++d) s = fmaf(q[d], Ks[j][d], s);
                s *= rsqrtf((float)D);
                float mn = fmaxf(m, s);
                float corr = __expf(m - mn), pj = __expf(s - mn);
                l = l * corr + pj;
#pragma unroll
                for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + pj * Vs[j][d];
                m = mn;
            }
        } else {
            // touch the tile so the loads cannot be deleted, but do no work
            acc[0] += Ks[tid & (BK - 1)][0] + Vs[tid & (BK - 1)][0];
            l += 1.f;
        }
        __syncthreads();
    }
    float inv = 1.f / (l + 1e-20f);
#pragma unroll
    for (int d = 0; d < D; ++d) O[(long)qrow * D + d] = acc[d] * inv;
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
static float time_ms(F launch, int iters, int reps = 3) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(); CHK(cudaDeviceSynchronize());
    float t[8];
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

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("#device,%s,%d.%d,%d,%zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, (size_t)p.l2CacheSize);

    const long NBIG = 1L << 26;                       // 256 MB
    float *big, *hot, *out;
    CHK(cudaMalloc(&big, NBIG * sizeof(float)));
    CHK(cudaMalloc(&hot, (1L << 17) * sizeof(float)));   // 512 KB
    CHK(cudaMalloc(&out, 1024 * sizeof(float)));
    CHK(cudaMemset(big, 0, NBIG * sizeof(float)));
    CHK(cudaMemset(hot, 0, (1L << 17) * sizeof(float)));

    for (int i = 0; i < 250; ++i) k_spin<<<19 * 8, 256>>>(out, 200000);
    CHK(cudaDeviceSynchronize());

    // ---------------- A. calibration ----------------
    // The two anchors of the thermometer, measured with ONE kernel so that
    // instruction overhead is identical and only locality differs.
    double B_L2 = 0, B_DRAM = 0;
    for (int shift = 15; shift <= 26; ++shift) {
        long n = 1L << shift;
        int rounds = (int)(1 + (1L << 24) / n);
        float ms = time_ms([&] { k_stream<<<19 * 16, 256>>>(big, out, n, rounds); }, 5, 5);
        double gbs = (double)n * rounds * 4 / (ms * 1e-3) / 1e9;
        printf("cal,%ld,%d,%.6f,%.2f\n", n * 4, rounds, ms, gbs);
        if (n * 4 == (1L << 19)) B_L2 = gbs;          // 512 KB working set
        if (shift == 26) B_DRAM = gbs;                // 256 MB working set
    }
    printf("#anchors,%.3f,%.3f\n", B_L2, B_DRAM);

    // ---------------- B. validation against a known hit rate ----------------
    {
        const int iters = 64;
        long threads = 19L * 16 * 256;
        double bytes = (double)threads * iters * 4;
        for (int pct : {0, 10, 25, 50, 75, 90, 100}) {
            unsigned thresh = (unsigned)((double)pct / 100.0 * 4294967295.0);
            float ms = time_ms([&] {
                k_mix<<<19 * 16, 256>>>(hot, big, out, (1u << 17) / 32 - 1,
                                        (unsigned)(NBIG / 32) - 1, thresh, iters);
            }, 10, 5);
            double gbs = bytes / (ms * 1e-3) / 1e9;
            double h = (1.0 / B_DRAM - 1.0 / gbs) / (1.0 / B_DRAM - 1.0 / B_L2);
            printf("valid,%d,%.6f,%.2f,%.4f\n", pct, ms, gbs, h);
        }
    }

    // ---------------- correctness: attention vs a CPU reference ----------------
    {
        const int S = 256;
        size_t sz = (size_t)S * D * sizeof(float);
        float *hQ = (float *)malloc(sz), *hK = (float *)malloc(sz);
        float *hV = (float *)malloc(sz), *hO = (float *)malloc(sz);
        unsigned st = 7u;
        auto rnd = [&] { st = st * 1103515245u + 12345u; return (float)((st >> 12) & 1023) / 1024.f - 0.5f; };
        for (int i = 0; i < S * D; ++i) { hQ[i] = rnd(); hK[i] = rnd(); hV[i] = rnd(); }
        float *Q, *K, *V, *O;
        CHK(cudaMalloc(&Q, sz)); CHK(cudaMalloc(&K, sz));
        CHK(cudaMalloc(&V, sz)); CHK(cudaMalloc(&O, sz));
        CHK(cudaMemcpy(Q, hQ, sz, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(K, hK, sz, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(V, hV, sz, cudaMemcpyHostToDevice));
        k_attn<true><<<S / BQ, BQ>>>(Q, K, V, O, S, 0);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(hO, O, sz, cudaMemcpyDeviceToHost));
        double worst = 0;
        float *sc = (float *)malloc(S * sizeof(float));
        for (int i = 0; i < S; ++i) {
            double mx = -1e30;
            for (int j = 0; j < S; ++j) {
                double s = 0; for (int d = 0; d < D; ++d) s += (double)hQ[i * D + d] * hK[j * D + d];
                sc[j] = (float)(s / sqrt((double)D)); if (sc[j] > mx) mx = sc[j];
            }
            double sum = 0; for (int j = 0; j < S; ++j) sum += exp(sc[j] - mx);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < S; ++j) o += exp(sc[j] - mx) * hV[j * D + d];
                o /= sum;
                double e = fabs(o - hO[i * D + d]);
                if (e > worst) worst = e;
            }
        }
        printf("#check,%d,%.3e\n", S, worst);
        free(hQ); free(hK); free(hV); free(hO); free(sc);
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    }

    // ---------------- C + D. attention ----------------
    for (int S : {512, 1024, 2048, 4096, 8192}) {
        // Enough heads that the grid is at least two full waves of blocks -
        // otherwise the kernel is latency-bound and its achieved bandwidth
        // measures occupancy, not cache behaviour.
        int nqb = S / BQ;
        int H = (228 + nqb - 1) / nqb;
        float *Q, *K, *V, *O;
        size_t sz = (size_t)H * S * D * sizeof(float);
        CHK(cudaMalloc(&Q, sz)); CHK(cudaMalloc(&K, sz));
        CHK(cudaMalloc(&V, sz)); CHK(cudaMalloc(&O, sz));
        CHK(cudaMemset(Q, 0, sz)); CHK(cudaMemset(K, 0, sz));
        CHK(cudaMemset(V, 0, sz));
        int nb = H * nqb;
        // bytes the kernel ASKS for: each block reads its Q tile, all of K and
        // V for its head, and writes its O tile
        double asked = (double)nb * (2.0 * BQ * D * 4 + 2.0 * S * D * 4);
        double compulsory = 4.0 * (double)H * S * D * 4;    // Q,K,V,O once each
        double kv_head = 2.0 * S * D * 4;                   // K+V for one head
        int it = S >= 4096 ? 3 : 10;
        for (int stag : {0, 1}) {
            float ms_c = time_ms([&] { k_attn<true><<<nb, BQ>>>(Q, K, V, O, S, stag); }, it);
            float ms_m = time_ms([&] { k_attn<false><<<nb, BQ>>>(Q, K, V, O, S, stag); }, it);
            double gbs = asked / (ms_m * 1e-3) / 1e9;
            double h = (1.0 / B_DRAM - 1.0 / gbs) / (1.0 / B_DRAM - 1.0 / B_L2);
            double flops = 4.0 * (double)H * S * S * D;     // QK^T and PV
            printf("attn,%d,%d,%d,%d,%.0f,%.0f,%.0f,%.6f,%.6f,%.2f,%.4f,%.3f\n",
                   S, stag, H, nb, asked, compulsory, kv_head, ms_c, ms_m, gbs, h,
                   flops / (ms_c * 1e-3) / 1e12);
        }
        cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    }

    cudaFree(big); cudaFree(hot); cudaFree(out);
    return 0;
}
