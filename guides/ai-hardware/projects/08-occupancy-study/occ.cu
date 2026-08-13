// Project 08 - occupancy: what the calculator says, what the GPU actually did,
// and how little either one predicts performance.
//
// "Occupancy" = resident warps on an SM / the most warps that SM can hold.
// Nsight reports two flavours and they are not the same number:
//   * THEORETICAL - an upper bound computable before the kernel runs, from
//     registers per thread, shared memory per block, and block size.
//   * ACHIEVED    - the time-average of what really happened, which is lower
//     whenever you did not launch enough blocks to fill the machine.
//
// Nsight Compute is unusable here (no counter permission - see the README), so
// we measure achieved occupancy ourselves, using the same definition Nsight
// uses: integrate resident warps over time and divide by the maximum.
//   achieved = sum_over_blocks(warps_in_block * block_lifetime_cycles)
//              / sum_over_SMs(SM_busy_cycles * max_warps_per_SM)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAXSM 256
#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

__device__ unsigned long long d_warpcycles[MAXSM];  // resident-warp x cycles
__device__ unsigned long long d_first[MAXSM];       // first block start
__device__ unsigned long long d_last[MAXSM];        // last block end
__device__ int d_blocks[MAXSM];                     // blocks that landed here
__device__ int d_resident[MAXSM];
__device__ int d_peak[MAXSM];                       // peak concurrent blocks

// %smid is a read-only special register naming the SM this warp runs on.
// There is no C API for it; you have to ask in PTX assembly.
__device__ __forceinline__ unsigned smid() {
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

struct Tag { unsigned sm; unsigned long long t0; };

__device__ __forceinline__ Tag enter() {
    __shared__ Tag tag;
    if (threadIdx.x == 0) {
        tag.sm = smid();
        tag.t0 = clock64();
        atomicMin(&d_first[tag.sm], tag.t0);
        atomicAdd(&d_blocks[tag.sm], 1);
        int now = atomicAdd(&d_resident[tag.sm], 1) + 1;
        atomicMax(&d_peak[tag.sm], now);
    }
    __syncthreads();
    return tag;
}

__device__ __forceinline__ void leave(Tag tag, int warps) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t1 = clock64();
        atomicMax(&d_last[tag.sm], t1);
        atomicAdd(&d_warpcycles[tag.sm], (t1 - tag.t0) * warps);
        atomicSub(&d_resident[tag.sm], 1);
    }
}

// ---------------------------------------------------------------------------
// WORKLOAD 1 - latency bound. Each thread walks a pointer chain in global
// memory: every load's address is the previous load's result, so one thread
// can never have two loads in flight. The only way to keep the memory system
// busy is to have many warps doing this at once. Occupancy is the whole game.
// ---------------------------------------------------------------------------
__global__ void chase(const int *next, float *out, int steps, int nthreads) {
    Tag tg = enter();
    int p = (blockIdx.x * blockDim.x + threadIdx.x) % nthreads;
    float acc = 0.f;
    for (int s = 0; s < steps; ++s) { p = next[p]; acc += p; }
    if (acc == 1.2345e30f) out[0] = acc;
    leave(tg, blockDim.x / 32);
}

// Identical walk, plus dynamic shared memory. The arithmetic does not change;
// only the SM's ability to host many blocks at once does.
extern __shared__ char smem[];
__global__ void chase_smem(const int *next, float *out, int steps, int nthreads) {
    Tag tg = enter();
    smem[threadIdx.x] = (char)threadIdx.x;      // touch it so it is not elided
    int p = (blockIdx.x * blockDim.x + threadIdx.x) % nthreads;
    float acc = 0.f;
    for (int s = 0; s < steps; ++s) { p = next[p]; acc += p; }
    if (acc == 1.2345e30f) out[0] = acc + smem[0];
    leave(tg, blockDim.x / 32);
}

// ---------------------------------------------------------------------------
// WORKLOAD 2 - compute bound, with a knob for instruction-level parallelism.
// CHAINS independent accumulator chains per thread, so each thread has CHAINS
// FMAs in flight at once instead of one. Past some point the chains need more
// live registers than the compiler can spare and occupancy starts to fall --
// which of the two effects wins is the experiment.
// ---------------------------------------------------------------------------
template <int CHAINS>
__global__ void ilp(float *out, int iters) {
    Tag tg = enter();
    float a[CHAINS];
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) a[c] = 1e-6f * (threadIdx.x + c);
    const float b = 1.0000001f, k = 1e-9f;
    for (int i = 0; i < iters; ++i)
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) a[c] = fmaf(a[c], b, k);
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < CHAINS; ++c) s += a[c];
    if (s == 1.2345e30f) out[0] = s;
    leave(tg, blockDim.x / 32);
}

// ---------------------------------------------------------------------------
static int g_sms, g_maxwarps;

static void reset_counters() {
    unsigned long long zl[MAXSM], big[MAXSM];
    int zi[MAXSM];
    for (int i = 0; i < MAXSM; ++i) { zl[i] = 0; big[i] = ~0ULL; zi[i] = 0; }
    CHK(cudaMemcpyToSymbol(d_warpcycles, zl, sizeof zl));
    CHK(cudaMemcpyToSymbol(d_last, zl, sizeof zl));
    CHK(cudaMemcpyToSymbol(d_first, big, sizeof big));
    CHK(cudaMemcpyToSymbol(d_blocks, zi, sizeof zi));
    CHK(cudaMemcpyToSymbol(d_resident, zi, sizeof zi));
    CHK(cudaMemcpyToSymbol(d_peak, zi, sizeof zi));
}

struct Occ { double achieved; int peak_blocks; int sms_used; };

static Occ read_occupancy() {
    unsigned long long wc[MAXSM], f[MAXSM], l[MAXSM];
    int nb[MAXSM], pk[MAXSM];
    CHK(cudaMemcpyFromSymbol(wc, d_warpcycles, sizeof wc));
    CHK(cudaMemcpyFromSymbol(f, d_first, sizeof f));
    CHK(cudaMemcpyFromSymbol(l, d_last, sizeof l));
    CHK(cudaMemcpyFromSymbol(nb, d_blocks, sizeof nb));
    CHK(cudaMemcpyFromSymbol(pk, d_peak, sizeof pk));

    // The denominator uses the LONGEST SM's busy window for every SM: an SM
    // that received no work still counts as idle capacity, which is exactly
    // what "achieved occupancy for this kernel" should penalise.
    unsigned long long span = 0;
    double num = 0; int used = 0, peak = 0;
    for (int i = 0; i < g_sms; ++i) {
        if (nb[i] == 0) continue;
        ++used;
        if (pk[i] > peak) peak = pk[i];
        unsigned long long s = l[i] - f[i];
        if (s > span) span = s;
        num += (double)wc[i];
    }
    double den = (double)span * g_maxwarps * g_sms;
    Occ o; o.achieved = den > 0 ? num / den : 0; o.peak_blocks = peak;
    o.sms_used = used;
    return o;
}

template <typename F>
static double theoretical(F f, int tpb, size_t dyn) {
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, tpb, dyn);
    return nb * ((tpb + 31) / 32) / (double)g_maxwarps;
}

template <typename F>
static int regs_of(F f) {
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, f); return a.numRegs;
}

static const int *G_next; static float *G_out;
static int G_steps, G_nthreads, G_tpb, G_blocks;
static size_t G_dyn;
static void go_chase() { chase<<<G_blocks, G_tpb>>>(G_next, G_out, G_steps, G_nthreads); }
static void go_chase_smem() {
    chase_smem<<<G_blocks, G_tpb, G_dyn>>>(G_next, G_out, G_steps, G_nthreads);
}

static float run_timed(void (*go)()) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    go(); CHK(cudaDeviceSynchronize());          // warm up (counters reset after)
    reset_counters();
    cudaEventRecord(s);
    go();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    CHK(cudaGetLastError());
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    g_sms = p.multiProcessorCount;
    g_maxwarps = p.maxThreadsPerMultiProcessor / p.warpSize;
    printf("#device,%s,%d.%d,%d,%d,%d,%d,%zu\n", p.name, p.major, p.minor,
           g_sms, g_maxwarps, p.maxThreadsPerMultiProcessor,
           p.regsPerMultiprocessor, (size_t)p.sharedMemPerMultiprocessor);

    // A random cycle: no prefetcher and no cache can predict the next address.
    const int NT = 1 << 20;
    int *hn = (int *)malloc(NT * sizeof(int)), *cyc = (int *)malloc(NT * sizeof(int));
    for (int i = 0; i < NT; ++i) hn[i] = i;
    unsigned st = 12345;
    for (int i = NT - 1; i > 0; --i) {                    // Fisher-Yates shuffle
        st = st * 1664525u + 1013904223u;
        int j = st % (i + 1);
        int t = hn[i]; hn[i] = hn[j]; hn[j] = t;
    }
    for (int i = 0; i < NT; ++i) cyc[hn[i]] = hn[(i + 1) % NT];
    int *dn; float *dout;
    CHK(cudaMalloc(&dn, NT * sizeof(int))); CHK(cudaMalloc(&dout, 1024));
    CHK(cudaMemcpy(dn, cyc, NT * sizeof(int), cudaMemcpyHostToDevice));
    G_next = dn; G_out = dout; G_nthreads = NT; G_steps = 512;

    // ---- A: block size, with a grid big enough to fill the GPU either way ----
    printf("#section,blocksize\n");
    for (int tpb = 32; tpb <= 1024; tpb *= 2) {
        G_tpb = tpb; G_blocks = g_sms * 32;
        double th = theoretical(chase, tpb, 0);
        float ms = run_timed(go_chase);
        Occ o = read_occupancy();
        double loads = (double)G_blocks * tpb * G_steps;
        printf("blocksize,%d,%d,%.4f,%.4f,%d,%.6f,%.3f\n", tpb, G_blocks, th,
               o.achieved, o.peak_blocks, ms, loads / (ms / 1e3) / 1e9);
    }

    // ---- B: grid size. Theoretical occupancy CANNOT change here. ----
    printf("#section,gridsize\n");
    G_tpb = 256;
    for (int nb = 1; nb <= g_sms * 32; nb *= 2) {
        G_blocks = nb;
        double th = theoretical(chase, G_tpb, 0);
        float ms = run_timed(go_chase);
        Occ o = read_occupancy();
        double loads = (double)nb * G_tpb * G_steps;
        printf("gridsize,%d,%d,%.4f,%.4f,%d,%.6f,%.3f\n", nb, G_tpb, th,
               o.achieved, o.peak_blocks, ms, loads / (ms / 1e3) / 1e9);
    }

    // ---- C: shared memory as a pure occupancy throttle ----
    printf("#section,sharedmem\n");
    G_tpb = 256; G_blocks = g_sms * 32;
    int kbs[] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48};
    for (int i = 0; i < 10; ++i) {
        G_dyn = (size_t)kbs[i] * 1024;
        // leave room for the kernel's own static __shared__ Tag
        if (G_dyn + 256 > p.sharedMemPerBlock) break;
        double th = theoretical(chase_smem, G_tpb, G_dyn);
        float ms = run_timed(go_chase_smem);
        Occ o = read_occupancy();
        double loads = (double)G_blocks * G_tpb * G_steps;
        printf("sharedmem,%d,%d,%.4f,%.4f,%d,%.6f,%.3f\n", kbs[i], G_tpb, th,
               o.achieved, o.peak_blocks, ms, loads / (ms / 1e3) / 1e9);
    }

    // ---- D: instruction-level parallelism vs occupancy ----
    // Two ways to keep the FMA pipeline fed while each FMA waits ~6 cycles for
    // the previous one: MANY WARPS (occupancy), or MANY INDEPENDENT CHAINS PER
    // THREAD (instruction-level parallelism). This sweeps both at once.
    printf("#section,ilp\n");
    G_tpb = 64;                       // 2 warps per block
    const int ITERS = 20000;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    #define RUN(C, BPS) do { \
        int nb = g_sms * (BPS); \
        double th = theoretical(ilp<C>, G_tpb, 0); \
        double thc = (BPS) * (G_tpb / 32) / (double)g_maxwarps; \
        int regs = regs_of(ilp<C>); \
        ilp<C><<<nb, G_tpb>>>(dout, 200); CHK(cudaDeviceSynchronize()); \
        reset_counters(); \
        cudaEventRecord(s); ilp<C><<<nb, G_tpb>>>(dout, ITERS); \
        cudaEventRecord(e); cudaEventSynchronize(e); \
        float ms; cudaEventElapsedTime(&ms, s, e); CHK(cudaGetLastError()); \
        Occ o = read_occupancy(); \
        double flop = 2.0 * (double)nb * G_tpb * ITERS * (C); \
        printf("ilp,%d,%d,%.4f,%.4f,%.4f,%d,%.6f,%.4f,%d\n", (C), (BPS), \
               th < thc ? th : thc, th, o.achieved, o.peak_blocks, ms, \
               flop / (ms / 1e3) / 1e12, regs); \
    } while (0)
    #define SWEEP(C) RUN(C,1); RUN(C,2); RUN(C,4); RUN(C,8); RUN(C,16); RUN(C,32)
    SWEEP(1); SWEEP(4); SWEEP(16); SWEEP(64);
    return 0;
}
