// Project 09 - thread divergence: what it costs, when it costs nothing, and
// the one case where the compiler quietly makes it disappear.
//
// A warp is 32 threads sharing ONE instruction pointer. When an `if` sends
// some of them one way and the rest another, the hardware cannot run both at
// once: it runs path A with the path-B threads switched off, then path B with
// the path-A threads switched off. The work adds up; the parallelism does not.
//
// Four experiments:
//   1. WAYS distinct paths per warp, identical work in each  -> the cost
//   2. the SAME number of paths, but chosen per-warp not per-lane -> the fix
//   3. a branch small enough that the compiler predicates it away -> no cost
//   4. a data-dependent loop count, random vs sorted -> the real-world version

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

// Each branch body does exactly the same amount of arithmetic, with different
// constants so the compiler cannot merge two bodies into one.
#define BODY(n)                                                     \
    { const float c = 1.0f + (n + 1) * 1e-7f, k = (n + 1) * 1e-9f;  \
      for (int i = 0; i < iters; ++i) a = fmaf(a, c, k); }

#define BRANCH(n) if ((n) < WAYS && sel == (n)) BODY(n) else

// mode 0: sel depends on the LANE -> every warp splits WAYS ways
// mode 1: sel depends on the WARP -> every warp is uniform, same path count
template <int WAYS>
__global__ void diverge(float *out, int iters, int mode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = tid >> 5;
    int sel = (mode == 0) ? (lane % WAYS) : (warp % WAYS);
    float a = 1e-6f * tid;

    BRANCH(0) BRANCH(1) BRANCH(2) BRANCH(3) BRANCH(4) BRANCH(5)
    BRANCH(6) BRANCH(7) BRANCH(8) BRANCH(9) BRANCH(10) BRANCH(11)
    BRANCH(12) BRANCH(13) BRANCH(14) BRANCH(15) BRANCH(16) BRANCH(17)
    BRANCH(18) BRANCH(19) BRANCH(20) BRANCH(21) BRANCH(22) BRANCH(23)
    BRANCH(24) BRANCH(25) BRANCH(26) BRANCH(27) BRANCH(28) BRANCH(29)
    BRANCH(30) BRANCH(31) { }

    if (a == 1.2345e30f) out[0] = a;
}

// ---- experiment 3: a branch too small to be worth branching over ----------
// Both arms are one FMA. The compiler emits both, each guarded by a predicate
// register, and the warp executes them back to back with no jump at all.
// Cost: 2 instructions instead of 1 - no 2x, no warp stall.
__global__ void tiny_branch(float *out, int iters, int diverge_on) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool take = diverge_on ? (lane < 16) : (blockIdx.x & 1);
    float a = 1e-6f * tid;
    for (int i = 0; i < iters; ++i)
        a = take ? fmaf(a, 1.0000001f, 1e-9f) : fmaf(a, 1.0000002f, 2e-9f);
    if (a == 1.2345e30f) out[0] = a;
}

// ---- experiment 4: data-dependent trip counts ----------------------------
// Every thread loops a different number of times. A warp finishes when its
// SLOWEST thread finishes, so the warp pays for its maximum, not its average.
__global__ void ragged(const int *trip, float *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = trip[tid];
    float a = 1e-6f * tid;
    for (int i = 0; i < t; ++i) a = fmaf(a, 1.0000001f, 1e-9f);
    if (a == 1.2345e30f) out[0] = a;
}

// --------------------------------------------------------------------------
static float timeit(void (*go)(), int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    go(); CHK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < reps; ++i) go();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    CHK(cudaGetLastError());
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / reps;
}

static float *G_out; static int G_iters, G_mode, G_blocks, G_tpb;
static const int *G_trip;

#define MKGO(W) static void go_d##W() { \
    diverge<W><<<G_blocks, G_tpb>>>(G_out, G_iters, G_mode); }
MKGO(1) MKGO(2) MKGO(4) MKGO(8) MKGO(16) MKGO(32)
static void go_tiny() { tiny_branch<<<G_blocks, G_tpb>>>(G_out, G_iters, G_mode); }
static void go_ragged() { ragged<<<G_blocks, G_tpb>>>(G_trip, G_out); }

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("#device,%s,%d.%d,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount);

    G_tpb = 256; G_blocks = p.multiProcessorCount * 8; G_iters = 4000;
    CHK(cudaMalloc(&G_out, 1024));
    int threads = G_blocks * G_tpb;

    // Spin the GPU up first. A card sitting at 164 MHz idle takes a few tens of
    // milliseconds to reach its boost clock, and whichever measurement runs
    // first would otherwise look slow for a reason that has nothing to do with
    // branching.
    G_mode = 1;
    for (int i = 0; i < 200; ++i) go_d1();
    CHK(cudaDeviceSynchronize());

    // ---- 1 + 2: divergent vs warp-aligned, same number of paths ----
    void (*gos[6])() = {go_d1, go_d2, go_d4, go_d8, go_d16, go_d32};
    int ways[6] = {1, 2, 4, 8, 16, 32};
    for (int m = 0; m < 2; ++m) {
        G_mode = m;
        for (int i = 0; i < 6; ++i) {
            float ms = timeit(gos[i], 8);
            double useful = 2.0 * threads * G_iters;   // FLOPs anyone asked for
            printf("ways,%d,%d,%.6f,%.4f\n", ways[i], m, ms,
                   useful / (ms / 1e3) / 1e12);
        }
    }

    // ---- 3: predication ----
    for (int m = 0; m < 2; ++m) {
        G_mode = m;
        float ms = timeit(go_tiny, 8);
        double useful = 2.0 * threads * G_iters;
        printf("tiny,%d,%.6f,%.4f\n", m, ms, useful / (ms / 1e3) / 1e12);
    }

    // ---- 4: ragged loop counts, random order vs sorted ----
    int *ht = (int *)malloc(threads * sizeof(int));
    unsigned st = 987654321u;
    long total = 0;
    for (int i = 0; i < threads; ++i) {
        st = st * 1664525u + 1013904223u;
        ht[i] = 8 + (int)(st % 2000);         // 8..2007 iterations
        total += ht[i];
    }
    int *dt; CHK(cudaMalloc(&dt, threads * sizeof(int)));
    G_trip = dt;

    CHK(cudaMemcpy(dt, ht, threads * sizeof(int), cudaMemcpyHostToDevice));
    float ms_rand = timeit(go_ragged, 5);

    // "warp-iterations": what the hardware really runs is the sum, over warps,
    // of that warp's SLOWEST thread - not the sum of all trip counts.
    long warpmax_random = 0;
    for (int w = 0; w < threads / 32; ++w) {
        int mx = 0;
        for (int l = 0; l < 32; ++l) if (ht[w * 32 + l] > mx) mx = ht[w * 32 + l];
        warpmax_random += mx;
    }

    // sort ascending so each warp holds 32 near-identical trip counts
    for (int gap = threads / 2; gap > 0; gap /= 2)          // shell sort
        for (int i = gap; i < threads; ++i) {
            int v = ht[i], j = i;
            while (j >= gap && ht[j - gap] > v) { ht[j] = ht[j - gap]; j -= gap; }
            ht[j] = v;
        }
    CHK(cudaMemcpy(dt, ht, threads * sizeof(int), cudaMemcpyHostToDevice));
    float ms_sort = timeit(go_ragged, 5);

    long warpmax_sorted = 0;
    for (int w = 0; w < threads / 32; ++w) {
        int mx = 0;
        for (int l = 0; l < 32; ++l) if (ht[w * 32 + l] > mx) mx = ht[w * 32 + l];
        warpmax_sorted += mx;
    }
    printf("ragged,random,%.6f,%ld,%ld\n", ms_rand, total, warpmax_random);
    printf("ragged,sorted,%.6f,%ld,%ld\n", ms_sort, total, warpmax_sorted);
    return 0;
}
