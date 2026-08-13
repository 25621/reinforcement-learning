// Project 16 - the anatomy of a CUDA kernel, measured.
//
// Sections (selected by argv[1]):
//   main : correctness, the async-timing trap, block-size sweep, grid-stride
//          loop, PCIe end-to-end accounting, and the reuse crossover
//   oob  : two out-of-bounds launches - one that is silently tolerated and one
//          that kills the CUDA context. Run in its own process, because after
//          an illegal access every later CUDA call in the process fails.
//
// Everything is printed as CSV on stdout; run.py parses, tabulates and plots.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);              \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------- kernels

// The canonical one-element-per-thread vector add.
__global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // <- the bounds check
}

// Identical, minus the bounds check. Used only by the "oob" section.
__global__ void vadd_nocheck(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// Grid-stride loop: a fixed number of threads, each handling many elements.
__global__ void vadd_gridstride(const float *a, const float *b, float *c,
                                int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// Heat the GPU up. An idle card sits at ~164 MHz; the first measurement of a
// cold run is otherwise ~35% slow for no reason at all.
__global__ void k_spin(float *sink, int iters) {
    float x = threadIdx.x * 1e-6f;
    for (int i = 0; i < iters; ++i) x = fmaf(x, 1.0000001f, 1e-7f);
    if (x == 12345.0f) sink[0] = x;
}

static void warm_up(float *sink) {
    for (int i = 0; i < 250; ++i) k_spin<<<152, 256>>>(sink, 200000);
    CK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------- timing

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

template <typename F>
static float time_ms(F f, int reps = 20) {
    Timer t;
    f();                                     // one untimed call
    CK(cudaDeviceSynchronize());
    t.start();
    for (int i = 0; i < reps; ++i) f();
    return t.stop() / reps;
}

// ---------------------------------------------------------------- main run

static const int N = 1 << 26;                // 64 Mi floats = 256 MB per array
static const double BYTES = 3.0 * N * 4.0;   // 2 reads + 1 write

static void section_main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("#device,%s,%d.%d,%d,%d\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.l2CacheSize);
    printf("#n,%d\n", N);

    // ---- host data, page-locked so the PCIe numbers are the good case
    float *ha, *hb, *hc;
    CK(cudaMallocHost(&ha, (size_t)N * 4));
    CK(cudaMallocHost(&hb, (size_t)N * 4));
    CK(cudaMallocHost(&hc, (size_t)N * 4));
    for (int i = 0; i < N; ++i) {
        ha[i] = (float)(i % 1000) * 0.001f;
        hb[i] = (float)(i % 7) * 0.1f;
    }

    float *da, *db, *dc, *sink;
    CK(cudaMalloc(&da, (size_t)N * 4));
    CK(cudaMalloc(&db, (size_t)N * 4));
    CK(cudaMalloc(&dc, (size_t)N * 4));
    CK(cudaMalloc(&sink, 4));

    CK(cudaMemcpy(da, ha, (size_t)N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb, (size_t)N * 4, cudaMemcpyHostToDevice));

    warm_up(sink);

    // ---- A. correctness against the CPU
    vadd<<<(N + 255) / 256, 256>>>(da, db, dc, N);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hc, dc, (size_t)N * 4, cudaMemcpyDeviceToHost));
    double maxerr = 0.0;
    for (int i = 0; i < N; ++i)
        maxerr = fmax(maxerr, fabs((double)hc[i] - ((double)ha[i] + hb[i])));
    printf("check,%.3e\n", maxerr);

    // ---- B. the async trap: wall time with and without a synchronize
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        // (i) host clock, no sync - the launch returns immediately
        double t0 = omp_get_wtime();
        vadd<<<(N + 255) / 256, 256>>>(da, db, dc, N);
        double t_nosync = (omp_get_wtime() - t0) * 1000.0;
        // (ii) host clock, with sync - the honest number
        CK(cudaDeviceSynchronize());
        t0 = omp_get_wtime();
        vadd<<<(N + 255) / 256, 256>>>(da, db, dc, N);
        CK(cudaDeviceSynchronize());
        double t_sync = (omp_get_wtime() - t0) * 1000.0;
        printf("async,%.4f,%.4f\n", t_nosync, t_sync);
    }

    // ---- C. block size sweep, one element per thread
    for (int bs = 32; bs <= 1024; bs *= 2) {
        int grid = (N + bs - 1) / bs;
        float ms = time_ms([&] { vadd<<<grid, bs>>>(da, db, dc, N); });
        printf("block,%d,%d,%.4f,%.2f\n", bs, grid, ms, BYTES / (ms * 1e6));
    }
    CK(cudaGetLastError());

    // ---- D. grid-stride loop: fewer threads, each doing more
    for (int waves = 1; waves <= 64; waves *= 2) {
        int blocks = p.multiProcessorCount * waves;
        float ms = time_ms([&] { vadd_gridstride<<<blocks, 256>>>(da, db, dc, N); });
        printf("stride,%d,%d,%.4f,%.2f\n", waves, blocks, ms,
               BYTES / (ms * 1e6));
    }
    CK(cudaGetLastError());

    float best_ms = time_ms([&] { vadd<<<(N + 255) / 256, 256>>>(da, db, dc, N); });

    // ---- E. the whole trip: PCIe in, compute, PCIe out, vs the CPU
    Timer t;
    t.start();
    for (int i = 0; i < 5; ++i) {
        CK(cudaMemcpy(da, ha, (size_t)N * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, hb, (size_t)N * 4, cudaMemcpyHostToDevice));
    }
    float h2d_ms = t.stop() / 5.0f;
    t.start();
    for (int i = 0; i < 5; ++i)
        CK(cudaMemcpy(hc, dc, (size_t)N * 4, cudaMemcpyDeviceToHost));
    float d2h_ms = t.stop() / 5.0f;

    // CPU reference, single thread and all threads
    double t0 = omp_get_wtime();
    for (int i = 0; i < N; ++i) hc[i] = ha[i] + hb[i];
    double cpu1_ms = (omp_get_wtime() - t0) * 1000.0;

    int threads = omp_get_max_threads();
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) hc[i] = ha[i] + hb[i];
        best = fmin(best, (omp_get_wtime() - t0) * 1000.0);
    }
    printf("e2e,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n", h2d_ms, best_ms, d2h_ms,
           cpu1_ms, best, threads);

    // ---- F. how many kernel calls before the trip pays for itself
    for (int k = 1; k <= 1024; k *= 2) {
        double gpu = h2d_ms + k * best_ms + d2h_ms;
        double cpu = k * best;
        printf("reuse,%d,%.3f,%.3f\n", k, gpu, cpu);
    }

    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); CK(cudaFree(sink));
    CK(cudaFreeHost(ha)); CK(cudaFreeHost(hb)); CK(cudaFreeHost(hc));
}

// ---------------------------------------------------------------- oob run

// Launch the check-free kernel with `grid*block` threads over an array of `n`
// elements and report what the runtime says. Called twice, and the second call
// is expected to poison the context - so this section runs in its own process.
static void oob_case(const char *label, int n, int overshoot) {
    float *da, *db, *dc;
    if (cudaMalloc(&da, (size_t)n * 4) != cudaSuccess) { printf("oob,%s,alloc-failed\n", label); return; }
    CK(cudaMalloc(&db, (size_t)n * 4));
    CK(cudaMalloc(&dc, (size_t)n * 4));
    CK(cudaMemset(da, 0, (size_t)n * 4));
    CK(cudaMemset(db, 0, (size_t)n * 4));

    int bs = 256;
    int grid = (n + overshoot + bs - 1) / bs;
    int threads = grid * bs;
    vadd_nocheck<<<grid, bs>>>(da, db, dc, n);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync = cudaDeviceSynchronize();
    printf("oob,%s,%d,%d,%d,%s,%s\n", label, n, threads, threads - n,
           cudaGetErrorName(launch), cudaGetErrorName(sync));
    fflush(stdout);
    if (sync != cudaSuccess) return;         // context is dead; do not continue
    cudaFree(da); cudaFree(db); cudaFree(dc);
}

static void section_oob() {
    // 1) a few threads past the end. The allocator hands out whole pages, so
    //    the extra writes land in slack the program owns but never asked for.
    oob_case("slack", 1000, 0);
    // 2) a megabyte past the end. Now it is a different page, and the MMU
    //    notices.
    oob_case("far", 1000, 1 << 20);
}

int main(int argc, char **argv) {
    const char *mode = (argc > 1) ? argv[1] : "main";
    if (mode[0] == 'o') section_oob();
    else section_main();
    return 0;
}
