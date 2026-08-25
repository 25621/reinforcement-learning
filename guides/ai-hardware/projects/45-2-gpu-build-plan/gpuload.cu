// heater.cu - a controllable, measurable GPU load for the power/thermal study.
//
// Every mode prints one CSV line per reporting interval, timestamped on the
// *same* clock Python's time.time() uses, so a telemetry sample taken by
// nvidia-smi can be matched to the work that was running at that instant:
//
//   sample,<unix_seconds>,<interval_s>,<achieved>,<tag>
//     achieved = GFLOP/s for compute modes, GB/s for the memory mode
//
// Modes:
//   compute <seconds>                  full-width FMA load (compute-bound)
//   memory  <seconds>                  full-width streaming load (DRAM-bound)
//   duty    <seconds> <on_ms> <off_ms> compute load, switched on and off
//   width   <sec_per_step>             compute load on 1..all SMs, stepwise
//
// Build: nvcc -O3 -arch=sm_61 heater.cu -o heater

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
    exit(1); } } while (0)

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// 128 fused multiply-adds per loop trip, all in registers: no memory traffic,
// so the only limit is the FP32 pipeline. This is the "worst case" a power
// supply has to survive.
__global__ void fma_load(float* sink, int trips) {
    float a = threadIdx.x * 1e-3f + 1.0f, b = 1.0000001f, c = 0.9999999f;
    float x0 = a, x1 = a + 1.f, x2 = a + 2.f, x3 = a + 3.f;
    for (int t = 0; t < trips; t++) {
#pragma unroll
        for (int k = 0; k < 32; k++) {
            x0 = fmaf(x0, b, c);
            x1 = fmaf(x1, b, c);
            x2 = fmaf(x2, b, c);
            x3 = fmaf(x3, b, c);
        }
    }
    if (x0 + x1 + x2 + x3 == 12345.6789f) sink[0] = x0;  // never taken
}

// Streaming copy: 8 bytes of DRAM traffic per element, almost no arithmetic.
__global__ void mem_load(const float* __restrict__ in, float* __restrict__ out,
                         size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i] * 1.0009f;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "compute";
    double seconds = argc > 2 ? atof(argv[2]) : 60.0;

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    CK(cudaSetDevice(0));
    const int sms = p.multiProcessorCount;
    const int threads = 256;
    const int blocks_full = sms * 8;          // 2048 threads/SM: fully occupied
    const int trips = 2000;                   // work per kernel launch

    float* sink; CK(cudaMalloc(&sink, sizeof(float) * 4));

    // FLOPs per launch: threads x trips x 32 unrolled x 4 fma x 2 flops each
    auto flops_of = [&](int blocks) {
        return (double)blocks * threads * trips * 32.0 * 4.0 * 2.0;
    };

    size_t n = (size_t)1 << 25;               // 128 MB in, 128 MB out
    float *d_in = nullptr, *d_out = nullptr;
    if (strcmp(mode, "memory") == 0) {
        CK(cudaMalloc(&d_in, n * sizeof(float)));
        CK(cudaMalloc(&d_out, n * sizeof(float)));
        CK(cudaMemset(d_in, 1, n * sizeof(float)));
    }

    printf("meta,%s,%d,%d,%d\n", p.name, sms, blocks_full, threads);
    fflush(stdout);

    // ---- width mode: how much of the chip is switching? ------------------
    // Runs the same kernel on 1, 2, 4 ... all SMs' worth of blocks, `seconds`
    // per step. Power at each step tells us how much of the board's draw is
    // fixed cost and how much is bought by doing work.
    if (strcmp(mode, "width") == 0) {
        const int width_steps[] = {1, 2, 4, 8, 19, 38, 76, 152};
        for (int i = 0; i < (int)(sizeof(width_steps) / sizeof(int)); i++) {
            int blocks = width_steps[i];
            double t_begin = now_s(), work = 0;
            while (now_s() - t_begin < seconds) {
                fma_load<<<blocks, threads>>>(sink, trips);
                CK(cudaDeviceSynchronize());
                work += flops_of(blocks);
            }
            double t = now_s(), dt = t - t_begin;
            printf("sample,%.3f,%.3f,%.2f,blocks=%d\n", t, dt, work / dt / 1e9,
                   blocks);
            fflush(stdout);
        }
        return 0;
    }

    // ---- steady modes: compute / memory / duty ---------------------------
    double on_ms = argc > 3 ? atof(argv[3]) : 100.0;
    double off_ms = argc > 4 ? atof(argv[4]) : 0.0;
    double t_start = now_s();
    double t_report = t_start;
    double work = 0;                            // flops or bytes since report

    while (now_s() - t_start < seconds) {
        if (strcmp(mode, "memory") == 0) {
            mem_load<<<blocks_full, threads>>>(d_in, d_out, n);
            CK(cudaDeviceSynchronize());
            work += 2.0 * n * sizeof(float);
        } else {
            // duty mode: keep the GPU busy for on_ms, then let it idle for
            // off_ms. Averaged over a second this delivers less work *and*
            // less power - the crude version of a power limit.
            double t_on = now_s();
            do {
                fma_load<<<blocks_full, threads>>>(sink, trips);
                CK(cudaDeviceSynchronize());
                work += flops_of(blocks_full);
            } while (strcmp(mode, "duty") == 0 && (now_s() - t_on) * 1000.0 < on_ms);
            if (strcmp(mode, "duty") == 0 && off_ms > 0)
                std::this_thread::sleep_for(
                    std::chrono::microseconds((long)(off_ms * 1000)));
        }

        double t = now_s(), dt = t - t_report;
        if (dt >= 1.0) {
            char tag[32];
            if (strcmp(mode, "duty") == 0)
                snprintf(tag, sizeof(tag), "on=%.0f,off=%.0f", on_ms, off_ms);
            else
                snprintf(tag, sizeof(tag), "full");
            printf("sample,%.3f,%.3f,%.2f,%s\n", t, dt, work / dt / 1e9, tag);
            fflush(stdout);
            work = 0; t_report = t;
        }
    }
    return 0;
}
