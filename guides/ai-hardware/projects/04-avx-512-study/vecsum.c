/* Project 04 - the same arithmetic written five ways.
 *
 * Two kernels:
 *   sum  : add up an array.            2 bytes read per FLOP -> memory-bound
 *   poly : 20 FMAs per element.        0.1 bytes read per FLOP -> compute-bound
 *
 * Five variants of each:
 *   scalar   - vectorisation switched off, one float at a time
 *   auto     - plain -O3, the compiler may vectorise if it is allowed to
 *   fastmath - -ffast-math, which permits reordering the additions
 *   avx2     - hand-written 8-wide intrinsics (256-bit ymm registers)
 *   avx512   - hand-written 16-wide intrinsics (512-bit zmm registers)
 *
 * Per-function attributes keep everything in one file, so the ONLY difference
 * between variants is the one flag named in the attribute.
 *
 * Build: gcc -O3 -march=native -o vecsum vecsum.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 20 coefficients -> 20 fused multiply-adds -> 40 FLOPs per element */
#define PDEG 20
static float COEF[PDEG];

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ sum */
__attribute__((optimize("no-tree-vectorize")))
float sum_scalar(const float *x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

float sum_auto(const float *x, size_t n) {          /* plain -O3 */
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

__attribute__((optimize("fast-math")))
float sum_fastmath(const float *x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

__attribute__((target("avx2")))
float sum_avx2(const float *x, size_t n) {
    /* four independent accumulators: one vaddps has ~4 cycles of latency but
       can start every cycle, so we need several in flight to keep the port busy */
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_loadu_ps(x + i));
        a1 = _mm256_add_ps(a1, _mm256_loadu_ps(x + i + 8));
        a2 = _mm256_add_ps(a2, _mm256_loadu_ps(x + i + 16));
        a3 = _mm256_add_ps(a3, _mm256_loadu_ps(x + i + 24));
    }
    a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    float tmp[8];
    _mm256_storeu_ps(tmp, a0);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) s += x[i];
    return s;
}

__attribute__((target("avx512f")))
float sum_avx512(const float *x, size_t n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_add_ps(a0, _mm512_loadu_ps(x + i));
        a1 = _mm512_add_ps(a1, _mm512_loadu_ps(x + i + 16));
        a2 = _mm512_add_ps(a2, _mm512_loadu_ps(x + i + 32));
        a3 = _mm512_add_ps(a3, _mm512_loadu_ps(x + i + 48));
    }
    a0 = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    float s = _mm512_reduce_add_ps(a0);
    for (; i < n; ++i) s += x[i];
    return s;
}

/* ----------------------------------------------------------------- poly */
/* Horner's rule: p = ((c0*x + c1)*x + c2)*x + ...  Each step is one FMA. */
__attribute__((optimize("no-tree-vectorize")))
float poly_scalar(const float *x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

float poly_auto(const float *x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

__attribute__((optimize("fast-math")))
float poly_fastmath(const float *x, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

__attribute__((target("avx2,fma")))
float poly_avx2(const float *x, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 p = _mm256_set1_ps(COEF[0]);
        for (int k = 1; k < PDEG; ++k)
            p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(COEF[k]));
        acc = _mm256_add_ps(acc, p);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

/* Same maths, four independent Horner chains in flight.
 * Horner is a SERIAL chain: each FMA needs the previous one's result, and an
 * FMA takes ~4 cycles to finish while a new one could start every cycle. One
 * chain therefore leaves ~3 of every 4 cycles empty. Four chains fill them. */
__attribute__((target("avx2,fma")))
float poly_avx2_ilp(const float *x, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(x + i), v1 = _mm256_loadu_ps(x + i + 8);
        __m256 v2 = _mm256_loadu_ps(x + i + 16), v3 = _mm256_loadu_ps(x + i + 24);
        __m256 c = _mm256_set1_ps(COEF[0]);
        __m256 p0 = c, p1 = c, p2 = c, p3 = c;
        for (int k = 1; k < PDEG; ++k) {
            __m256 ck = _mm256_set1_ps(COEF[k]);
            p0 = _mm256_fmadd_ps(p0, v0, ck);
            p1 = _mm256_fmadd_ps(p1, v1, ck);
            p2 = _mm256_fmadd_ps(p2, v2, ck);
            p3 = _mm256_fmadd_ps(p3, v3, ck);
        }
        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(p0, p1),
                                               _mm256_add_ps(p2, p3)));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

__attribute__((target("avx512f")))
float poly_avx512(const float *x, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512 v0 = _mm512_loadu_ps(x + i), v1 = _mm512_loadu_ps(x + i + 16);
        __m512 v2 = _mm512_loadu_ps(x + i + 32), v3 = _mm512_loadu_ps(x + i + 48);
        __m512 c = _mm512_set1_ps(COEF[0]);
        __m512 p0 = c, p1 = c, p2 = c, p3 = c;
        for (int k = 1; k < PDEG; ++k) {
            __m512 ck = _mm512_set1_ps(COEF[k]);
            p0 = _mm512_fmadd_ps(p0, v0, ck);
            p1 = _mm512_fmadd_ps(p1, v1, ck);
            p2 = _mm512_fmadd_ps(p2, v2, ck);
            p3 = _mm512_fmadd_ps(p3, v3, ck);
        }
        acc = _mm512_add_ps(acc, _mm512_add_ps(_mm512_add_ps(p0, p1),
                                               _mm512_add_ps(p2, p3)));
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
        float v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

/* --------------------------------------------------------------- driver */
typedef float (*kern_t)(const float *, size_t);

typedef struct { const char *name; kern_t fn; int needs_avx512; } variant_t;

/* The exact answer, accumulated in double. Neither variant computes this; it
   is only here so we can say which float answer is CLOSER to the truth. */
static double sum_reference(const float *x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double)x[i];
    return s;
}

static double poly_reference(const float *x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v = x[i], p = COEF[0];
        for (int k = 1; k < PDEG; ++k) p = p * v + COEF[k];
        s += p;
    }
    return s;
}

static double bench(kern_t fn, const float *x, size_t n, float *out) {
    /* enough repeats that each timed round is a few tens of milliseconds */
    long reps = (long)(1.2e8 / (double)n);
    if (reps < 3) reps = 3;
    if (reps > 20000) reps = 20000;
    float sink = 0.f;
    for (long r = 0; r < 2; ++r) sink += fn(x, n);          /* warm up */
    double best = 1e30;
    for (int round = 0; round < 5; ++round) {
        double t0 = now();
        for (long r = 0; r < reps; ++r) sink += fn(x, n);
        double dt = (now() - t0) / reps;
        if (dt < best) best = dt;
    }
    *out = fn(x, n);
    if (sink == 1.2345e30f) printf("#impossible\n");         /* keep `sink` live */
    return best;
}

int main(int argc, char **argv) {
    const int have512 = !!__builtin_cpu_supports("avx512f");
    const int have2 = !!__builtin_cpu_supports("avx2");
    printf("#cpu,avx2=%d,avx512f=%d\n", have2, have512);

    for (int k = 0; k < PDEG; ++k) COEF[k] = 0.5f + 0.01f * k;

    /* --force-avx512 skips the runtime check on purpose, so you can see what
       happens when 512-bit code meets a CPU that has no 512-bit registers.
       On this machine the kernel kills the process with SIGILL. */
    if (argc > 1 && strcmp(argv[1], "--force-avx512") == 0) {
        size_t n = 4096;
        float *x = aligned_alloc(64, n * sizeof(float));
        for (size_t i = 0; i < n; ++i) x[i] = 1.f;
        printf("#calling sum_avx512 without checking the CPU...\n");
        fflush(stdout);
        float s = sum_avx512(x, n);
        printf("#survived, sum=%f\n", s);
        free(x);
        return 0;
    }

    /* 8 KiB (fits L1), 128 KiB (L2), 4 MiB (L3), 128 MiB (DRAM only) */
    const size_t sizes[] = {2048, 32768, 1048576, 33554432};
    const char *where[] = {"L1", "L2", "L3", "DRAM"};

    variant_t sums[] = {
        {"scalar", sum_scalar, 0}, {"auto", sum_auto, 0},
        {"fastmath", sum_fastmath, 0}, {"avx2", sum_avx2, 0},
        {"avx512", sum_avx512, 1},
    };
    variant_t polys[] = {
        {"scalar", poly_scalar, 0}, {"auto", poly_auto, 0},
        {"fastmath", poly_fastmath, 0}, {"avx2", poly_avx2, 0},
        {"avx2_ilp", poly_avx2_ilp, 0}, {"avx512", poly_avx512, 1},
    };
    const int NS = sizeof(sums) / sizeof(sums[0]);
    const int NP = sizeof(polys) / sizeof(polys[0]);

    for (int si = 0; si < 4; ++si) {
        size_t n = sizes[si];
        float *x = aligned_alloc(64, n * sizeof(float));
        for (size_t i = 0; i < n; ++i) x[i] = 1.f / (float)(1 + (i % 977));

        double ref_sum = sum_reference(x, n), ref_poly = poly_reference(x, n);
        printf("ref,sum,%zu,%s,%.15e\n", n, where[si], ref_sum);
        printf("ref,poly,%zu,%s,%.15e\n", n, where[si], ref_poly);

        for (int v = 0; v < NS; ++v) {
            if (sums[v].needs_avx512 && !have512) {
                printf("sum,%s,%zu,%s,SKIPPED,0\n", sums[v].name, n, where[si]);
                continue;
            }
            float res;
            double t = bench(sums[v].fn, x, n, &res);
            /* 1 FLOP per element, 4 bytes read per element */
            printf("sum,%s,%zu,%s,%.9e,%.9f\n", sums[v].name, n, where[si], t, res);
        }
        for (int v = 0; v < NP; ++v) {
            if (polys[v].needs_avx512 && !have512) {
                printf("poly,%s,%zu,%s,SKIPPED,0\n", polys[v].name, n, where[si]);
                continue;
            }
            float res;
            double t = bench(polys[v].fn, x, n, &res);
            /* 40 FLOPs per element (20 FMAs), 4 bytes read per element */
            printf("poly,%s,%zu,%s,%.9e,%.9f\n", polys[v].name, n, where[si], t, res);
        }
        free(x);
    }
    return 0;
}
