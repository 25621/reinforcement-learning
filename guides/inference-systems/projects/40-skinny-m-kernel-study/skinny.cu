// skinny.cu -- cuBLAS reference for project 40.
//
// PyTorch's own cuBLAS refuses this card (cuBLAS 13 dropped Pascal), but the
// system CUDA 12.0 toolkit still supports sm_61, so the vendor library is
// reachable from a small C++ program:
//
//   nvcc -O3 -arch=sm_61 skinny.cu -o skinny -lcublas
//
// Prints CSV: kernel,m,k,n,ms,gflops,gbs
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "cuda error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
  exit(1); } } while (0)
#define CB(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cublas error %d at line %d\n", (int)s, __LINE__); exit(1); } } while (0)

__global__ void k_spin(float *out, int iters) {
  float acc = 0.f;
  for (int i = 0; i < iters; i++) acc = acc * 1.0000001f + 1.f;
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static float *spin_buf = nullptr;
static void warm() {                       // same clock-stabilising trick as enginelib
  if (!spin_buf) CK(cudaMalloc(&spin_buf, 256 * 256 * sizeof(float)));
  for (int i = 0; i < 20; i++) k_spin<<<256, 256>>>(spin_buf, 20000);
  CK(cudaDeviceSynchronize());
}

template <typename F>
static double time_ms(F f, int reps) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; i++) f();
  CK(cudaDeviceSynchronize());
  warm();
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++) f();
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms / reps;
}

int main(int argc, char **argv) {
  int ms_list[] = {1, 2, 4, 8, 16, 32, 64, 128};
  int shapes[][2] = {{8192, 8192}, {1024, 1024}};
  cublasHandle_t h;
  CB(cublasCreate(&h));
  printf("kernel,m,k,n,ms,gflops,gbs\n");
  for (auto &shape : shapes) {
    int K = shape[0], N = shape[1];
    float *dx, *dw, *dy;
    CK(cudaMalloc(&dw, (size_t)K * N * sizeof(float)));
    CK(cudaMalloc(&dx, (size_t)128 * K * sizeof(float)));
    CK(cudaMalloc(&dy, (size_t)128 * N * sizeof(float)));
    std::vector<float> hw((size_t)K * N);
    for (size_t i = 0; i < hw.size(); i++) hw[i] = (float)((i * 1103515245u + 12345u) % 1000) / 1000.f - 0.5f;
    CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(dx, 0, (size_t)128 * K * sizeof(float)));
    const float alpha = 1.f, beta = 0.f;
    for (int M : ms_list) {
      int reps = (K > 4096) ? 20 : 100;
      // Row-major y[M,N] = x[M,K] * w[K,N] is column-major
      // y'[N,M] = w'[N,K] * x'[K,M], i.e. swap the operands.
      double t = time_ms([&] {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, dw, N, dx, K, &beta, dy, N);
      }, reps);
      double flops = 2.0 * M * K * N;
      double bytes = (double)K * N * 4 + (double)M * K * 4 + (double)M * N * 4;
      printf("cublas_gemm,%d,%d,%d,%.6f,%.2f,%.2f\n", M, K, N, t,
             flops / t / 1e6, bytes / t / 1e6);
      if (M == 1) {
        // The dedicated matrix-vector routine: same maths, different kernel.
        double tv = time_ms([&] {
          cublasSgemv(h, CUBLAS_OP_T, N, K, &alpha, dw, N, dx, 1, &beta, dy, 1);
        }, reps);
        printf("cublas_gemv,%d,%d,%d,%.6f,%.2f,%.2f\n", M, K, N, tv,
               flops / tv / 1e6, bytes / tv / 1e6);
      }
    }
    CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(dy));
  }
  CB(cublasDestroy(h));
  return 0;
}
