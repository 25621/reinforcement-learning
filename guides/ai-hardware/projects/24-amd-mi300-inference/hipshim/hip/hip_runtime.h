// A HIP-to-CUDA shim, used here as a TEST, not as a way to ship code.
//
// The point: if `hipify.py`'s port is faithful, then mapping every HIP name
// straight back onto its CUDA original must reproduce the exact behaviour of
// the original source. Compiling the ported file with `nvcc -I hipshim` and
// diffing its output against the original binary's output is that test.
//
// AMD ships the real thing in reverse (`hip_runtime.h` maps onto ROCm on AMD
// cards and onto CUDA on NVIDIA cards), which is exactly why HIP source is
// portable in the first place. This file is a 40-line version of that idea.

#pragma once
#include <cuda_runtime.h>

// --- types ---------------------------------------------------------------
#define hipError_t          cudaError_t
#define hipEvent_t          cudaEvent_t
#define hipStream_t         cudaStream_t
#define hipDeviceProp_t     cudaDeviceProp
#define hipFuncAttributes   cudaFuncAttributes

// --- enums ---------------------------------------------------------------
#define hipSuccess                  cudaSuccess
#define hipMemcpyHostToDevice       cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost       cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice     cudaMemcpyDeviceToDevice

// --- runtime API ---------------------------------------------------------
#define hipMalloc                   cudaMalloc
#define hipMallocHost               cudaMallocHost
#define hipFree                     cudaFree
#define hipFreeHost                 cudaFreeHost
#define hipMemcpy                   cudaMemcpy
#define hipMemcpyAsync              cudaMemcpyAsync
#define hipMemcpyToSymbol           cudaMemcpyToSymbol
#define hipMemcpyFromSymbol         cudaMemcpyFromSymbol
#define hipMemset                   cudaMemset
#define hipMemGetInfo               cudaMemGetInfo
#define hipDeviceSynchronize        cudaDeviceSynchronize
#define hipGetLastError             cudaGetLastError
#define hipGetErrorName             cudaGetErrorName
#define hipGetErrorString           cudaGetErrorString
#define hipGetDeviceCount           cudaGetDeviceCount
#define hipGetDeviceProperties      cudaGetDeviceProperties
#define hipDriverGetVersion         cudaDriverGetVersion
#define hipRuntimeGetVersion        cudaRuntimeGetVersion
#define hipFuncGetAttributes        cudaFuncGetAttributes
#define hipEventCreate              cudaEventCreate
#define hipEventDestroy             cudaEventDestroy
#define hipEventRecord              cudaEventRecord
#define hipEventSynchronize         cudaEventSynchronize
#define hipEventElapsedTime         cudaEventElapsedTime
#define hipOccupancyMaxActiveBlocksPerMultiprocessor \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor
