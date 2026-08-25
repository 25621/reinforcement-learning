// Project 07 - the smallest possible Tensor Core program.
//
// run.py compiles this twice, at -arch=sm_61 and at -arch=sm_70, and reports
// what happens. The point is that Tensor Core support is not a runtime check
// you can add a fallback for: below compute capability 7.0 the `nvcuda`
// namespace does not exist at all, so the file does not even parse.
//
// `wmma` = Warp Matrix Multiply-Accumulate. The name says what it is: the unit
// of work is a whole WARP cooperating on one small MATRIX MULTIPLY, ACCUMULATED
// into an existing tile. A `fragment` is one warp's private slice of that tile;
// you never index it yourself because the hardware decides which lane holds
// which element.

#include <mma.h>
using namespace nvcuda;

__global__ void tiny_tensor_core_matmul(const half *A, const half *B, float *C) {
    // 16x16x16: this warp multiplies a 16x16 by a 16x16 and accumulates.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;

    wmma::fill_fragment(fc, 0.0f);
    wmma::load_matrix_sync(fa, A, 16);
    wmma::load_matrix_sync(fb, B, 16);
    wmma::mma_sync(fc, fa, fb, fc);          // <- one Tensor Core instruction
    wmma::store_matrix_sync(C, fc, 16, wmma::mem_row_major);
}

int main() { return 0; }
