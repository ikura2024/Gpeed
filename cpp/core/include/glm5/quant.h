#pragma once

#include <cstdint>
#include <vector>

#include "glm5/common.h"

namespace glm5 {

// Block geometry constants (must match ggml).
constexpr int QK_K = 256;  // super-block size for K-quants / IQ-quants

// Number of bytes to read for a single block of `t`.
size_t dtype_block_bytes(DType t);

// Dequantizes `n` elements from `src` into `dst` (float). `n` must be a multiple
// of the block size. Throws if the type is not implemented yet.
void dequantize(DType t, const void* src, float* dst, uint64_t n);

// Same as `dequantize` but never spawns a parallel region. Callers that already
// parallelize (e.g. over output rows) should use this to avoid nested teams.
void dequantize_row(DType t, const void* src, float* dst, uint64_t n);

}  // namespace glm5
