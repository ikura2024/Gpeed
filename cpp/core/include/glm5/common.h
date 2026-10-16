#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace glm5 {

// ggml tensor types (subset relevant to GGUF inference; values match ggml).
enum class DType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    COUNT = 40,
};

const char* dtype_name(DType t);

// Number of elements encoded by one block of `t` (1 for non-blocked types).
int dtype_block_size(DType t);

// Bytes per block for blocked types, or bytes per element for scalar types.
// Returns 0 for types whose layout is not implemented.
size_t dtype_type_size(DType t);

// Bytes required to store `n_elements` of type `t`. Returns 0 if unsupported.
size_t dtype_row_bytes(DType t, uint64_t n_elements);

}  // namespace glm5
