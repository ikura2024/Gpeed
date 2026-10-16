#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace glm5 {

// Storage precision for the per-position KV / indexer caches.
//   BF16 : raw bfloat16 (2 bytes / element, no scaling)
//   Q8_0 : ggml Q8_0 layout (fp16 scale + 32 int8)          -> 34 bytes / 32
//   Q4_0 : ggml Q4_0 layout (fp16 scale + 16 packed nibbles) -> 18 bytes / 32
//   Q2_0 : custom symmetric 2-bit, mid-tread levels {-1.5,-0.5,0.5,1.5}
//          (fp16 scale d = amax / 1.5 + 8 packed bytes)       -> 10 bytes / 32
enum class KvDtype : int { BF16 = 0, Q8_0 = 1, Q4_0 = 2, Q2_0 = 3 };

constexpr int kKvBlock = 32;  // elements per block for every KvDtype

const char* kv_dtype_name(KvDtype dt);
bool kv_dtype_from_string(const std::string& s, KvDtype* out);
size_t kv_block_bytes(KvDtype dt);

// A growable stack of quantized rows, addressed by absolute position.
// Rows are `dim` floats wide (must be a multiple of 32) and stored quantized.
class KvCache {
public:
    KvCache() = default;
    KvCache(int dim, size_t max_pos, KvDtype dt);

    void reset();
    void append(const float* row);   // quantize row at position `size()`
    size_t size() const { return size_; }
    void truncate(size_t n) { size_ = n; }
    size_t capacity() const { return max_pos_; }

    int dim() const { return dim_; }
    KvDtype dtype() const { return dt_; }
    size_t bytes() const { return data_.size(); }

    // acc += dot(row[pos], v)
    void dot_row(size_t pos, const float* v, float* acc) const;
    // acc += a * row[pos]
    void axpy_row(size_t pos, float a, float* acc) const;
    // expand row[pos] into `out` (dim floats)
    void dequant_row(size_t pos, float* out) const;

private:
    int dim_ = 0;
    size_t max_pos_ = 0;
    size_t size_ = 0;
    KvDtype dt_ = KvDtype::BF16;
    size_t row_bytes_ = 0;
    std::vector<uint8_t> data_;
};

}  // namespace glm5
