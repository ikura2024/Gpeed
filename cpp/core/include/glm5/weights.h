#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "glm5/gguf.h"

namespace glm5 {

// Dequantizes and caches (as F32) the non-expert weights of a model.
// Expert tensors are far too large to cache and are read on demand per expert.
class WeightStore {
public:
    WeightStore() = default;
    explicit WeightStore(const GgufModel& model, bool cache_all = true);

    const GgufModel& model() const { return *model_; }

    bool has(const std::string& name) const { return model_->find_tensor(name) != nullptr; }
    const TensorInfo* info(const std::string& name) const { return model_->find_tensor(name); }

    // Cached F32 view of `name` (dequantized on first use). nullptr if missing.
    const float* get(const std::string& name);

    // Element count of `name`.
    uint64_t size(const std::string& name) const;

    // Reads one expert from a 3D expert tensor into `dst` (ne0*ne1 floats).
    // `expert` indexes ne2.
    void read_expert(const std::string& name, int expert, float* dst);

    // Reads `n` elements starting at element offset `elem_off` (block aligned) into dst.
    void read_slice(const std::string& name, uint64_t elem_off, uint64_t n, float* dst);

    // dst[out] = sum_i W[i + in_dim*out] * x[i], evaluated straight from the quantized
    // payload starting at row `row0` (no F32 materialization, no RAM round-trip).
    // Used for expert tensors, which are far too large to cache.
    void matvec_quant(const std::string& name, uint64_t row0, int in_dim, int out_dim,
                      const float* x, float* dst);

    // Y[t][out] = sum_i W[i + in_dim*out] * X[t][i] for t in [0, n_tok).
    // Same input as matvec_quant; blocked so that `X` is read ~outer/JB times
    // instead of once per weight row.
    void matmul_quant(const std::string& name, uint64_t row0, int in_dim, int out_dim,
                      const float* X, int n_tok, float* Y);

    size_t cached_bytes() const { return cached_bytes_; }

    // Copies the raw (still quantized) payload of every non-expert tensor into RAM
    // so that per-token matvecs no longer stream them from disk. Routed experts are
    // excluded: they are far larger than RAM and are handled by the expert cache.
    // Returns the number of bytes pinned.
    size_t pin_nonexpert();

    size_t resident_bytes() const { return resident_bytes_; }

private:
    const GgufModel* model_ = nullptr;
    bool cache_all_;
    std::unordered_map<std::string, std::vector<float>> cache_;
    std::unordered_map<std::string, std::unique_ptr<std::vector<float>>> scratch_;
    size_t cached_bytes_ = 0;

    // Raw quantized payloads kept resident (bypasses the mmap / page cache).
    std::unordered_map<std::string, std::vector<uint8_t>> resident_;
    size_t resident_bytes_ = 0;

    // Base pointer of a tensor's quantized payload: the resident copy when pinned,
    // otherwise the zero-copy mmap view (nullptr when the file was not mapped).
    const uint8_t* payload(const std::string& name, const TensorInfo& t) const;
};

}  // namespace glm5
