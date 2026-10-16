#include "glm5/weights.h"

#include <cstring>
#include <stdexcept>

#include "glm5/quant.h"
#include "glm5/simd.h"

namespace glm5 {

WeightStore::WeightStore(const GgufModel& model, bool cache_all)
    : model_(&model), cache_all_(cache_all) {}

uint64_t WeightStore::size(const std::string& name) const {
    const TensorInfo* t = info(name);
    return t ? t->n_elements() : 0;
}

void WeightStore::read_slice(const std::string& name, uint64_t elem_off, uint64_t n, float* dst) {
    const TensorInfo* t = info(name);
    if (!t) throw std::runtime_error("weights: missing tensor " + name);

    const int bs = dtype_block_size(t->type);
    if (elem_off % static_cast<uint64_t>(bs) != 0 || n % static_cast<uint64_t>(bs) != 0) {
        throw std::runtime_error("weights: unaligned slice for " + name);
    }
    const size_t bytes = dtype_row_bytes(t->type, n);
    const uint64_t byte_off = elem_off / static_cast<uint64_t>(bs) * dtype_block_bytes(t->type);

    const GgufFile& f = model_->file(static_cast<size_t>(t->shard));
    std::vector<uint8_t> raw(bytes);
    f.read_at(t->offset + byte_off, bytes, raw.data());
    dequantize(t->type, raw.data(), dst, n);
}

void WeightStore::read_expert(const std::string& name, int expert, float* dst) {
    const TensorInfo* t = info(name);
    if (!t) throw std::runtime_error("weights: missing tensor " + name);
    if (t->dims.size() != 3) throw std::runtime_error("weights: not a 3D expert tensor: " + name);
    const uint64_t per = t->dims[0] * t->dims[1];
    read_slice(name, static_cast<uint64_t>(expert) * per, per, dst);
}

const float* WeightStore::get(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second.data();

    const TensorInfo* t = info(name);
    if (!t) return nullptr;

    const uint64_t n = t->n_elements();
    std::vector<uint8_t> raw(dtype_row_bytes(t->type, n));
    model_->read_tensor(*t, raw.data());

    std::vector<float> f(n);
    dequantize(t->type, raw.data(), f.data(), n);
    cached_bytes_ += n * sizeof(float);
    auto [ins, ok] = cache_.emplace(name, std::move(f));
    return ins->second.data();
}

const uint8_t* WeightStore::payload(const std::string& name, const TensorInfo& t) const {
    auto it = resident_.find(name);
    if (it != resident_.end()) return it->second.data();
    const GgufFile& f = model_->file(static_cast<size_t>(t.shard));
    const uint8_t* base = f.data_ptr();
    return base ? base + t.offset : nullptr;
}

size_t WeightStore::pin_nonexpert() {
    for (const std::string& name : model_->tensor_names()) {
        // Routed experts stay on disk / in the expert cache.
        if (name.find("exps.weight") != std::string::npos) continue;
        const TensorInfo* t = info(name);
        if (!t || t->type == DType::F32) continue;  // F32 tensors are small and cached via get()
        const size_t bytes = dtype_row_bytes(t->type, t->n_elements());
        if (bytes == 0 || bytes < 4096) continue;
        std::vector<uint8_t> buf(bytes);
        model_->read_tensor(*t, buf.data());
        resident_bytes_ += bytes;
        resident_.emplace(name, std::move(buf));
    }
    return resident_bytes_;
}

void WeightStore::matmul_quant(const std::string& name, uint64_t row0, int in_dim, int out_dim,
                               const float* X, int n_tok, float* Y) {
    if (n_tok <= 0 || out_dim <= 0) return;
    const TensorInfo* t = info(name);
    if (!t) throw std::runtime_error("weights: missing tensor " + name);

    const int bs = dtype_block_size(t->type);
    if (bs != 1 && in_dim % bs != 0)
        throw std::runtime_error("weights: row not block aligned for " + name);
    const size_t row_bytes = dtype_row_bytes(t->type, static_cast<uint64_t>(in_dim));
    const uint64_t byte_off = row0 * static_cast<uint64_t>(row_bytes);

    const GgufFile& f = model_->file(static_cast<size_t>(t->shard));
    const uint8_t* src = payload(name, *t);
    std::vector<uint8_t> tmp;
    if (src) {
        src += byte_off;
    } else {
        tmp.resize(row_bytes * static_cast<size_t>(out_dim));
        f.read_at(t->offset + byte_off, tmp.size(), tmp.data());
        src = tmp.data();
    }

    // Two regimes:
    //  * small batch (X fits comfortably in cache) -> parallelize over output
    //    rows, each row dequantized once and reused by every token;
    //  * large batch -> block over rows so X is not streamed once per row.
    const size_t x_bytes = static_cast<size_t>(n_tok) * static_cast<size_t>(in_dim) * sizeof(float);
    const bool row_parallel = x_bytes <= (16u << 20);
    if (row_parallel) {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<float> rowbuf(static_cast<size_t>(in_dim));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int j = 0; j < out_dim; ++j) {
                dequantize_row(t->type, src + row_bytes * static_cast<size_t>(j), rowbuf.data(),
                           static_cast<uint64_t>(in_dim));
                for (int s = 0; s < n_tok; ++s)
                    Y[static_cast<size_t>(s) * out_dim + j] =
                        dot_f32(rowbuf.data(), X + static_cast<size_t>(s) * in_dim, in_dim);
            }
        }
        return;
    }

    const int JB = 64;
    std::vector<float> wbuf(static_cast<size_t>(JB) * in_dim);
    const int64_t nb = (out_dim + JB - 1) / JB;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        for (int64_t b = 0; b < nb; ++b) {
            const int jb = static_cast<int>(b * JB);
            const int jn = std::min(JB, out_dim - jb);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int j = 0; j < jn; ++j) {
                dequantize_row(t->type, src + row_bytes * static_cast<size_t>(jb + j),
                           wbuf.data() + static_cast<size_t>(j) * in_dim,
                           static_cast<uint64_t>(in_dim));
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int idx = 0; idx < n_tok * jn; ++idx) {
                const int s = idx / jn;
                const int j = idx - s * jn;
                Y[static_cast<size_t>(s) * out_dim + jb + j] =
                    dot_f32(wbuf.data() + static_cast<size_t>(j) * in_dim,
                            X + static_cast<size_t>(s) * in_dim, in_dim);
            }
        }
    }
}

void WeightStore::matvec_quant(const std::string& name, uint64_t row0, int in_dim, int out_dim,
                               const float* x, float* dst) {
    const TensorInfo* t = info(name);
    if (!t) throw std::runtime_error("weights: missing tensor " + name);

    const int bs = dtype_block_size(t->type);
    if (bs != 1 && in_dim % bs != 0)
        throw std::runtime_error("weights: row not block aligned for " + name);
    const size_t row_bytes = dtype_row_bytes(t->type, static_cast<uint64_t>(in_dim));
    const uint64_t byte_off = row0 * static_cast<uint64_t>(row_bytes);

    const GgufFile& f = model_->file(static_cast<size_t>(t->shard));
    const uint8_t* src = payload(name, *t);
    std::vector<uint8_t> tmp;
    if (src) {
        src += byte_off;
    } else {
        tmp.resize(row_bytes * static_cast<size_t>(out_dim));
        f.read_at(t->offset + byte_off, tmp.size(), tmp.data());
        src = tmp.data();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> row(static_cast<size_t>(in_dim));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int j = 0; j < out_dim; ++j) {
            dequantize_row(t->type, src + row_bytes * static_cast<size_t>(j), row.data(),
                       static_cast<uint64_t>(in_dim));
            dst[j] = dot_f32(row.data(), x, in_dim);
        }
    }
}

}  // namespace glm5
