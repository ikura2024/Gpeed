#include "glm5/kv_cache.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace glm5 {
namespace {

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x0200u : 0u));
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t m = man >> shift;
        if (man & (1u << (shift - 1))) ++m;
        return static_cast<uint16_t>(sign | m);
    }
    uint32_t m = man >> 13;
    if (man & 0x1000u) {
        ++m;
        if (m == 0x400u) {
            m = 0;
            if (exp + 1 >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | m);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    if ((x & 0x7F800000u) != 0x7F800000u) x += 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<uint16_t>(x >> 16);
}

inline float amax_of(const float* x) {
    float amax = 0.0f;
    for (int i = 0; i < kKvBlock; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

// ---- per-block quantizers -------------------------------------------------

inline void quant_bf16(const float* x, uint8_t* out) {
    uint16_t* d = reinterpret_cast<uint16_t*>(out);
    for (int i = 0; i < kKvBlock; ++i) d[i] = f32_to_bf16(x[i]);
}

inline void quant_q8_0(const float* x, uint8_t* out) {
    const float amax = amax_of(x);
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = f32_to_f16(d);
    std::memcpy(out, &dh, 2);
    int8_t* q = reinterpret_cast<int8_t*>(out + 2);
    for (int i = 0; i < kKvBlock; ++i) {
        int v = static_cast<int>(std::lround(x[i] * id));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = static_cast<int8_t>(v);
    }
}

inline void quant_q4_0(const float* x, uint8_t* out) {
    float amax = 0.0f, mx = 0.0f;
    for (int i = 0; i < kKvBlock; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) {
            amax = a;
            mx = x[i];
        }
    }
    const float d = mx / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = f32_to_f16(d);
    std::memcpy(out, &dh, 2);
    uint8_t* qs = out + 2;
    for (int j = 0; j < kKvBlock / 2; ++j) {
        int v0 = static_cast<int>(x[j] * id + 8.5f);
        int v1 = static_cast<int>(x[j + kKvBlock / 2] * id + 8.5f);
        if (v0 < 0) v0 = 0;
        if (v0 > 15) v0 = 15;
        if (v1 < 0) v1 = 0;
        if (v1 > 15) v1 = 15;
        qs[j] = static_cast<uint8_t>(v0 | (v1 << 4));
    }
}

inline void quant_q2_0(const float* x, uint8_t* out) {
    const float amax = amax_of(x);
    const float d = amax / 1.5f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint16_t dh = f32_to_f16(d);
    std::memcpy(out, &dh, 2);
    uint8_t* qs = out + 2;
    std::memset(qs, 0, kKvBlock / 4);
    for (int i = 0; i < kKvBlock; ++i) {
        int v = static_cast<int>(std::lround(x[i] * id + 1.5f));
        if (v < 0) v = 0;
        if (v > 3) v = 3;
        qs[i >> 2] |= static_cast<uint8_t>(v << ((i & 3) * 2));
    }
}

// ---- fused accumulated ops ------------------------------------------------

inline void dot_bf16(const uint8_t* q, const float* v, float& acc) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(q);
    for (int i = 0; i < kKvBlock; ++i) acc += bf16_to_f32(s[i]) * v[i];
}

inline void dot_q8_0(const uint8_t* q, const float* v, float& acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = f16_to_f32(dh);
    const int8_t* s = reinterpret_cast<const int8_t*>(q + 2);
    for (int i = 0; i < kKvBlock; ++i) acc += d * static_cast<float>(s[i]) * v[i];
}

inline void dot_q4_0(const uint8_t* q, const float* v, float& acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = q + 2;
    for (int j = 0; j < kKvBlock / 2; ++j) {
        acc += d * static_cast<float>((qs[j] & 0x0F) - 8) * v[j];
        acc += d * static_cast<float>((qs[j] >> 4) - 8) * v[j + kKvBlock / 2];
    }
}

inline void dot_q2_0(const uint8_t* q, const float* v, float& acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = q + 2;
    for (int i = 0; i < kKvBlock; ++i) {
        const int vv = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
        acc += d * (static_cast<float>(vv) - 1.5f) * v[i];
    }
}

inline void axpy_bf16(const uint8_t* q, float a, float* acc) {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(q);
    for (int i = 0; i < kKvBlock; ++i) acc[i] += a * bf16_to_f32(s[i]);
}

inline void axpy_q8_0(const uint8_t* q, float a, float* acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = a * f16_to_f32(dh);
    const int8_t* s = reinterpret_cast<const int8_t*>(q + 2);
    for (int i = 0; i < kKvBlock; ++i) acc[i] += d * static_cast<float>(s[i]);
}

inline void axpy_q4_0(const uint8_t* q, float a, float* acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = a * f16_to_f32(dh);
    const uint8_t* qs = q + 2;
    for (int j = 0; j < kKvBlock / 2; ++j) {
        acc[j] += d * static_cast<float>((qs[j] & 0x0F) - 8);
        acc[j + kKvBlock / 2] += d * static_cast<float>((qs[j] >> 4) - 8);
    }
}

inline void axpy_q2_0(const uint8_t* q, float a, float* acc) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const float d = a * f16_to_f32(dh);
    const uint8_t* qs = q + 2;
    for (int i = 0; i < kKvBlock; ++i) {
        const int vv = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
        acc[i] += d * (static_cast<float>(vv) - 1.5f);
    }
}

}  // namespace

const char* kv_dtype_name(KvDtype dt) {
    switch (dt) {
        case KvDtype::BF16: return "bf16";
        case KvDtype::Q8_0: return "q8";
        case KvDtype::Q4_0: return "q4";
        case KvDtype::Q2_0: return "q2";
    }
    return "?";
}

bool kv_dtype_from_string(const std::string& s, KvDtype* out) {
    if (s == "bf16" || s == "bf16_0") { *out = KvDtype::BF16; return true; }
    if (s == "q8" || s == "q8_0") { *out = KvDtype::Q8_0; return true; }
    if (s == "q4" || s == "q4_0") { *out = KvDtype::Q4_0; return true; }
    if (s == "q2" || s == "q2_0") { *out = KvDtype::Q2_0; return true; }
    return false;
}

size_t kv_block_bytes(KvDtype dt) {
    switch (dt) {
        case KvDtype::BF16: return kKvBlock * 2;
        case KvDtype::Q8_0: return 2 + kKvBlock;
        case KvDtype::Q4_0: return 2 + kKvBlock / 2;
        case KvDtype::Q2_0: return 2 + kKvBlock / 4;
    }
    return 0;
}

KvCache::KvCache(int dim, size_t max_pos, KvDtype dt)
    : dim_(dim), max_pos_(max_pos), dt_(dt) {
    if (dim_ <= 0 || (dim_ % kKvBlock) != 0)
        throw std::runtime_error("KvCache: dim must be a positive multiple of 32");
    const size_t blocks = static_cast<size_t>(dim_) / kKvBlock;
    row_bytes_ = blocks * kv_block_bytes(dt_);
    data_.assign(max_pos_ * row_bytes_, 0);
}

void KvCache::reset() {
    size_ = 0;
}

void KvCache::append(const float* row) {
    if (size_ >= max_pos_)
        throw std::runtime_error("KvCache: capacity exceeded");
    uint8_t* dst = data_.data() + size_ * row_bytes_;
    const size_t blocks = static_cast<size_t>(dim_) / kKvBlock;
    for (size_t b = 0; b < blocks; ++b) {
        const float* x = row + b * kKvBlock;
        uint8_t* o = dst + b * kv_block_bytes(dt_);
        switch (dt_) {
            case KvDtype::BF16: quant_bf16(x, o); break;
            case KvDtype::Q8_0: quant_q8_0(x, o); break;
            case KvDtype::Q4_0: quant_q4_0(x, o); break;
            case KvDtype::Q2_0: quant_q2_0(x, o); break;
        }
    }
    ++size_;
}

void KvCache::dot_row(size_t pos, const float* v, float* acc) const {
    const uint8_t* row = data_.data() + pos * row_bytes_;
    const size_t blocks = static_cast<size_t>(dim_) / kKvBlock;
    float a = *acc;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* q = row + b * kv_block_bytes(dt_);
        const float* vb = v + b * kKvBlock;
        switch (dt_) {
            case KvDtype::BF16: dot_bf16(q, vb, a); break;
            case KvDtype::Q8_0: dot_q8_0(q, vb, a); break;
            case KvDtype::Q4_0: dot_q4_0(q, vb, a); break;
            case KvDtype::Q2_0: dot_q2_0(q, vb, a); break;
        }
    }
    *acc = a;
}

void KvCache::axpy_row(size_t pos, float a, float* acc) const {
    const uint8_t* row = data_.data() + pos * row_bytes_;
    const size_t blocks = static_cast<size_t>(dim_) / kKvBlock;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* q = row + b * kv_block_bytes(dt_);
        float* ab = acc + b * kKvBlock;
        switch (dt_) {
            case KvDtype::BF16: axpy_bf16(q, a, ab); break;
            case KvDtype::Q8_0: axpy_q8_0(q, a, ab); break;
            case KvDtype::Q4_0: axpy_q4_0(q, a, ab); break;
            case KvDtype::Q2_0: axpy_q2_0(q, a, ab); break;
        }
    }
}

void KvCache::dequant_row(size_t pos, float* out) const {
    const uint8_t* row = data_.data() + pos * row_bytes_;
    const size_t blocks = static_cast<size_t>(dim_) / kKvBlock;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* q = row + b * kv_block_bytes(dt_);
        float* o = out + b * kKvBlock;
        float* acc = o;
        switch (dt_) {
            case KvDtype::BF16: {
                const uint16_t* s = reinterpret_cast<const uint16_t*>(q);
                for (int i = 0; i < kKvBlock; ++i) o[i] = bf16_to_f32(s[i]);
                break;
            }
            case KvDtype::Q8_0: {
                uint16_t dh;
                std::memcpy(&dh, q, 2);
                const float d = f16_to_f32(dh);
                const int8_t* s = reinterpret_cast<const int8_t*>(q + 2);
                for (int i = 0; i < kKvBlock; ++i) o[i] = d * static_cast<float>(s[i]);
                break;
            }
            case KvDtype::Q4_0: {
                uint16_t dh;
                std::memcpy(&dh, q, 2);
                const float d = f16_to_f32(dh);
                const uint8_t* qs = q + 2;
                for (int j = 0; j < kKvBlock / 2; ++j) {
                    o[j] = d * static_cast<float>((qs[j] & 0x0F) - 8);
                    o[j + kKvBlock / 2] = d * static_cast<float>((qs[j] >> 4) - 8);
                }
                break;
            }
            case KvDtype::Q2_0: {
                uint16_t dh;
                std::memcpy(&dh, q, 2);
                const float d = f16_to_f32(dh);
                const uint8_t* qs = q + 2;
                for (int i = 0; i < kKvBlock; ++i) {
                    const int vv = (qs[i >> 2] >> ((i & 3) * 2)) & 0x3;
                    o[i] = d * (static_cast<float>(vv) - 1.5f);
                }
                break;
            }
        }
        (void)acc;
    }
}

}  // namespace glm5
