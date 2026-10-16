#include "glm5/quant.h"

#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "glm5/quant_tables.h"

namespace glm5 {

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q8_K: return "Q8_K";
        case DType::IQ2_XXS: return "IQ2_XXS";
        case DType::IQ2_XS: return "IQ2_XS";
        case DType::IQ3_XXS: return "IQ3_XXS";
        case DType::IQ1_S: return "IQ1_S";
        case DType::IQ4_NL: return "IQ4_NL";
        case DType::IQ3_S: return "IQ3_S";
        case DType::IQ2_S: return "IQ2_S";
        case DType::IQ4_XS: return "IQ4_XS";
        case DType::I8: return "I8";
        case DType::I16: return "I16";
        case DType::I32: return "I32";
        case DType::I64: return "I64";
        case DType::F64: return "F64";
        case DType::IQ1_M: return "IQ1_M";
        case DType::BF16: return "BF16";
        case DType::TQ1_0: return "TQ1_0";
        case DType::TQ2_0: return "TQ2_0";
        default: return "UNKNOWN";
    }
}

int dtype_block_size(DType t) {
    switch (t) {
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q5_0:
        case DType::Q5_1:
        case DType::Q8_0:
        case DType::Q8_1:
            return 32;
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K:
        case DType::Q8_K:
        case DType::IQ2_XXS:
        case DType::IQ2_XS:
        case DType::IQ3_XXS:
        case DType::IQ1_S:
        case DType::IQ3_S:
        case DType::IQ2_S:
        case DType::IQ4_XS:
        case DType::IQ1_M:
        case DType::TQ1_0:
        case DType::TQ2_0:
            return QK_K;
        default:
            return 1;
    }
}

size_t dtype_type_size(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::F64: return 8;
        case DType::I8: return 1;
        case DType::I16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::Q4_0: return 2 + 16;               // d + 16 packed nibbles
        case DType::Q4_1: return 2 + 2 + 16;
        case DType::Q5_0: return 2 + 4 + 16;
        case DType::Q5_1: return 2 + 2 + 4 + 16;
        case DType::Q8_0: return 2 + 32;
        case DType::Q8_1: return 2 + 2 + 32;
        case DType::Q2_K: return 84;   // scales[256/16] + qs[256/4] + d + dmin (fp16)
        case DType::Q3_K: return 110;  // hmask[256/8] + qs[256/4] + scales[12] + d
        case DType::Q4_K: return 144;
        case DType::Q5_K: return 176;
        case DType::Q6_K: return 210;
        case DType::Q8_K: return 292;
        case DType::IQ2_XXS: return 66;
        case DType::IQ2_XS: return 74;
        case DType::IQ2_S: return 82;
        case DType::IQ3_XXS: return 98;  // d + qs (indices + scales/signs)
        case DType::IQ3_S: return 110;
        case DType::IQ1_S: return 50;
        case DType::IQ4_NL: return 18;
        case DType::IQ4_XS: return 136;
        default: return 0;
    }
}

size_t dtype_block_bytes(DType t) { return dtype_type_size(t); }

size_t dtype_row_bytes(DType t, uint64_t n_elements) {
    const int bs = dtype_block_size(t);
    const size_t tb = dtype_type_size(t);
    if (tb == 0) return 0;
    if (bs == 1) return static_cast<size_t>(n_elements) * tb;
    if (n_elements % static_cast<uint64_t>(bs) != 0) return 0;
    return static_cast<size_t>(n_elements / static_cast<uint64_t>(bs)) * tb;
}

namespace {

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // subnormal
            uint32_t e = 127 - 15 + 1;
            uint32_t m = man;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
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
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// --- dequantizers, one ggml block per call (no OpenMP) ---------------------

inline void blk_q8_0(const uint8_t* blk, float* y) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
#if defined(__AVX2__)
    const __m256 dv = _mm256_set1_ps(d);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    _mm256_storeu_ps(y + 0, _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
    _mm256_storeu_ps(y + 8,
                     _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
    _mm256_storeu_ps(y + 16, _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
    _mm256_storeu_ps(y + 24,
                     _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
#else
    for (int i = 0; i < 32; ++i) y[i] = d * static_cast<float>(q[i]);
#endif
}

inline void blk_iq3_xxs(const uint8_t* blk, float* y) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales_and_signs = qs + QK_K / 4;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32;
        std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = kSignsIq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 0]]);
            const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&kIq3xxsGrid[qs[2 * l + 1]]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * static_cast<float>(grid1[j]) *
                           ((signs & kMaskIq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] = db * static_cast<float>(grid2[j]) *
                           ((signs & kMaskIq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
    }
}

inline void blk_q6_K(const uint8_t* blk, float* y) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
    uint16_t dh;
    std::memcpy(&dh, blk + 208, 2);
    const float d = f16_to_f32(dh);
#if defined(__AVX2__)
    const __m256i m0F = _mm256_set1_epi8(0x0F);
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i sub32 = _mm256_set1_epi8(32);
    for (int n2 = 0; n2 < QK_K; n2 += 128) {
        const __m256i qla = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i qlb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

        const __m256i lo_a = _mm256_and_si256(qla, m0F);
        const __m256i lo_b = _mm256_and_si256(qlb, m0F);
        const __m256i hi_a = _mm256_and_si256(_mm256_srli_epi16(qla, 4), m0F);
        const __m256i hi_b = _mm256_and_si256(_mm256_srli_epi16(qlb, 4), m0F);

        const __m256i t1 = _mm256_slli_epi16(_mm256_and_si256(qhv, m03), 4);
        const __m256i t2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m03), 4);
        const __m256i t3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m03), 4);
        const __m256i t4 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m03), 4);

        const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(lo_a, t1), sub32);
        const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(lo_b, t2), sub32);
        const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(hi_a, t3), sub32);
        const __m256i q4 = _mm256_sub_epi8(_mm256_or_si256(hi_b, t4), sub32);

        const float s0 = d * static_cast<float>(sc[0]);
        const float s1 = d * static_cast<float>(sc[1]);
        const float s2 = d * static_cast<float>(sc[2]);
        const float s3 = d * static_cast<float>(sc[3]);
        const float s4 = d * static_cast<float>(sc[4]);
        const float s5 = d * static_cast<float>(sc[5]);
        const float s6 = d * static_cast<float>(sc[6]);
        const float s7 = d * static_cast<float>(sc[7]);

        // Each 32-value group covers 16 values with scale (a) and 16 with scale (b).
        const __m256 v0 = _mm256_set1_ps(s0), v1 = _mm256_set1_ps(s1);
        const __m256 v2 = _mm256_set1_ps(s2), v3 = _mm256_set1_ps(s3);
        const __m256 v4 = _mm256_set1_ps(s4), v5 = _mm256_set1_ps(s5);
        const __m256 v6 = _mm256_set1_ps(s6), v7 = _mm256_set1_ps(s7);

        const __m128i a1 = _mm256_castsi256_si128(q1), a1h = _mm256_extracti128_si256(q1, 1);
        const __m128i b1 = _mm256_castsi256_si128(q2), b1h = _mm256_extracti128_si256(q2, 1);
        const __m128i c1 = _mm256_castsi256_si128(q3), c1h = _mm256_extracti128_si256(q3, 1);
        const __m128i e1 = _mm256_castsi256_si128(q4), e1h = _mm256_extracti128_si256(q4, 1);

        _mm256_storeu_ps(y + 0, _mm256_mul_ps(v0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a1))));
        _mm256_storeu_ps(y + 8, _mm256_mul_ps(v0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a1, 8)))));
        _mm256_storeu_ps(y + 16, _mm256_mul_ps(v1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a1h))));
        _mm256_storeu_ps(y + 24, _mm256_mul_ps(v1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a1h, 8)))));

        _mm256_storeu_ps(y + 32, _mm256_mul_ps(v2, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1))));
        _mm256_storeu_ps(y + 40, _mm256_mul_ps(v2, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8)))));
        _mm256_storeu_ps(y + 48, _mm256_mul_ps(v3, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1h))));
        _mm256_storeu_ps(y + 56, _mm256_mul_ps(v3, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1h, 8)))));

        _mm256_storeu_ps(y + 64, _mm256_mul_ps(v4, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c1))));
        _mm256_storeu_ps(y + 72, _mm256_mul_ps(v4, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c1, 8)))));
        _mm256_storeu_ps(y + 80, _mm256_mul_ps(v5, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c1h))));
        _mm256_storeu_ps(y + 88, _mm256_mul_ps(v5, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c1h, 8)))));

        _mm256_storeu_ps(y + 96, _mm256_mul_ps(v6, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e1))));
        _mm256_storeu_ps(y + 104, _mm256_mul_ps(v6, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e1, 8)))));
        _mm256_storeu_ps(y + 112, _mm256_mul_ps(v7, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e1h))));
        _mm256_storeu_ps(y + 120, _mm256_mul_ps(v7, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e1h, 8)))));

        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
#else
    for (int n2 = 0; n2 < QK_K; n2 += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
#endif
}

inline void blk_q2_K(const uint8_t* blk, float* y) {
    const uint8_t* scales = blk;
    const uint8_t* q = blk + 16;
    uint16_t dh, dmh;
    std::memcpy(&dh, blk + 80, 2);
    std::memcpy(&dmh, blk + 82, 2);
    const float d = f16_to_f32(dh);
    const float mn = f16_to_f32(dmh);
    int is = 0;
    for (int n2 = 0; n2 < QK_K; n2 += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t scv = scales[is++];
            float dl = d * static_cast<float>(scv & 0xF);
            float ml = mn * static_cast<float>(scv >> 4);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
            scv = scales[is++];
            dl = d * static_cast<float>(scv & 0xF);
            ml = mn * static_cast<float>(scv >> 4);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

inline void blk_q3_K(const uint8_t* blk, float* y) {
    const uint8_t* hm = blk;
    const uint8_t* q = blk + 32;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    std::memcpy(aux, blk + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    uint16_t dh;
    std::memcpy(&dh, blk + 108, 2);
    const float d_all = f16_to_f32(dh);
    int is = 0;
    uint8_t m = 1;
    for (int n2 = 0; n2 < QK_K; n2 += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<float>(
                    static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * static_cast<float>(
                    static_cast<int>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

inline void blk_iq2_s(const uint8_t* blk, float* y) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = qs + 64;
    const uint8_t* scales = qh + 8;
    const uint8_t* signs = qs + QK_K / 8;
#if defined(__AVX2__)
    const __m128i maskbits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(kMaskIq2xs));
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l < 2) ? db0 : db1;
            const uint32_t idx = static_cast<uint32_t>(qs[l]) |
                                 ((static_cast<uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u);
            const __m128i gv = _mm_cvtsi64_si128(static_cast<long long>(kIq2sGrid[idx]));
            const __m128i sb = _mm_set1_epi8(static_cast<char>(signs[l]));
            const __m128i m = _mm_cmpeq_epi8(_mm_and_si128(sb, maskbits), maskbits);
            const __m128i sv = _mm_sub_epi8(_mm_xor_si128(gv, m), m);
            _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_set1_ps(dl),
                                             _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sv))));
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
#else
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const float db0 = d * (0.5f + static_cast<float>(scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l < 2) ? db0 : db1;
            const uint32_t idx = static_cast<uint32_t>(qs[l]) |
                                 ((static_cast<uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u);
            const uint8_t* grid = reinterpret_cast<const uint8_t*>(&kIq2sGrid[idx]);
            for (int j = 0; j < 8; ++j)
                y[j] = dl * static_cast<float>(grid[j]) *
                       ((signs[l] & kMaskIq2xs[j]) ? -1.0f : 1.0f);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
#endif
}

inline void blk_iq3_s(const uint8_t* blk, float* y) {
    uint16_t dh;
    std::memcpy(&dh, blk, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = qs + 64;
    const uint8_t* signs = qh + 8;
    const uint8_t* scales = signs + 32;
#if defined(__AVX2__)
    const __m128i maskbits = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(kMaskIq2xs));
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const float db1 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] & 0xf));
        const float db2 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] >> 4));
        for (int pass = 0; pass < 2; ++pass) {
            const float db = (pass == 0) ? db1 : db2;
            for (int l = 0; l < 4; ++l) {
                const uint32_t i1 = static_cast<uint32_t>(qs[2 * l + 0]) |
                                    ((static_cast<uint32_t>(qh[pass]) << (8 - 2 * l)) & 256u);
                const uint32_t i2 = static_cast<uint32_t>(qs[2 * l + 1]) |
                                    ((static_cast<uint32_t>(qh[pass]) << (7 - 2 * l)) & 256u);
                const uint64_t g = static_cast<uint64_t>(kIq3sGrid[i1]) |
                                   (static_cast<uint64_t>(kIq3sGrid[i2]) << 32);
                const __m128i gv = _mm_cvtsi64_si128(static_cast<long long>(g));
                const __m128i sb = _mm_set1_epi8(static_cast<char>(signs[l]));
                const __m128i m = _mm_cmpeq_epi8(_mm_and_si128(sb, maskbits), maskbits);
                const __m128i sv = _mm_sub_epi8(_mm_xor_si128(gv, m), m);
                _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_set1_ps(db),
                                                 _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sv))));
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
#else
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const float db1 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] & 0xf));
        const float db2 = d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] >> 4));
        for (int pass = 0; pass < 2; ++pass) {
            const float db = (pass == 0) ? db1 : db2;
            for (int l = 0; l < 4; ++l) {
                const uint32_t i1 = static_cast<uint32_t>(qs[2 * l + 0]) |
                                    ((static_cast<uint32_t>(qh[pass]) << (8 - 2 * l)) & 256u);
                const uint32_t i2 = static_cast<uint32_t>(qs[2 * l + 1]) |
                                    ((static_cast<uint32_t>(qh[pass]) << (7 - 2 * l)) & 256u);
                const uint8_t* g1 = reinterpret_cast<const uint8_t*>(&kIq3sGrid[i1]);
                const uint8_t* g2 = reinterpret_cast<const uint8_t*>(&kIq3sGrid[i2]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * static_cast<float>(g1[j]) *
                               ((signs[l] & kMaskIq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db * static_cast<float>(g2[j]) *
                               ((signs[l] & kMaskIq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
#endif
}

inline void blk_iq4_xs(const uint8_t* blk, float* y) {
    uint16_t dh, sh;
    std::memcpy(&dh, blk, 2);
    std::memcpy(&sh, blk + 2, 2);
    const float d = f16_to_f32(dh);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = (static_cast<int>((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)) |
                       ((static_cast<int>((sh >> (2 * ib)) & 3)) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = dl * static_cast<float>(kValuesIq4nl[qs[j] & 0xf]);
            y[j + 16] = dl * static_cast<float>(kValuesIq4nl[qs[j] >> 4]);
        }
        y += 32;
        qs += 16;
    }
}

using BlockFn = void (*)(const uint8_t*, float*);

inline bool block_fn(DType t, size_t* blk_bytes, uint64_t* blk_values, BlockFn* fn) {
    switch (t) {
        case DType::Q8_0: *blk_bytes = 34; *blk_values = 32; *fn = &blk_q8_0; return true;
        case DType::IQ3_XXS: *blk_bytes = 98; *blk_values = QK_K; *fn = &blk_iq3_xxs; return true;
        case DType::Q6_K: *blk_bytes = 210; *blk_values = QK_K; *fn = &blk_q6_K; return true;
        case DType::Q2_K: *blk_bytes = 84; *blk_values = QK_K; *fn = &blk_q2_K; return true;
        case DType::Q3_K: *blk_bytes = 110; *blk_values = QK_K; *fn = &blk_q3_K; return true;
        case DType::IQ2_S: *blk_bytes = 82; *blk_values = QK_K; *fn = &blk_iq2_s; return true;
        case DType::IQ3_S: *blk_bytes = 110; *blk_values = QK_K; *fn = &blk_iq3_s; return true;
        case DType::IQ4_XS: *blk_bytes = 136; *blk_values = QK_K; *fn = &blk_iq4_xs; return true;
        default: return false;
    }
}

void dequantize_impl(DType t, const void* src, float* dst, uint64_t n, bool parallel) {
    const int bs = dtype_block_size(t);
    if (bs != 1 && (n % static_cast<uint64_t>(bs)) != 0)
        throw std::runtime_error("dequantize: element count not a multiple of block size");

    switch (t) {
        case DType::F32: {
            std::memcpy(dst, src, static_cast<size_t>(n) * 4);
            return;
        }
        case DType::F16: {
            const uint16_t* s = static_cast<const uint16_t*>(src);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] = f16_to_f32(s[i]);
            return;
        }
        case DType::BF16: {
            const uint16_t* s = static_cast<const uint16_t*>(src);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] = bf16_to_f32(s[i]);
            return;
        }
        default: break;
    }

    size_t blk_bytes = 0;
    uint64_t blk_values = 0;
    BlockFn fn = nullptr;
    if (!block_fn(t, &blk_bytes, &blk_values, &fn))
        throw std::runtime_error(std::string("dequantize: unimplemented type ") + dtype_name(t));

    const int64_t nb = static_cast<int64_t>(n / blk_values);
    const uint8_t* p = static_cast<const uint8_t*>(src);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (int64_t b = 0; b < nb; ++b)
        fn(p + static_cast<size_t>(b) * blk_bytes, dst + static_cast<uint64_t>(b) * blk_values);
}

}  // namespace

void dequantize(DType t, const void* src, float* dst, uint64_t n) {
    dequantize_impl(t, src, dst, n, true);
}

void dequantize_row(DType t, const void* src, float* dst, uint64_t n) {
    dequantize_impl(t, src, dst, n, false);
}

}  // namespace glm5
