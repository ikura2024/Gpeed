#pragma once

#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace glm5 {

// Dot product of two contiguous F32 vectors.
inline float dot_f32(const float* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    __m256 s = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float acc = _mm_cvtss_f32(lo);
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
#endif
}

// a[i] += s * b[i]
inline void axpy_f32(float* a, const float* b, float s, int n) {
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_fmadd_ps(vs, _mm256_loadu_ps(b + i), _mm256_loadu_ps(a + i)));
    for (; i < n; ++i) a[i] += s * b[i];
#else
    for (int i = 0; i < n; ++i) a[i] += s * b[i];
#endif
}

// a[i] += b[i]
inline void add_f32(float* a, const float* b, int n) {
#if defined(__AVX2__)
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; ++i) a[i] += b[i];
#else
    for (int i = 0; i < n; ++i) a[i] += b[i];
#endif
}

// a[i] *= s
inline void scale_f32(float* a, float s, int n) {
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(a + i, _mm256_mul_ps(vs, _mm256_loadu_ps(a + i)));
    for (; i < n; ++i) a[i] *= s;
#else
    for (int i = 0; i < n; ++i) a[i] *= s;
#endif
}

}  // namespace glm5
