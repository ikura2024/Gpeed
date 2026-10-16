#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace glm5 {

// "blk.<layer>.<suffix>" tensor name.
inline std::string blk(int layer, const char* suffix) {
    char b[128];
    std::snprintf(b, sizeof(b), "blk.%d.%s", layer, suffix);
    return std::string(b);
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }

// y = x * rsqrt(mean(x^2) + eps) * w   (weighted RMSNorm)
inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float rstd = 1.0f / std::sqrt(ss / n + eps);
    if (w) {
        for (int i = 0; i < n; ++i) out[i] = x[i] * rstd * w[i];
    } else {
        for (int i = 0; i < n; ++i) out[i] = x[i] * rstd;
    }
}

// y = (x - mean) / sqrt(var + eps) * w + b   (LayerNorm, biased variance)
inline void layernorm(float* out, const float* x, const float* w, const float* b, int n, float eps) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float rstd = static_cast<float>(1.0 / std::sqrt(var + eps));
    for (int i = 0; i < n; ++i)
        out[i] = static_cast<float>(x[i] - mean) * rstd * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

}  // namespace glm5
