#pragma once

#include <string>

namespace glm5 {

int cuda_device_count();
std::string cuda_device_name(int index);

// Computes dst[i] = a[i] + b[i]. Smoke test for the custom-kernel build pipeline.
void cuda_add(const float* a, const float* b, float* dst, int n);

}  // namespace glm5
