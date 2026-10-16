#include "glm5/kernels.h"

#include <cuda_runtime.h>

namespace glm5 {

__global__ void add_kernel(const float* a, const float* b, float* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

int cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

std::string cuda_device_name(int index) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) return "<none>";
    return prop.name;
}

void cuda_add(const float* a, const float* b, float* dst, int n) {
    float *da = nullptr, *db = nullptr, *dd = nullptr;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dd, n * sizeof(float));
    cudaMemcpy(da, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(da, db, dd, n);
    cudaMemcpy(dst, dd, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dd);
}

}  // namespace glm5
