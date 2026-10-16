// Dequantize a single tensor from a (sharded) GGUF and print statistics.
// Used to validate the dequant codecs against real model weights, and to
// export the float values for cross-implementation comparison.
//
// usage: gguf_dequant <any-shard.gguf> <tensor-name> [out.f32] [n_limit]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "glm5/gguf.h"
#include "glm5/quant.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <any-shard.gguf> <tensor-name> [out.f32] [n_limit]\n",
                     argv[0]);
        return 2;
    }
    const std::string shard = argv[1];
    const std::string name = argv[2];
    const char* out_path = (argc > 3) ? argv[3] : nullptr;
    uint64_t n_limit = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 0;

    try {
        glm5::GgufModel model;
        model.load(glm5::GgufModel::discover_shards(shard));

        const glm5::TensorInfo* t = model.find_tensor(name);
        if (!t) {
            std::fprintf(stderr, "tensor not found: %s\n", name.c_str());
            return 1;
        }

        const uint64_t n = t->n_elements();
        std::printf("tensor      : %s\n", t->name.c_str());
        std::printf("type        : %s\n", glm5::dtype_name(t->type));
        std::printf("dims        : [");
        for (size_t i = 0; i < t->dims.size(); ++i)
            std::printf("%s%llu", i ? "," : "", (unsigned long long)t->dims[i]);
        std::printf("]\n");
        std::printf("elements    : %llu\n", (unsigned long long)n);
        std::printf("block bytes : %zu\n", glm5::dtype_type_size(t->type));

        const size_t raw_bytes = glm5::dtype_row_bytes(t->type, n);
        if (raw_bytes == 0) {
            std::fprintf(stderr, "unsupported type for size computation\n");
            return 1;
        }
        std::vector<uint8_t> raw(raw_bytes);
        model.read_tensor(*t, raw.data());

        std::vector<float> f(n);
        glm5::dequantize(t->type, raw.data(), f.data(), n);

        double sum = 0.0, absum = 0.0, sq = 0.0;
        float mn = f[0], mx = f[0];
        uint64_t nonfinite = 0;
        for (uint64_t i = 0; i < n; ++i) {
            const float v = f[i];
            if (!std::isfinite(v)) ++nonfinite;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            absum += std::fabs((double)v);
            sq += (double)v * (double)v;
        }
        const double mean = sum / (double)n;
        const double var = sq / (double)n - mean * mean;

        std::printf("min         : %.6g\n", mn);
        std::printf("max         : %.6g\n", mx);
        std::printf("mean        : %.6g\n", mean);
        std::printf("std         : %.6g\n", var > 0 ? std::sqrt(var) : 0.0);
        std::printf("absmean     : %.6g\n", absum / (double)n);
        std::printf("nonfinite   : %llu\n", (unsigned long long)nonfinite);

        // First 8 values for a quick eyeball / hash check.
        std::printf("first8      :");
        for (uint64_t i = 0; i < n && i < 8; ++i) std::printf(" %.7g", f[i]);
        std::printf("\n");

        if (out_path) {
            const uint64_t m = (n_limit && n_limit < n) ? n_limit : n;
            FILE* fp = std::fopen(out_path, "wb");
            if (!fp) {
                std::fprintf(stderr, "cannot open %s for writing\n", out_path);
                return 1;
            }
            std::fwrite(f.data(), sizeof(float), (size_t)m, fp);
            std::fclose(fp);
            std::printf("wrote       : %llu floats to %s\n", (unsigned long long)m, out_path);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
