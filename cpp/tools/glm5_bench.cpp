// Micro-benchmark for the quantized dequantize + dot path.
//
// Reads real rows out of a GGUF model and measures how long it takes to
// dequantize them and dot them against an F32 vector, reporting the aggregate
// element rate. Used to decide where the CPU decode time actually goes.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "glm5/common.h"
#include "glm5/gguf.h"
#include "glm5/quant.h"
#include "glm5/simd.h"

using namespace glm5;

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: glm5_bench <shard.gguf> [name-substr] [reps]\n");
        return 1;
    }
    const std::string needle = (argc >= 3) ? argv[2] : "ffn_gate_exps.weight";
    const int reps = (argc >= 4) ? std::stoi(argv[3]) : 8;

    auto shards = GgufModel::discover_shards(argv[1]);
    GgufModel model;
    model.load(shards);

    for (const std::string& name : model.tensor_names()) {
        if (name.find(needle) == std::string::npos) continue;
        const TensorInfo* t = model.find_tensor(name);
        if (!t) continue;

        const uint64_t in_dim = t->dims.empty() ? 0 : t->dims[0];
        const uint64_t nrows = in_dim ? t->n_elements() / in_dim : 0;
        const size_t row_bytes = static_cast<size_t>(t->row_bytes());
        const size_t total_bytes = row_bytes * static_cast<size_t>(nrows);

        std::vector<uint8_t> raw(total_bytes);
        model.read_tensor(*t, raw.data());

        std::vector<float> x(static_cast<size_t>(in_dim), 0.5f);
        std::vector<float> row(static_cast<size_t>(in_dim));
        std::vector<float> out(static_cast<size_t>(nrows));

        // Dump the first block so it can be compared with an independent
        // reference implementation.
        if (std::getenv("GLM5_BENCH_DUMP")) {
            const uint64_t bv = static_cast<uint64_t>(dtype_block_size(t->type));
            const size_t bb = dtype_row_bytes(t->type, bv);
            std::vector<float> blk(bv);
            for (int k = 0; k < 8; ++k) {
                dequantize_row(t->type, raw.data() + static_cast<size_t>(k) * bb, blk.data(), bv);
                double s = 0.0;
                for (uint64_t i = 0; i < bv; ++i) s += blk[i];
                std::printf("  blk[%d] sum=%.6f first=%.6f %.6f %.6f\n", k, s, blk[0], blk[1], blk[2]);
            }
        }

        // Warm the page cache / branch predictors.
        for (uint64_t j = 0; j < nrows; ++j)
            dequantize(t->type, raw.data() + row_bytes * j, row.data(), in_dim);

        double best = 1e30;
        double sum = 0.0;
        for (int r = 0; r < reps; ++r) {
            const double a = now_s();
            for (uint64_t j = 0; j < nrows; ++j) {
                dequantize(t->type, raw.data() + row_bytes * j, row.data(), in_dim);
                out[j] = dot_f32(row.data(), x.data(), static_cast<int>(in_dim));
            }
            const double b = now_s();
            for (uint64_t j = 0; j < nrows; ++j) sum += out[j];
            if (b - a < best) best = b - a;
        }

        const double elems = static_cast<double>(t->n_elements());
        std::printf("%-40s %-7s rows=%-6llu in=%-5llu bytes=%-10zu  %.1f Mrow/s  %.2f Gelem/s  %.2f GB/s\n",
                    name.c_str(), dtype_name(t->type), static_cast<unsigned long long>(nrows),
                    static_cast<unsigned long long>(in_dim), total_bytes,
                    static_cast<double>(nrows) / best / 1e6, elems / best / 1e9,
                    static_cast<double>(total_bytes) / best / 1e9);
        std::printf("    (checksum %.3f)\n", sum);
    }
    return 0;
}
