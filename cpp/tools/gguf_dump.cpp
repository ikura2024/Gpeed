#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "glm5/gguf.h"
#include "glm5/quant.h"

using namespace glm5;

static std::string human_bytes(double b) {
    const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    while (b >= 1024.0 && i < 4) { b /= 1024.0; ++i; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", b, u[i]);
    return buf;
}

static void print_value(const GgufValue& v) {
    switch (v.type) {
        case GgufValueType::STRING: std::printf("%s", v.as_str().c_str()); break;
        case GgufValueType::FLOAT32:
        case GgufValueType::FLOAT64: std::printf("%g", v.as_f64()); break;
        case GgufValueType::BOOL: std::printf("%s", v.as_bool() ? "true" : "false"); break;
        case GgufValueType::ARRAY:
            std::printf("<array[%zu] type=%d>", v.array.size(), (int)v.array_type);
            if (v.array.size() <= 8) {
                std::printf(" {");
                for (size_t i = 0; i < v.array.size(); ++i) {
                    std::printf("%s", i ? "," : "");
                    print_value(v.array[i]);
                }
                std::printf("}");
            }
            break;
        default:
            if (v.type == GgufValueType::INT8 || v.type == GgufValueType::INT16 ||
                v.type == GgufValueType::INT32 || v.type == GgufValueType::INT64)
                std::printf("%lld", (long long)v.as_i64());
            else
                std::printf("%llu", (unsigned long long)v.as_u64());
            break;
    }
}

static void print_metadata(const GgufModel& m) {
    std::printf("== metadata ==\n");
    for (const std::string& k : m.keys()) {
        const GgufValue* v = m.find(k);
        if (!v) continue;
        std::printf("  %-46s = ", k.c_str());
        print_value(*v);
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gguf_dump <model-00001-of-NNNNN.gguf> [tensor_substr]\n");
        return 1;
    }
    try {
        const std::string first = argv[1];
        const std::string filter = argc > 2 ? argv[2] : "";

        auto shards = GgufModel::discover_shards(first);
        std::printf("shards: %zu\n", shards.size());
        for (const auto& s : shards) std::printf("  %s\n", s.c_str());

        GgufModel model;
        model.load(shards);

        print_metadata(model);

        std::printf("== tensors: %zu (unique names) ==\n", model.tensor_names().size());
        std::map<DType, int> hist;
        double total_bytes = 0, total_params = 0;
        for (const std::string& name : model.tensor_names()) {
            const TensorInfo* t = model.find_tensor(name);
            if (!t) continue;
            hist[t->type]++;
            total_bytes += static_cast<double>(t->row_bytes()) *
                           static_cast<double>(t->n_elements() / (t->dims.empty() ? 1 : t->dims[0]));
            total_params += static_cast<double>(t->n_elements());
        }
        std::printf("total parameters: %.3f B\n", total_params / 1e9);
        std::printf("total data size : %s\n", human_bytes(total_bytes).c_str());
        std::printf("== dtype histogram ==\n");
        for (auto& [dt, c] : hist) std::printf("  %-10s %d\n", dtype_name(dt), c);

        if (!filter.empty()) {
            std::printf("== tensors matching '%s' ==\n", filter.c_str());
            int shown = 0;
            for (const std::string& name : model.tensor_names()) {
                if (name.find(filter) == std::string::npos) continue;
                const TensorInfo* t = model.find_tensor(name);
                std::printf("  %-70s %-8s dims=[", name.c_str(), dtype_name(t->type));
                for (size_t i = 0; i < t->dims.size(); ++i)
                    std::printf("%s%llu", i ? "," : "", (unsigned long long)t->dims[i]);
                std::printf("] shard=%d off=%llu\n", t->shard, (unsigned long long)t->offset);
                if (++shown >= 64) { std::printf("  ...\n"); break; }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
