#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "glm5/gguf.h"
#include "glm5/quant.h"

using namespace glm5;

static bool approx(float a, float b) {
    const float diff = std::fabs(a - b);
    const float scale = std::max(1.0f, std::fabs(b));
    return diff <= 1e-4f * scale;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: gguf_selftest <test.gguf> <test.expected.txt>\n");
        return 1;
    }

    // Load expected values: "name\nv0 v1 v2 ...\n" repeated.
    std::map<std::string, std::vector<float>> expected;
    {
        std::ifstream in(argv[2]);
        if (!in) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        std::string name;
        while (std::getline(in, name)) {
            if (name.empty()) continue;
            std::string line;
            std::getline(in, line);
            std::istringstream ss(line);
            std::vector<float> vals;
            float v;
            while (ss >> v) vals.push_back(v);
            expected[name] = std::move(vals);
        }
    }

    int failures = 0;
    try {
        auto f = GgufFile::open(argv[1]);

        // Metadata checks.
        auto check_str = [&](const char* k, const char* want) {
            const GgufValue* v = f->find(k);
            bool ok = v && v->type == GgufValueType::STRING && v->as_str() == want;
            std::printf("[%s] metadata %s = %s\n", ok ? " OK " : "FAIL", k,
                        v ? v->as_str().c_str() : "<missing>");
            if (!ok) ++failures;
        };
        check_str("general.architecture", "glm5_next");
        std::printf("[%s] alignment = %llu\n",
                    f->get("general.alignment").as_u64() == 32 ? " OK " : "FAIL",
                    (unsigned long long)f->get("general.alignment").as_u64());
        {
            const GgufValue* arr = f->find("test.array");
            bool ok = arr && arr->is_array() && arr->array.size() == 3 &&
                      arr->array[0].as_u64() == 1 && arr->array[1].as_u64() == 2 &&
                      arr->array[2].as_u64() == 3;
            std::printf("[%s] test.array = [1,2,3]\n", ok ? " OK " : "FAIL");
            if (!ok) ++failures;
        }

        // Tensor checks.
        for (const auto& [name, want] : expected) {
            const TensorInfo* t = f->find_tensor(name);
            if (!t) { std::printf("[FAIL] tensor %s missing\n", name.c_str()); ++failures; continue; }

            std::vector<uint8_t> raw(static_cast<size_t>(t->row_bytes()) *
                                     static_cast<size_t>(t->n_elements() / t->dims[0]));
            f->read_tensor(*t, raw.data());
            std::vector<float> got(t->n_elements());
            dequantize(t->type, raw.data(), got.data(), t->n_elements());

            bool ok = got.size() == want.size();
            if (ok) {
                for (size_t i = 0; i < got.size(); ++i) {
                    if (!approx(got[i], want[i])) {
                        std::printf("       %s[%zu]: got %.8g want %.8g\n", name.c_str(), i,
                                    got[i], want[i]);
                        ok = false;
                        break;
                    }
                }
            }
            std::printf("[%s] %-14s %-8s n=%llu\n", ok ? " OK " : "FAIL", name.c_str(),
                        dtype_name(t->type), (unsigned long long)t->n_elements());
            if (!ok) ++failures;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 2;
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "TESTS FAILED", failures);
    return failures == 0 ? 0 : 1;
}
