#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "glm5/gguf.h"
#include "glm5/model.h"
#include "glm5/tokenizer.h"

using namespace glm5;

namespace {

struct Args {
    std::string model;
    std::string prompt = "Hello!";
    int max_tokens = 32;
    int ctx = 2048;
    float temp = 0.0f;
    int top_k = 0;
    bool raw = false;
    bool tokenize_only = false;
    bool show_tokens = false;
    bool no_indexer = false;
    bool no_pin = false;
    int batch = 1;
    std::string kv_dtype = "bf16";
    std::string index_dtype = "bf16";
};

void usage() {
    std::printf(
        "glm5_generate --model <shard.gguf> [--prompt TEXT] [--max-tokens N] [--ctx N]\n"
        "              [--temp F] [--top-k N] [--raw] [--tokenize-only] [--show-tokens]\n"
        "              [--kv-dtype bf16|q8|q4|q2] [--index-dtype bf16|q8|q4|q2] [--no-indexer]\n"
        "              [--batch N]   chunked prefill size (default 1 = token-wise)\n"
        "              [--no-pin]    do not keep non-expert weights resident in RAM\n");
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--model") a.model = next();
        else if (k == "--prompt") a.prompt = next();
        else if (k == "--max-tokens") a.max_tokens = std::stoi(next());
        else if (k == "--ctx") a.ctx = std::stoi(next());
        else if (k == "--temp") a.temp = std::stof(next());
        else if (k == "--top-k") a.top_k = std::stoi(next());
        else if (k == "--raw") a.raw = true;
        else if (k == "--tokenize-only") a.tokenize_only = true;
        else if (k == "--show-tokens") a.show_tokens = true;
        else if (k == "--kv-dtype") a.kv_dtype = next();
        else if (k == "--index-dtype") a.index_dtype = next();
        else if (k == "--no-indexer") a.no_indexer = true;
        else if (k == "--no-pin") a.no_pin = true;
        else if (k == "--batch") a.batch = std::stoi(next());
        else if (k == "-h" || k == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(); return 1; }
    }
    if (a.model.empty()) { usage(); return 1; }

    try {
        auto shards = GgufModel::discover_shards(a.model);
        std::printf("[load] %zu shard(s)\n", shards.size());
        GgufModel model;
        model.load(shards);

        Tokenizer tok(model);
        std::printf("[tok ] vocab=%d eos=%d eot=%d eom=%d\n", tok.vocab_size(), tok.eos_id(), tok.eot_id(), tok.eom_id());

        const std::string text = a.raw ? a.prompt
                                       : ("[gMASK]<sop><|user|>\n" + a.prompt + "<|assistant|>\n");
        std::vector<int> ids = tok.encode(text);
        std::printf("[tok ] %zu tokens: ", ids.size());
        for (int id : ids) std::printf("%d ", id);
        std::printf("\n");
        if (a.show_tokens) {
            for (int id : ids) std::printf("  %d -> '%s'\n", id, tok.id_to_token(id).c_str());
        }
        if (a.tokenize_only) return 0;

        ModelOptions mo;
        mo.max_ctx = a.ctx;
        mo.batch = a.batch;
        mo.use_indexer = !a.no_indexer;
        mo.pin_nonexpert = !a.no_pin;
        if (!kv_dtype_from_string(a.kv_dtype, &mo.kv_dtype)) {
            std::fprintf(stderr, "bad --kv-dtype: %s\n", a.kv_dtype.c_str());
            return 1;
        }
        if (!kv_dtype_from_string(a.index_dtype, &mo.index_dtype)) {
            std::fprintf(stderr, "bad --index-dtype: %s\n", a.index_dtype.c_str());
            return 1;
        }

        Model m(model, mo);
        const ModelConfig& cfg = m.config();
        std::printf("[cfg ] layers=%d hidden=%d heads=%d experts=%d top_k=%d dense=%d\n",
                    cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_experts, cfg.top_k, cfg.n_dense);
        std::printf("[cfg ] indexer=%s kv_dtype=%s index_dtype=%s ctx=%d batch=%d\n",
                    mo.use_indexer ? "on" : "off", kv_dtype_name(mo.kv_dtype),
                    kv_dtype_name(mo.index_dtype), a.ctx, mo.batch);

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float>* logits = &m.prefill(ids, 0);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("[pp  ] %zu tokens in %.2f s (%.2f tok/s)\n", ids.size(),
                    std::chrono::duration<double>(t1 - t0).count(),
                    ids.size() / std::chrono::duration<double>(t1 - t0).count());

        int pos = static_cast<int>(ids.size());
        std::mt19937 rng(1234);
        std::vector<int> gen;
        for (int step = 0; step < a.max_tokens && pos < a.ctx; ++step) {
            int next_id = 0;
            if (a.temp <= 0.0f) {
                next_id = static_cast<int>(std::max_element(logits->begin(), logits->end()) - logits->begin());
            } else {
                std::vector<int> idx(cfg.vocab);
                for (int i = 0; i < cfg.vocab; ++i) idx[i] = i;
                const int kk = (a.top_k > 0) ? std::min(a.top_k, cfg.vocab) : cfg.vocab;
                std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                                  [&](int x, int y) { return (*logits)[x] > (*logits)[y]; });
                std::vector<double> p(kk);
                double sum = 0.0;
                for (int i = 0; i < kk; ++i) {
                    p[i] = std::exp(((*logits)[idx[i]] - (*logits)[idx[0]]) / a.temp);
                    sum += p[i];
                }
                std::uniform_real_distribution<double> u(0.0, sum);
                const double r = u(rng);
                double acc = 0.0;
                next_id = idx[kk - 1];
                for (int i = 0; i < kk; ++i) {
                    acc += p[i];
                    if (r <= acc) { next_id = idx[i]; break; }
                }
            }

            if (next_id == tok.eos_id() || next_id == tok.eot_id() || next_id == tok.eom_id()) {
                std::printf("\n[stop] hit special token %d at step %d\n", next_id, step);
                break;
            }
            gen.push_back(next_id);
            const std::string piece = tok.decode({next_id});
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);

            const auto s0 = std::chrono::steady_clock::now();
            logits = &m.forward(next_id, pos);
            const auto s1 = std::chrono::steady_clock::now();
            ++pos;
            if (step == 0) std::printf("\n[dec ] first step %.3f s\n",
                                       std::chrono::duration<double>(s1 - s0).count());
        }
        std::printf("\n[done] generated %zu tokens\n", gen.size());
        std::printf("--- decoded ---\n%s\n", tok.decode(gen).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
