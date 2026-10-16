// Self-test for the KV cache quantization and the DSA indexer selection logic.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "glm5/indexer.h"
#include "glm5/kv_cache.h"

using namespace glm5;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what);
        ++g_fail;
    }
}

void test_kv_roundtrip(KvDtype dt, float max_expected) {
    const int dim = 512;
    const size_t npos = 37;
    KvCache cache(dim, npos, dt);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<std::vector<float>> rows(npos, std::vector<float>(dim));
    for (size_t p = 0; p < npos; ++p) {
        for (int d = 0; d < dim; ++d) rows[p][d] = ud(rng);
        cache.append(rows[p].data());
    }

    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::vector<float> got(dim), acc(dim), dq(dim);
    for (size_t p = 0; p < npos; ++p) {
        cache.dequant_row(p, got.data());
        for (int d = 0; d < dim; ++d) {
            const double e = std::fabs(got[d] - rows[p][d]);
            sum_sq += e * e;
            max_abs = std::max(max_abs, e);
        }
        // axpy_row must agree with dequant_row.
        std::fill(acc.begin(), acc.end(), 0.0f);
        cache.axpy_row(p, 1.0f, acc.data());
        for (int d = 0; d < dim; ++d)
            if (acc[d] != got[d]) {
                check(false, "axpy_row != dequant_row");
                break;
            }
        // dot_row against the dequantized row.
        std::vector<float> probe(dim);
        for (int d = 0; d < dim; ++d) probe[d] = std::sin(0.1f * static_cast<float>(d + p));
        float dot = 0.0f;
        cache.dot_row(p, probe.data(), &dot);
        double ref = 0.0;
        for (int d = 0; d < dim; ++d) ref += static_cast<double>(got[d]) * probe[d];
        if (std::fabs(static_cast<double>(dot) - ref) > 1e-4 * std::fabs(ref) + 1e-3)
            check(false, "dot_row mismatch");
    }
    const double rms = std::sqrt(sum_sq / (npos * dim));
    std::printf("  %-4s rms=%.5f max=%.5f (%.2f bytes/elem)\n", kv_dtype_name(dt), rms, max_abs,
                static_cast<double>(kv_block_bytes(dt)) / kKvBlock);
    check(max_abs <= max_expected, "quantization error above expectation");
}

void test_kv() {
    std::printf("[kv] round-trip\n");
    // Thresholds: bf16 ~1e-3 (8-bit mantissa), q8_0 ~amax/254, and the ggml
    // Q4_0 layout is asymmetric (-8..7 levels) so its worst case is amax/8.
    test_kv_roundtrip(KvDtype::BF16, 0.01f);
    test_kv_roundtrip(KvDtype::Q8_0, 0.02f);
    test_kv_roundtrip(KvDtype::Q4_0, 0.13f);
    test_kv_roundtrip(KvDtype::Q2_0, 0.40f);
}

void test_capacity() {
    std::printf("[kv] capacity / reset\n");
    KvCache cache(128, 4, KvDtype::Q8_0);
    std::vector<float> row(128, 1.0f);
    for (int i = 0; i < 4; ++i) cache.append(row.data());
    check(cache.size() == 4, "size after appends");
    bool threw = false;
    try {
        cache.append(row.data());
    } catch (...) {
        threw = true;
    }
    check(threw, "append past capacity must throw");
    cache.reset();
    check(cache.size() == 0, "size after reset");
}

std::vector<int> run_select(const IndexerParams& p, int kv_len, int n_pools, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> pool_keys(static_cast<size_t>(n_pools) * p.head_dim);
    for (auto& v : pool_keys) v = ud(rng);
    std::vector<float> q(static_cast<size_t>(p.n_heads) * p.head_dim);
    for (auto& v : q) v = ud(rng);
    std::vector<float> w(static_cast<size_t>(p.n_heads));
    for (auto& v : w) v = ud(rng);
    return indexer_select(p, pool_keys.data(), n_pools, q.data(), w.data(), kv_len);
}

void test_indexer() {
    std::printf("[idx] selection\n");
    IndexerParams p;  // defaults: 32 heads, 128 dim, kpool 4, topk 2048, tail on

    // Exactness boundary: while the whole context fits in the top-k budget the
    // selection must be every position (== plain causal attention).
    for (int kv_len = 1; kv_len <= 2051; kv_len += 1 + kv_len / 3) {
        const int n_pools = kv_len / p.kpool;
        auto sel = run_select(p, kv_len, n_pools, 7u);
        check(static_cast<int>(sel.size()) == kv_len, "kv_len <= 2051 selects everything");
        for (int i = 0; i < kv_len; ++i)
            if (sel[static_cast<size_t>(i)] != i) check(false, "kv_len <= 2051 selection order");
    }

    // Beyond the budget: 512 pools (= 2048 tokens) plus the tail.
    for (int kv_len : {2052, 3000, 5000, 5003, 8192, 65537}) {
        const int n_pools = kv_len / p.kpool;
        auto sel = run_select(p, kv_len, n_pools, 11u);
        const int expect = 2048 + (kv_len % p.kpool);
        check(static_cast<int>(sel.size()) == expect, "sparse selection size");
        check(std::is_sorted(sel.begin(), sel.end()), "selection sorted");
        check(std::adjacent_find(sel.begin(), sel.end()) == sel.end(), "selection unique");
        for (int s : sel) check(s >= 0 && s < kv_len, "selection in range");
        // The tail always keeps the most recent tokens visible.
        for (int i = 0; i < kv_len % p.kpool; ++i)
            check(std::find(sel.begin(), sel.end(), kv_len - 1 - i) != sel.end(), "tail present");
    }

    // Determinism.
    const int n_pools = 4000 / p.kpool;
    auto a = run_select(p, 4000, n_pools, 99u);
    auto b = run_select(p, 4000, n_pools, 99u);
    check(a == b, "selection deterministic");
}

void test_pool_key() {
    std::printf("[idx] pool key\n");
    IndexerParams p;
    const int KP = p.kpool, D = p.head_dim;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<float> k(static_cast<size_t>(KP) * D), gate(static_cast<size_t>(KP) * D, 0.0f);
    std::vector<float> ape(static_cast<size_t>(KP) * D, 0.0f);
    for (auto& v : k) v = ud(rng);

    // With zero gate and zero ape the pooled key is the arithmetic mean.
    std::vector<float> out(D);
    indexer_pool_key(p, k.data(), gate.data(), ape.data(), out.data());
    for (int d = 0; d < D; ++d) {
        float mean = 0.0f;
        for (int pp = 0; pp < KP; ++pp) mean += k[pp * D + d];
        mean /= KP;
        if (std::fabs(out[d] - mean) > 1e-5f) {
            check(false, "uniform softmax must average the pool");
            break;
        }
    }
}

}  // namespace

int main() {
    test_kv();
    test_capacity();
    test_indexer();
    test_pool_key();
    if (g_fail == 0) {
        std::printf("glm5_selftest: all checks passed\n");
        return 0;
    }
    std::printf("glm5_selftest: %d check(s) failed\n", g_fail);
    return 1;
}
