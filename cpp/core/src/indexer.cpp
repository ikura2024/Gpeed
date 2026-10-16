#include "glm5/indexer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace glm5 {

void indexer_pool_key(const IndexerParams& p, const float* k_rows, const float* gate_rows,
                      const float* ape, float* out) {
    const int KP = p.kpool;
    const int D = p.head_dim;
    for (int d = 0; d < D; ++d) {
        float mx = -INFINITY;
        for (int pp = 0; pp < KP; ++pp) mx = std::max(mx, gate_rows[pp * D + d] + ape[pp * D + d]);
        float sum = 0.0f;
        float acc = 0.0f;
        for (int pp = 0; pp < KP; ++pp) {
            const float e = std::exp(gate_rows[pp * D + d] + ape[pp * D + d] - mx);
            sum += e;
            acc += e * k_rows[pp * D + d];
        }
        out[d] = sum > 0.0f ? acc / sum : 0.0f;
    }
}

std::vector<int> indexer_select(const IndexerParams& p, const float* pool_keys, int n_pools,
                                const float* q, const float* q_w, int kv_len) {
    const int KP = std::max(1, p.kpool);
    const int D = p.head_dim;
    const int NH = p.n_heads;
    const int select_k = std::min(p.topk / KP, n_pools);
    const float iscale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> scores(static_cast<size_t>(n_pools));
    for (int g = 0; g < n_pools; ++g) {
        const float* pk = pool_keys + static_cast<size_t>(g) * D;
        float s = 0.0f;
        for (int h = 0; h < NH; ++h) {
            const float* qh = q + static_cast<size_t>(h) * D;
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += qh[d] * pk[d];
            acc *= iscale;
            if (acc < 0.0f) acc = 0.0f;
            s += q_w[h] * acc;
        }
        scores[static_cast<size_t>(g)] = s;
    }

    std::vector<int> order(static_cast<size_t>(n_pools));
    std::iota(order.begin(), order.end(), 0);
    if (select_k < n_pools) {
        std::partial_sort(order.begin(), order.begin() + select_k, order.end(),
                          [&](int a, int b) { return scores[a] > scores[b]; });
        order.resize(static_cast<size_t>(select_k));
    }

    std::vector<int> sel;
    sel.reserve(static_cast<size_t>(select_k) * KP + KP);
    for (int g : order)
        for (int pp = 0; pp < KP; ++pp) sel.push_back(g * KP + pp);

    if (p.always_select_tail) {
        const int tail_count = kv_len % KP;
        const int tail_start = kv_len - tail_count;
        for (int i = 0; i < tail_count && i < KP - 1; ++i) sel.push_back(tail_start + i);
    }

    std::sort(sel.begin(), sel.end());
    sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
    if (sel.empty() && kv_len > 0) sel.push_back(kv_len - 1);
    return sel;
}

}  // namespace glm5
