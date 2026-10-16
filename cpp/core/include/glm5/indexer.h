#pragma once

#include <vector>

namespace glm5 {

// DSA (DeepSeek sparse attention) indexer parameters.
struct IndexerParams {
    int n_heads = 32;              // index_n_heads
    int head_dim = 128;            // index_head_dim
    int kpool = 4;                 // index_kpool
    int topk = 2048;               // index_topk (budget in *tokens*)
    bool always_select_tail = true;
};

// Compresses `kpool` consecutive positions into one pool key: a per-dimension
// softmax over the gate scores (+ additive positional embedding) weights the
// pooled key rows.  All inputs are row-major [kpool][head_dim].
//   k_rows    : indexer keys
//   gate_rows : compressor gate scores
//   ape       : index_kpool_compress_ape, [kpool][head_dim]
void indexer_pool_key(const IndexerParams& p, const float* k_rows, const float* gate_rows,
                      const float* ape, float* out);

// Selects the absolute key positions a single query attends to.
//   pool_keys : [n_pools][head_dim]
//   q         : [n_heads][head_dim]  indexer query (wq_b applied)
//   q_w       : [n_heads]            pre-scaled per-head weights
//   kv_len    : number of cached positions (pools are aligned at position 0)
// Returns ascending, de-duplicated positions.
std::vector<int> indexer_select(const IndexerParams& p, const float* pool_keys, int n_pools,
                                const float* q, const float* q_w, int kv_len);

}  // namespace glm5
