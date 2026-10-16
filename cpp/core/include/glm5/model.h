#pragma once

#include <cstdint>
#include <vector>

#include "glm5/indexer.h"
#include "glm5/kv_cache.h"
#include "glm5/model_config.h"
#include "glm5/weights.h"

namespace glm5 {

// Runtime options for the inference engine.
struct ModelOptions {
    int max_ctx = 2048;
    int batch = 1;                         // prefill chunk size (1 = token-wise)
    KvDtype kv_dtype = KvDtype::BF16;      // MLA latent cache precision
    KvDtype index_dtype = KvDtype::BF16;   // DSA indexer cache precision
    bool use_indexer = true;               // false => plain causal attention
    bool pin_nonexpert = true;             // keep non-expert weights resident in RAM
};

// Single-token-step inference engine for GLM-5.3-Flash (CPU reference path).
//
// The MLA attention caches the *compressed latent* (kv_lora) and performs
// absorbed attention:  score[h][p] = (W_k[h]^T q_nope[h]) . c[p]
//                      out[h]     = W_v[h] (sum_p a[p] c[p]).
//
// The DSA indexer (k-pool compression + top-k selection) is implemented for
// contexts longer than `index_topk`. Below that, every pool is selected, so
// sparse attention reproduces plain causal attention exactly.
class Model {
public:
    Model(const GgufModel& model, int max_ctx);
    Model(const GgufModel& model, const ModelOptions& opts);

    const ModelConfig& config() const { return cfg_; }
    WeightStore& weights() { return store_; }

    void reset();

    // Runs a single token at absolute position `pos`; returns logits (vocab_size).
    const std::vector<float>& forward(int token_id, int pos);

    // Runs a full prompt and returns the logits of the last token.
    const std::vector<float>& prefill(const std::vector<int>& tokens, int pos_start);

    // Chunked (batched) pass over `n` tokens; returns the last token's logits.
    // Amortizes expert weight traffic across the whole chunk.
    const std::vector<float>& run_chunk(const int* tokens, int n, int pos_start);

private:
    struct KdaState {
        std::vector<float> conv;    // [3][conv_dim] rolling window (raw inputs)
        std::vector<float> rec;     // [heads][kdim][vdim]
    };
    struct MlaCache {
        KvCache latent;             // per-position compressed kv latent (kv_lora)
        KvCache idx_key;            // per-position indexer key (index_head_dim)
        KvCache idx_gate;           // per-position indexer gate (index_head_dim)
        std::vector<float> pool_keys;  // [n_pools][index_head_dim] compressed keys
        int n_pools = 0;
    };

    void init(const GgufModel& model, const ModelOptions& opts);
    IndexerParams indexer_params() const;
    void layer_forward(int layer, int pos);
    void hc_forward(const std::string& prefix, float* post, float* comb);
    void attention_forward(int layer, int pos);

    // Precomputed projections for one token. When null, the token-wise path
    // computes them itself; the chunked path fills them for the whole chunk.
    struct KdaProjView {
        const float* qkv = nullptr;    // [3*QKV] raw q/k/v
        const float* fg = nullptr;     // [QKV]  f_b(f_a(x))
        const float* beta = nullptr;   // [NH]   raw beta logits
        const float* ggate = nullptr;  // [QKV]  g_b(g_a(x))
    };
    struct MlaProjView {
        const float* q = nullptr;         // [NH*DK]
        const float* k_pass = nullptr;    // [KVL]  rmsnormed latent
        const float* idx_k = nullptr;     // [D]    layernormed indexer key
        const float* idx_gate = nullptr;  // [D]
        const float* idx_q = nullptr;     // [IH*D]
        const float* idx_w = nullptr;     // [IH]
    };

    void kda_forward(int layer, int pos, const KdaProjView* pv = nullptr,
                     float* gated_out = nullptr);
    void mla_forward(int layer, int pos, const MlaProjView* pv = nullptr, float* out = nullptr);
    void indexer_append(int layer, const float* k_val, const float* g_val);
    void indexer_select_vals(int layer, const float* idx_q, const float* idx_w,
                             std::vector<int>& sel);
    void kda_project(int layer, const float* X, int n);
    void mla_project(int layer, const float* X, int n);
    void mlp_forward(int layer);
    void moe_forward(int layer);
    void dense_mlp(const char* gate, const char* up, const char* down);

    // --- batched (chunked prefill) helpers ---
    void ensure_batch(int n);
    void moe_batch(int layer, const float* X, int n, float* Y);
    void dense_mlp_batch(const char* gate, const char* up, const char* down, const float* X, int n,
                         float* Y);

    const GgufModel* model_ = nullptr;
    WeightStore store_;
    ModelConfig cfg_;
    ModelOptions opts_;

    // Activations / scratch.
    std::vector<float> streams_;   // [hc_mult][hidden]
    std::vector<float> post_;      // [hc_mult]
    std::vector<float> pre_;       // [hc_mult]
    std::vector<float> comb_;      // [hc_mult][hc_mult]
    std::vector<float> mixed_;     // [hidden] collapsed output
    std::vector<float> buf_;       // generic scratch
    std::vector<float> qkv_;       // KDA qkv / MLA q buf
    std::vector<float> attn_out_;  // [hidden]
    std::vector<float> mlp_out_;   // [hidden]
    std::vector<float> logits_;
    std::vector<float> res_in_;    // [hc_mult][hidden] snapshot for mHC mixing
    std::vector<float> hc_flat_;   // [hc_mult*hidden]
    std::vector<float> moe_x_;     // [hidden] MoE input copy
    std::vector<float> e_gate_, e_up_, e_down_, e_act_, e_dn_;

    // MLA / indexer scratch.
    std::vector<float> q_abs_;     // [n_heads][kv_lora] absorbed query
    std::vector<float> c_comb_;    // [kv_lora] weighted latent sum
    std::vector<float> idx_k_, idx_gate_, idx_q_;  // indexer scratch
    std::vector<float> idx_w_;
    std::vector<int> idx_sel_;

    std::vector<KdaState> kda_;
    std::vector<MlaCache> mla_;
    std::vector<char> is_kda_;

    // Chunked-prefill buffers ([n] tokens).
    int batch_cap_ = 0;
    std::vector<float> b_streams_;  // [n][hc_mult][hidden]
    std::vector<float> b_post_, b_post2_;
    std::vector<float> b_comb_, b_comb2_;
    std::vector<float> b_mixed_, b_mlp_;
    std::vector<float> b_router_, b_xg_, b_gg_, b_ug_, b_hg_, b_dg_;
    std::vector<size_t> b_off_;
    std::vector<int> b_order_, b_eptr_, b_tok_;
    std::vector<float> b_wt_;
    // Batched projection buffers ([n] tokens).
    std::vector<float> bp_qkv_;    // [n][3*QKV]
    std::vector<float> bp_tq_;     // [n][QKV] scratch (q before interleave)
    std::vector<float> bp_tk_;     // [n][QKV] scratch (k before interleave)
    std::vector<float> bp_tv_;     // [n][QKV] scratch (v before interleave)
    std::vector<float> bp_fg_;     // [n][QKV]
    std::vector<float> bp_beta_;   // [n][NH]
    std::vector<float> bp_ggate_;  // [n][QKV]
    std::vector<float> bp_gated_;  // [n][QKV]
    std::vector<float> bp_q_;      // [n][NH*DK]
    std::vector<float> bp_kpass_;  // [n][KVL]
    std::vector<float> bp_idxk_;   // [n][D]
    std::vector<float> bp_idxg_;   // [n][D]
    std::vector<float> bp_idxq_;   // [n][IH*D]
    std::vector<float> bp_idxw_;   // [n][IH]
    std::vector<float> bp_aout_;   // [n][NH*DV]
    std::vector<float> b_attn_;    // [n][H]
};

}  // namespace glm5
