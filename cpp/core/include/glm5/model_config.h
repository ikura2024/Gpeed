#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "glm5/gguf.h"

namespace glm5 {

// Per-layer attention kind.
enum class AttnKind {
    Kda,   // linear attention (Kimi Delta Attention / gated delta net)
    Mla,   // DeepSeek sparse MLA attention (a.k.a. "indexed")
};

// Per-layer MLP kind.
enum class MlpKind {
    Dense,   // classic SwiGLU MLP (first `n_dense` layers)
    Sparse,  // MoE
};

struct ModelConfig {
    // Sizes
    int hidden = 4096;
    int n_layers = 45;
    int vocab = 154880;
    int intermediate = 12288;

    // Attention (MLA branch)
    int n_heads = 64;
    int qk_nope = 256;   // key_length_mla
    int qk_rope = 0;     // rope.dimension_count (NoPE)
    int v_head = 256;    // value_length_mla
    int q_lora = 1536;
    int kv_lora = 512;

    // Indexer
    int index_n_heads = 32;
    int index_head_dim = 128;
    int index_topk = 2048;
    int index_kpool = 4;
    bool index_kpool_always_select_tail = true;
    bool index_kpool_compress = true;

    // Hyper-connections (mHC)
    int hc_mult = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps = 1e-6f;

    // KDA (linear attention branch)
    int kda_heads = 64;
    int kda_head_dim = 128;
    int kda_conv = 4;
    float kda_gate_lower_bound = -5.0f;

    // MoE
    int n_dense = 3;
    int n_experts = 288;
    int moe_inter = 2048;
    int top_k = 8;
    int n_shared = 1;
    float routed_scale = 2.5f;
    int scoring_func = 2;      // 2 == sigmoid
    bool norm_topk = true;

    // Norms / misc
    float rms_eps = 1e-5f;
    float norm_eps = 1e-6f;    // nn.LayerNorm layers (indexer k_norm) + hc
    float swiglu_limit = 10.0f;

    // Derived
    std::vector<AttnKind> layer_attn;
    std::vector<MlpKind> layer_mlp;

    int kda_qkv_dim() const { return kda_heads * kda_head_dim; }   // 8192
    int mla_q_dim() const { return n_heads * qk_nope; }            // 16384
    int mla_out_dim() const { return n_heads * v_head; }           // 16384
    int index_q_dim() const { return index_n_heads * index_head_dim; }  // 4096
    bool kda_out_dim() const { return true; }

    static ModelConfig from_gguf(const GgufModel& m);

    // Layer-type helpers (reference defaults if metadata is missing).
    void build_layer_types();
};

}  // namespace glm5
