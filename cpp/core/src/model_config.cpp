#include "glm5/model_config.h"

#include <algorithm>

namespace glm5 {
namespace {

const GgufValue* find_any(const GgufModel& m, const char* key) { return m.find(key); }

int get_int(const GgufModel& m, const char* key, int def) {
    const GgufValue* v = find_any(m, key);
    return v ? static_cast<int>(v->as_i64()) : def;
}

float get_float(const GgufModel& m, const char* key, float def) {
    const GgufValue* v = find_any(m, key);
    return v ? static_cast<float>(v->as_f64()) : def;
}

bool get_bool(const GgufModel& m, const char* key, bool def) {
    const GgufValue* v = find_any(m, key);
    return v ? v->as_bool() : def;
}

}  // namespace

void ModelConfig::build_layer_types() {
    layer_attn.assign(n_layers, AttnKind::Kda);
    layer_mlp.assign(n_layers, MlpKind::Sparse);
    // Reference: sparse MLA at indices 3, 7, 11, ... (every 4th starting at 3).
    for (int i = 0; i < n_layers; ++i) {
        layer_attn[i] = ((i >= 3) && ((i - 3) % 4 == 0)) ? AttnKind::Mla : AttnKind::Kda;
        layer_mlp[i] = (i < n_dense) ? MlpKind::Dense : MlpKind::Sparse;
    }
}

ModelConfig ModelConfig::from_gguf(const GgufModel& m) {
    const char* p = "glm5-next.";
    auto K = [&](const char* suffix) {
        static thread_local std::string s;
        s = std::string(p) + suffix;
        return s.c_str();
    };

    ModelConfig c;
    c.n_layers = get_int(m, K("block_count"), c.n_layers) - 1;  // exclude MTP layer
    c.hidden = get_int(m, K("embedding_length"), c.hidden);
    c.intermediate = get_int(m, K("feed_forward_length"), c.intermediate);
    c.vocab = get_int(m, K("vocab_size"), c.vocab);

    c.n_heads = get_int(m, K("attention.head_count"), c.n_heads);
    c.qk_nope = get_int(m, K("attention.key_length_mla"), c.qk_nope);
    c.v_head = get_int(m, K("attention.value_length_mla"), c.v_head);
    c.qk_rope = get_int(m, K("rope.dimension_count"), c.qk_rope);
    c.q_lora = get_int(m, K("attention.q_lora_rank"), c.q_lora);
    c.kv_lora = get_int(m, K("attention.kv_lora_rank"), c.kv_lora);

    c.index_n_heads = get_int(m, K("attention.indexer.head_count"), c.index_n_heads);
    c.index_head_dim = get_int(m, K("attention.indexer.key_length"), c.index_head_dim);
    c.index_topk = get_int(m, K("attention.indexer.top_k"), c.index_topk);
    c.index_kpool = get_int(m, K("attention.indexer.kpool"), c.index_kpool);

    c.hc_mult = get_int(m, K("hyper_connection.count"), c.hc_mult);
    c.hc_sinkhorn_iters = get_int(m, K("hyper_connection.sinkhorn_iterations"), c.hc_sinkhorn_iters);
    c.hc_eps = get_float(m, K("hyper_connection.epsilon"), c.hc_eps);

    c.kda_head_dim = get_int(m, K("kda.head_dim"), c.kda_head_dim);
    c.kda_conv = get_int(m, K("ssm.conv_kernel"), c.kda_conv);
    c.kda_gate_lower_bound = get_float(m, K("kda.gate_lower_bound"), c.kda_gate_lower_bound);
    {
        // num_v_heads for KDA: qkv dim per token = kda_heads*head_dim. Detect from tensor shape.
        const TensorInfo* t = m.find_tensor("blk.0.attn_q.weight");
        if (t && t->dims.size() >= 2) {
            c.kda_heads = static_cast<int>(t->dims[1] / c.kda_head_dim);
        } else {
            c.kda_heads = 64;
        }
    }

    c.n_dense = get_int(m, K("leading_dense_block_count"), c.n_dense);
    c.n_experts = get_int(m, K("expert_count"), c.n_experts);
    c.moe_inter = get_int(m, K("expert_feed_forward_length"), c.moe_inter);
    c.top_k = get_int(m, K("expert_used_count"), c.top_k);
    c.n_shared = get_int(m, K("expert_shared_count"), c.n_shared);
    c.routed_scale = get_float(m, K("expert_weights_scale"), c.routed_scale);
    c.scoring_func = get_int(m, K("expert_gating_func"), c.scoring_func);
    c.norm_topk = get_bool(m, K("expert_weights_norm"), c.norm_topk);

    c.rms_eps = get_float(m, K("attention.layer_norm_rms_epsilon"), c.rms_eps);
    c.norm_eps = get_float(m, K("attention.layer_norm_epsilon"), c.norm_eps);

    {
        const GgufValue* v = m.find(K("swiglu_clamp_exp"));
        if (v && v->is_array() && !v->array.empty()) c.swiglu_limit = static_cast<float>(v->array[0].as_f64());
    }

    c.build_layer_types();
    return c;
}

}  // namespace glm5
