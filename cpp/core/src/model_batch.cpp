// Chunked (batched) prefill path.
//
// The token-wise path reads every expert weight once per token (~8.8 MB x 8
// experts x 42 layers per token). Chunking turns that into one read per
// *chunk*: all tokens in the chunk are routed first, then each touched expert
// is dequantized once and applied to every token assigned to it.
//
// Per-token sub-steps that are inherently sequential (mHC mixing, KDA
// recurrence, MLA sparse attention, indexer) still run token-by-token, reusing
// the verified token-wise kernels; only the projection GEMMs and the MoE are
// batched.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "glm5/model.h"
#include "glm5/ops.h"
#include "glm5/simd.h"

namespace glm5 {

void Model::kda_project(int layer, const float* X, int n) {
    const int H = cfg_.hidden;
    const int NH = cfg_.kda_heads;
    const int D = cfg_.kda_head_dim;
    const int QKV = NH * D;
    store_.matmul_quant(blk(layer, "attn_q.weight"), 0, H, QKV, X, n, bp_tq_.data());
    store_.matmul_quant(blk(layer, "attn_k.weight"), 0, H, QKV, X, n, bp_tk_.data());
    store_.matmul_quant(blk(layer, "attn_v.weight"), 0, H, QKV, X, n, bp_tv_.data());
    for (int t = 0; t < n; ++t) {
        float* qkv = bp_qkv_.data() + static_cast<size_t>(t) * 3 * QKV;
        std::copy(bp_tq_.data() + static_cast<size_t>(t) * QKV,
                  bp_tq_.data() + static_cast<size_t>(t + 1) * QKV, qkv);
        std::copy(bp_tk_.data() + static_cast<size_t>(t) * QKV,
                  bp_tk_.data() + static_cast<size_t>(t + 1) * QKV, qkv + QKV);
        std::copy(bp_tv_.data() + static_cast<size_t>(t) * QKV,
                  bp_tv_.data() + static_cast<size_t>(t + 1) * QKV, qkv + 2 * QKV);
    }
    std::vector<float> t128(static_cast<size_t>(n) * 128);
    store_.matmul_quant(blk(layer, "ssm_f_a.weight"), 0, H, 128, X, n, t128.data());
    store_.matmul_quant(blk(layer, "ssm_f_b.weight"), 0, 128, QKV, t128.data(), n, bp_fg_.data());
    store_.matmul_quant(blk(layer, "ssm_g_a.weight"), 0, H, 128, X, n, t128.data());
    store_.matmul_quant(blk(layer, "ssm_g_b.weight"), 0, 128, QKV, t128.data(), n,
                        bp_ggate_.data());
    store_.matmul_quant(blk(layer, "ssm_beta.weight"), 0, H, NH, X, n, bp_beta_.data());
}

void Model::mla_project(int layer, const float* X, int n) {
    const int H = cfg_.hidden;
    const int NH = cfg_.n_heads;
    const int DK = cfg_.qk_nope;
    const int LR = cfg_.q_lora;
    const int KVL = cfg_.kv_lora;
    const int D = cfg_.index_head_dim;
    const int IH = cfg_.index_n_heads;

    std::vector<float> qa(static_cast<size_t>(n) * LR), qr(static_cast<size_t>(n) * LR),
        kv(static_cast<size_t>(n) * KVL);
    store_.matmul_quant(blk(layer, "attn_q_a.weight"), 0, H, LR, X, n, qa.data());
    const float* qan = store_.get(blk(layer, "attn_q_a_norm.weight"));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; ++t)
        rmsnorm(qr.data() + static_cast<size_t>(t) * LR, qa.data() + static_cast<size_t>(t) * LR,
                qan, LR, cfg_.rms_eps);
    store_.matmul_quant(blk(layer, "attn_q_b.weight"), 0, LR, NH * DK, qr.data(), n, bp_q_.data());

    store_.matmul_quant(blk(layer, "attn_kv_a_mqa.weight"), 0, H, KVL, X, n, kv.data());
    const float* kvan = store_.get(blk(layer, "attn_kv_a_norm.weight"));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; ++t)
        rmsnorm(bp_kpass_.data() + static_cast<size_t>(t) * KVL,
                kv.data() + static_cast<size_t>(t) * KVL, kvan, KVL, cfg_.rms_eps);

    // Indexer: key, compressor gate, query, and per-head weight.
    const float* knw = store_.get(blk(layer, "indexer.k_norm.weight"));
    const float* knb = store_.get(blk(layer, "indexer.k_norm.bias"));
    store_.matmul_quant(blk(layer, "indexer.attn_k.weight"), 0, H, D, X, n, bp_idxk_.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n; ++t)
        layernorm(bp_idxk_.data() + static_cast<size_t>(t) * D,
                  bp_idxk_.data() + static_cast<size_t>(t) * D, knw, knb, D, cfg_.norm_eps);
    store_.matmul_quant(blk(layer, "indexer_compressor_gate.weight"), 0, H, D, X, n,
                        bp_idxg_.data());
    store_.matmul_quant(blk(layer, "indexer.attn_q_b.weight"), 0, LR, IH * D, qr.data(), n,
                        bp_idxq_.data());
    store_.matmul_quant(blk(layer, "indexer.proj.weight"), 0, H, IH, X, n, bp_idxw_.data());
    const float wscale = 1.0f / std::sqrt(static_cast<float>(IH));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n * IH; ++t) bp_idxw_[static_cast<size_t>(t)] *= wscale;
}

void Model::ensure_batch(int n) {
    if (n <= batch_cap_) return;
    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;
    const size_t cap = static_cast<size_t>(n);
    b_streams_.assign(cap * N * H, 0.0f);
    b_post_.assign(cap * N, 0.0f);
    b_post2_.assign(cap * N, 0.0f);
    b_comb_.assign(cap * N * N, 0.0f);
    b_comb2_.assign(cap * N * N, 0.0f);
    b_mixed_.assign(cap * H, 0.0f);
    b_mlp_.assign(cap * H, 0.0f);
    b_router_.assign(cap * cfg_.n_experts, 0.0f);
    b_xg_.assign(cap * H, 0.0f);
    b_gg_.assign(cap * cfg_.moe_inter, 0.0f);
    b_ug_.assign(cap * cfg_.moe_inter, 0.0f);
    b_hg_.assign(cap * cfg_.moe_inter, 0.0f);
    b_dg_.assign(cap * H, 0.0f);
    b_off_.assign(cap + 1, 0);
    b_order_.assign(cap * cfg_.top_k, 0);
    b_eptr_.assign(static_cast<size_t>(cfg_.n_experts) + 1, 0);
    b_tok_.assign(cap * cfg_.top_k, 0);
    b_wt_.assign(cap * cfg_.top_k, 0.0f);
    bp_qkv_.assign(cap * 3 * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_tq_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_tk_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_tv_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_fg_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_beta_.assign(cap * cfg_.kda_heads, 0.0f);
    bp_ggate_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_gated_.assign(cap * cfg_.kda_heads * cfg_.kda_head_dim, 0.0f);
    bp_q_.assign(cap * cfg_.n_heads * cfg_.qk_nope, 0.0f);
    bp_kpass_.assign(cap * cfg_.kv_lora, 0.0f);
    bp_idxk_.assign(cap * cfg_.index_head_dim, 0.0f);
    bp_idxg_.assign(cap * cfg_.index_head_dim, 0.0f);
    bp_idxq_.assign(cap * cfg_.index_n_heads * cfg_.index_head_dim, 0.0f);
    bp_idxw_.assign(cap * cfg_.index_n_heads, 0.0f);
    bp_aout_.assign(cap * cfg_.n_heads * cfg_.v_head, 0.0f);
    b_attn_.assign(cap * H, 0.0f);
    batch_cap_ = n;
}

void Model::dense_mlp_batch(const char* gate, const char* up, const char* down, const float* X,
                            int n, float* Y) {
    const int H = cfg_.hidden;
    const int I = cfg_.intermediate;
    std::vector<float> g(static_cast<size_t>(n) * I), u(static_cast<size_t>(n) * I),
        h(static_cast<size_t>(n) * I);
    store_.matmul_quant(gate, 0, H, I, X, n, g.data());
    store_.matmul_quant(up, 0, H, I, X, n, u.data());
    const float lim = cfg_.swiglu_limit;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n * I; ++i) {
        const float gi = std::min(g[i], lim);
        const float ui = std::clamp(u[i], -lim, lim);
        h[i] = (gi / (1.0f + std::exp(-gi))) * ui;
    }
    store_.matmul_quant(down, 0, I, H, h.data(), n, Y);
}

void Model::moe_batch(int layer, const float* X, int n, float* Y) {
    const int H = cfg_.hidden;
    const int I = cfg_.moe_inter;
    const int E = cfg_.n_experts;
    const int K = cfg_.top_k;

    // 1. Router for all tokens at once.
    store_.matmul_quant(blk(layer, "ffn_gate_inp.weight"), 0, H, E, X, n, b_router_.data());
    const float* corr = store_.get(blk(layer, "exp_probs_b.bias"));

    // 2. Per-token top-k, then bucket the assignments by expert.
    struct Assign {
        int tok;
        int e;
        float w;
    };
    std::vector<Assign> asg;
    asg.reserve(static_cast<size_t>(n) * K);
    std::vector<int> cnt(static_cast<size_t>(E), 0);
    std::vector<int> idx(static_cast<size_t>(E));
    std::vector<float> choice(static_cast<size_t>(E)), score(static_cast<size_t>(E));
    for (int t = 0; t < n; ++t) {
        const float* lg = b_router_.data() + static_cast<size_t>(t) * E;
        for (int e = 0; e < E; ++e) {
            score[e] = 1.0f / (1.0f + std::exp(-lg[e]));
            choice[e] = score[e] + corr[e];
            idx[e] = e;
        }
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return choice[a] > choice[b]; });
        float denom = 1e-20f;
        for (int k = 0; k < K; ++k) denom += score[idx[k]];
        const float inv = cfg_.norm_topk ? cfg_.routed_scale / denom : cfg_.routed_scale;
        for (int k = 0; k < K; ++k) {
            const int e = idx[k];
            asg.push_back({t, e, score[e] * inv});
            ++cnt[static_cast<size_t>(e)];
        }
    }
    size_t acc = 0;
    for (int e = 0; e < E; ++e) {
        b_eptr_[static_cast<size_t>(e)] = static_cast<int>(acc);
        acc += cnt[static_cast<size_t>(e)];
    }
    b_eptr_[static_cast<size_t>(E)] = static_cast<int>(acc);

    // Stable bucket sort by expert id.
    std::vector<int> cursor(b_eptr_.begin(), b_eptr_.end() - 1);
    std::vector<int> pos(asg.size());
    for (size_t a = 0; a < asg.size(); ++a)
        pos[static_cast<size_t>(cursor[static_cast<size_t>(asg[a].e)]++)] = static_cast<int>(a);
    for (size_t i = 0; i < pos.size(); ++i) {
        b_tok_[i] = asg[static_cast<size_t>(pos[i])].tok;
        b_wt_[i] = asg[static_cast<size_t>(pos[i])].w;
    }

    std::fill(Y, Y + static_cast<size_t>(n) * H, 0.0f);

    // 3. One grouped pass per touched expert.
    for (int e = 0; e < E; ++e) {
        const int lo = b_eptr_[static_cast<size_t>(e)];
        const int hi = b_eptr_[static_cast<size_t>(e) + 1];
        const int g = hi - lo;
        if (g == 0) continue;

        float* Xg = b_xg_.data();
        for (int i = 0; i < g; ++i) {
            const int t = b_tok_[static_cast<size_t>(lo + i)];
            std::memcpy(Xg + static_cast<size_t>(i) * H, X + static_cast<size_t>(t) * H,
                        sizeof(float) * H);
        }

        float* Gg = b_gg_.data();
        float* Ug = b_ug_.data();
        float* Hg = b_hg_.data();
        float* Dg = b_dg_.data();
        store_.matmul_quant(blk(layer, "ffn_gate_exps.weight"), static_cast<uint64_t>(e) * I, H, I,
                            Xg, g, Gg);
        store_.matmul_quant(blk(layer, "ffn_up_exps.weight"), static_cast<uint64_t>(e) * I, H, I, Xg,
                            g, Ug);
        const float lim = cfg_.swiglu_limit;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < g * I; ++i) {
            const float gi = std::min(Gg[i], lim);
            const float ui = std::clamp(Ug[i], -lim, lim);
            Hg[i] = (gi / (1.0f + std::exp(-gi))) * ui;
        }
        store_.matmul_quant(blk(layer, "ffn_down_exps.weight"), static_cast<uint64_t>(e) * H, I, H,
                            Hg, g, Dg);

        for (int i = 0; i < g; ++i) {
            const int t = b_tok_[static_cast<size_t>(lo + i)];
            axpy_f32(Y + static_cast<size_t>(t) * H, Dg + static_cast<size_t>(i) * H,
                     b_wt_[static_cast<size_t>(lo + i)], H);
        }
    }

    // 4. Shared expert, applied to every token in the chunk.
    std::vector<float> sg(static_cast<size_t>(n) * I), su(static_cast<size_t>(n) * I),
        sh(static_cast<size_t>(n) * I);
    store_.matmul_quant(blk(layer, "ffn_gate_shexp.weight"), 0, H, I, X, n, sg.data());
    store_.matmul_quant(blk(layer, "ffn_up_shexp.weight"), 0, H, I, X, n, su.data());
    const float lim = cfg_.swiglu_limit;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n * I; ++i) {
        const float gi = std::min(sg[i], lim);
        const float ui = std::clamp(su[i], -lim, lim);
        sh[i] = (gi / (1.0f + std::exp(-gi))) * ui;
    }
    std::vector<float> sd(static_cast<size_t>(n) * H);
    store_.matmul_quant(blk(layer, "ffn_down_shexp.weight"), 0, I, H, sh.data(), n, sd.data());
    add_f32(Y, sd.data(), n * H);
}

const std::vector<float>& Model::run_chunk(const int* toks, int n, int pos_start) {
    if (n <= 0) throw std::runtime_error("run_chunk: empty chunk");
    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;
    ensure_batch(n);

    // Embeddings.
    for (int t = 0; t < n; ++t) {
        std::vector<float> emb(H);
        store_.read_slice("token_embd.weight", static_cast<uint64_t>(toks[t]) * H, H, emb.data());
        float* S = b_streams_.data() + static_cast<size_t>(t) * N * H;
        for (int j = 0; j < N; ++j)
            std::copy(emb.begin(), emb.end(), S + static_cast<size_t>(j) * H);
    }

    const int KDA_QKV = cfg_.kda_heads * cfg_.kda_head_dim;
    const int MLA_DV = cfg_.n_heads * cfg_.v_head;

    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        const std::string ap = "blk." + std::to_string(layer) + ".hc_attn_";
        const std::string fp = "blk." + std::to_string(layer) + ".hc_ffn_";
        const float* an = store_.get(blk(layer, "attn_norm.weight"));
        const float* fn_ = store_.get(blk(layer, "ffn_norm.weight"));
        std::vector<float> mixed_next(static_cast<size_t>(N) * H);

        // Pass 1: mHC + attention norm for every token in the chunk.
        for (int t = 0; t < n; ++t) {
            float* S = b_streams_.data() + static_cast<size_t>(t) * N * H;
            float* post = b_post_.data() + static_cast<size_t>(t) * N;
            float* comb = b_comb_.data() + static_cast<size_t>(t) * N * N;
            float* mix = b_mixed_.data() + static_cast<size_t>(t) * H;
            std::copy(S, S + static_cast<size_t>(N) * H, streams_.begin());
            hc_forward(ap, post, comb);
            rmsnorm(mix, mixed_.data(), an, H, cfg_.rms_eps);
        }

        // Pass 2: batched projections, then the per-token attention core.
        if (is_kda_[layer]) {
            kda_project(layer, b_mixed_.data(), n);
            for (int t = 0; t < n; ++t) {
                KdaProjView pv;
                pv.qkv = bp_qkv_.data() + static_cast<size_t>(t) * 3 * KDA_QKV;
                pv.fg = bp_fg_.data() + static_cast<size_t>(t) * KDA_QKV;
                pv.beta = bp_beta_.data() + static_cast<size_t>(t) * cfg_.kda_heads;
                pv.ggate = bp_ggate_.data() + static_cast<size_t>(t) * KDA_QKV;
                std::copy(b_mixed_.data() + static_cast<size_t>(t) * H,
                          b_mixed_.data() + static_cast<size_t>(t + 1) * H, mixed_.begin());
                kda_forward(layer, pos_start + t, &pv,
                            bp_gated_.data() + static_cast<size_t>(t) * KDA_QKV);
            }
            store_.matmul_quant(blk(layer, "attn_output.weight"), 0, KDA_QKV, H, bp_gated_.data(),
                                n, b_attn_.data());
        } else {
            mla_project(layer, b_mixed_.data(), n);
            for (int t = 0; t < n; ++t) {
                MlaProjView pv;
                pv.q = bp_q_.data() + static_cast<size_t>(t) * cfg_.n_heads * cfg_.qk_nope;
                pv.k_pass = bp_kpass_.data() + static_cast<size_t>(t) * cfg_.kv_lora;
                pv.idx_k = bp_idxk_.data() + static_cast<size_t>(t) * cfg_.index_head_dim;
                pv.idx_gate = bp_idxg_.data() + static_cast<size_t>(t) * cfg_.index_head_dim;
                pv.idx_q = bp_idxq_.data() + static_cast<size_t>(t) * cfg_.index_n_heads *
                                                  cfg_.index_head_dim;
                pv.idx_w = bp_idxw_.data() + static_cast<size_t>(t) * cfg_.index_n_heads;
                std::copy(b_mixed_.data() + static_cast<size_t>(t) * H,
                          b_mixed_.data() + static_cast<size_t>(t + 1) * H, mixed_.begin());
                mla_forward(layer, pos_start + t, &pv,
                            bp_aout_.data() + static_cast<size_t>(t) * MLA_DV);
            }
            store_.matmul_quant(blk(layer, "attn_output.weight"), 0, MLA_DV, H, bp_aout_.data(), n,
                                b_attn_.data());
        }

        // Pass 3: attention output mixing for every token.
        for (int t = 0; t < n; ++t) {
            const float* S = b_streams_.data() + static_cast<size_t>(t) * N * H;
            const float* post = b_post_.data() + static_cast<size_t>(t) * N;
            const float* comb = b_comb_.data() + static_cast<size_t>(t) * N * N;
            const float* attn = b_attn_.data() + static_cast<size_t>(t) * H;
            for (int nn = 0; nn < N; ++nn) {
                for (int d = 0; d < H; ++d) {
                    float acc = post[nn] * attn[d];
                    for (int m = 0; m < N; ++m)
                        acc += comb[static_cast<size_t>(m) * N + nn] * S[static_cast<size_t>(m) * H + d];
                    mixed_next[static_cast<size_t>(nn) * H + d] = acc;
                }
            }
            std::copy(mixed_next.begin(), mixed_next.end(),
                      b_streams_.data() + static_cast<size_t>(t) * N * H);
        }

        // Pass 4: mHC + ffn norm for every token.
        for (int t = 0; t < n; ++t) {
            float* S = b_streams_.data() + static_cast<size_t>(t) * N * H;
            float* post2 = b_post2_.data() + static_cast<size_t>(t) * N;
            float* comb2 = b_comb2_.data() + static_cast<size_t>(t) * N * N;
            float* mix = b_mixed_.data() + static_cast<size_t>(t) * H;
            std::copy(S, S + static_cast<size_t>(N) * H, streams_.begin());
            hc_forward(fp, post2, comb2);
            rmsnorm(mix, mixed_.data(), fn_, H, cfg_.rms_eps);
        }

        // Pass 5: batched MLP over the whole chunk.
        if (cfg_.layer_mlp[layer] == MlpKind::Dense) {
            dense_mlp_batch(blk(layer, "ffn_gate.weight").c_str(),
                            blk(layer, "ffn_up.weight").c_str(),
                            blk(layer, "ffn_down.weight").c_str(), b_mixed_.data(), n,
                            b_mlp_.data());
        } else {
            moe_batch(layer, b_mixed_.data(), n, b_mlp_.data());
        }

        // Pass 6: ffn mixing per token (write to `mixed_next` first: it aliases S).
        for (int t = 0; t < n; ++t) {
            const float* S = b_streams_.data() + static_cast<size_t>(t) * N * H;
            const float* post2 = b_post2_.data() + static_cast<size_t>(t) * N;
            const float* comb2 = b_comb2_.data() + static_cast<size_t>(t) * N * N;
            const float* mlp = b_mlp_.data() + static_cast<size_t>(t) * H;
            float* out = b_streams_.data() + static_cast<size_t>(t) * N * H;
            for (int nn = 0; nn < N; ++nn) {
                for (int d = 0; d < H; ++d) {
                    float acc = post2[nn] * mlp[d];
                    for (int m = 0; m < N; ++m)
                        acc += comb2[static_cast<size_t>(m) * N + nn] * S[static_cast<size_t>(m) * H + d];
                    mixed_next[static_cast<size_t>(nn) * H + d] = acc;
                }
            }
            std::copy(mixed_next.begin(), mixed_next.end(), out);
        }
    }

    // Logits from the last token.
    const float* S = b_streams_.data() + static_cast<size_t>(n - 1) * N * H;
    std::vector<float> merged(H, 0.0f);
    for (int nn = 0; nn < N; ++nn)
        for (int d = 0; d < H; ++d) merged[d] += S[static_cast<size_t>(nn) * H + d];
    const float inv_n = 1.0f / N;
    for (int d = 0; d < H; ++d) merged[d] *= inv_n;

    std::vector<float> normed(H);
    rmsnorm(normed.data(), merged.data(), store_.get("output_norm.weight"), H, cfg_.rms_eps);
    store_.matvec_quant("output.weight", 0, H, cfg_.vocab, normed.data(), logits_.data());
    return logits_;
}

}  // namespace glm5
