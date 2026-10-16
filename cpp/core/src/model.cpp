#include "glm5/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "glm5/ops.h"
#include "glm5/simd.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glm5 {
namespace {

// Optional phase profiler (set GLM5_PROFILE=1).
struct Profiler {
    double hc = 0, attn = 0, mlp = 0, head = 0;
    double t_last = 0;
    bool on = false;
};
Profiler& prof() {
    static Profiler p = [] {
        Profiler q;
        q.on = std::getenv("GLM5_PROFILE") != nullptr;
        return q;
    }();
    return p;
}
using Clock = std::chrono::steady_clock;
inline double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// dst[j] = sum_i W[i + in_dim * j] * x[i]   (ggml ne = (in_dim, out_dim))
void matvec(float* dst, const float* W, const float* x, int in_dim, int out_dim) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < out_dim; ++j) {
        dst[j] = dot_f32(W + static_cast<size_t>(in_dim) * j, x, in_dim);
    }
}

// y = (x - mean) / sqrt(var + eps) * w + b   (LayerNorm, biased variance)
void softmax(std::vector<float>& s, int n) {
    float mx = s[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, s[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        s[i] = std::exp(s[i] - mx);
        sum += s[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) s[i] *= inv;
}

}  // namespace

Model::Model(const GgufModel& model, int max_ctx) {
    ModelOptions opts;
    opts.max_ctx = max_ctx;
    init(model, opts);
}

Model::Model(const GgufModel& model, const ModelOptions& opts) {
    init(model, opts);
}

void Model::init(const GgufModel& model, const ModelOptions& opts) {
    model_ = &model;
    store_ = WeightStore(model);
    cfg_ = ModelConfig::from_gguf(model);
    opts_ = opts;

    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;

    streams_.assign(static_cast<size_t>(N) * H, 0.0f);
    post_.assign(N, 0.0f);
    pre_.assign(N, 0.0f);
    comb_.assign(static_cast<size_t>(N) * N, 0.0f);
    mixed_.assign(H, 0.0f);
    buf_.assign(std::max<size_t>(cfg_.kda_qkv_dim() * 3, std::max(cfg_.mla_q_dim(), cfg_.index_q_dim())), 0.0f);
    qkv_.assign(cfg_.kda_qkv_dim() * 3, 0.0f);
    attn_out_.assign(H, 0.0f);
    mlp_out_.assign(H, 0.0f);
    logits_.assign(cfg_.vocab, 0.0f);
    res_in_.assign(static_cast<size_t>(N) * H, 0.0f);
    hc_flat_.assign(static_cast<size_t>(N) * H, 0.0f);
    moe_x_.assign(H, 0.0f);

    const int inter = cfg_.moe_inter;
    e_gate_.assign(static_cast<size_t>(inter) * H, 0.0f);
    e_up_.assign(static_cast<size_t>(inter) * H, 0.0f);
    e_down_.assign(static_cast<size_t>(H) * inter, 0.0f);
    e_act_.assign(inter, 0.0f);
    e_dn_.assign(H, 0.0f);

    q_abs_.assign(static_cast<size_t>(cfg_.n_heads) * cfg_.kv_lora, 0.0f);
    c_comb_.assign(cfg_.kv_lora, 0.0f);
    idx_k_.assign(cfg_.index_head_dim, 0.0f);
    idx_gate_.assign(cfg_.index_head_dim, 0.0f);
    idx_q_.assign(static_cast<size_t>(cfg_.index_n_heads) * cfg_.index_head_dim, 0.0f);
    idx_w_.assign(cfg_.index_n_heads, 0.0f);

    const size_t max_pos = static_cast<size_t>(opts_.max_ctx);
    const size_t max_pools = max_pos / static_cast<size_t>(cfg_.index_kpool) + 1;
    idx_sel_.reserve(static_cast<size_t>(cfg_.index_topk) + cfg_.index_kpool);

    is_kda_.assign(cfg_.n_layers, 0);
    kda_.resize(cfg_.n_layers);
    mla_.resize(cfg_.n_layers);
    for (int i = 0; i < cfg_.n_layers; ++i) {
        if (cfg_.layer_attn[i] == AttnKind::Kda) {
            is_kda_[i] = 1;
            kda_[i].conv.assign(static_cast<size_t>(cfg_.kda_conv - 1) * cfg_.kda_qkv_dim() * 3, 0.0f);
            kda_[i].rec.assign(static_cast<size_t>(cfg_.kda_heads) * cfg_.kda_head_dim * cfg_.kda_head_dim, 0.0f);
        } else {
            MlaCache& c = mla_[i];
            c.latent = KvCache(cfg_.kv_lora, max_pos, opts_.kv_dtype);
            c.idx_key = KvCache(cfg_.index_head_dim, max_pos, opts_.index_dtype);
            c.idx_gate = KvCache(cfg_.index_head_dim, max_pos, opts_.index_dtype);
            c.pool_keys.assign(max_pools * static_cast<size_t>(cfg_.index_head_dim), 0.0f);
            c.n_pools = 0;
        }
    }

    if (opts_.pin_nonexpert) {
        const size_t pinned = store_.pin_nonexpert();
        std::fprintf(stderr, "[pin ] non-expert weights resident: %.2f GiB\n",
                     static_cast<double>(pinned) / (1024.0 * 1024.0 * 1024.0));
    }
}

void Model::reset() {
    for (int i = 0; i < cfg_.n_layers; ++i) {
        std::fill(kda_[i].conv.begin(), kda_[i].conv.end(), 0.0f);
        std::fill(kda_[i].rec.begin(), kda_[i].rec.end(), 0.0f);
        mla_[i].latent.reset();
        mla_[i].idx_key.reset();
        mla_[i].idx_gate.reset();
        mla_[i].n_pools = 0;
    }
}

const std::vector<float>& Model::forward(int token_id, int pos) {
    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;
    const double t_begin = now_s();

    // Embedding: single row of token_embd.weight (ne = (hidden, vocab)).
    std::vector<float> emb(H);
    store_.read_slice("token_embd.weight", static_cast<uint64_t>(token_id) * H, H, emb.data());
    for (int n = 0; n < N; ++n) std::copy(emb.begin(), emb.end(), streams_.begin() + static_cast<size_t>(n) * H);

    for (int i = 0; i < cfg_.n_layers; ++i) layer_forward(i, pos);

    // hc_head: mean over streams, then final RMSNorm.
    std::vector<float> merged(H, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int d = 0; d < H; ++d) merged[d] += streams_[static_cast<size_t>(n) * H + d];
    const float inv_n = 1.0f / N;
    for (int d = 0; d < H; ++d) merged[d] *= inv_n;

    const float* final_norm = store_.get("output_norm.weight");
    std::vector<float> normed(H);
    rmsnorm(normed.data(), merged.data(), final_norm, H, cfg_.rms_eps);

    store_.matvec_quant("output.weight", 0, H, cfg_.vocab, normed.data(), logits_.data());
    if (prof().on) {
        const double t_end = now_s();
        const double layers = prof().hc + prof().attn + prof().mlp;
        std::fprintf(stderr,
                     "[prof] pos=%4d hc=%.3f attn=%.3f mlp=%.3f layers=%.3f head=%.3f total=%.3f\n",
                     pos, prof().hc, prof().attn, prof().mlp, layers, t_end - t_begin - layers,
                     t_end - t_begin);
        prof().hc = prof().attn = prof().mlp = 0;
    }
    return logits_;
}

const std::vector<float>& Model::prefill(const std::vector<int>& tokens, int pos_start) {
    const std::vector<float>* out = nullptr;
    if (opts_.batch > 1 && tokens.size() > 1) {
        size_t i = 0;
        while (i < tokens.size()) {
            const int n = static_cast<int>(
                std::min<size_t>(static_cast<size_t>(opts_.batch), tokens.size() - i));
            out = &run_chunk(tokens.data() + i, n, pos_start + static_cast<int>(i));
            i += static_cast<size_t>(n);
        }
        return *out;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        out = &forward(tokens[i], pos_start + static_cast<int>(i));
    }
    if (!out) throw std::runtime_error("prefill: empty prompt");
    return *out;
}

void Model::hc_forward(const std::string& prefix, float* post, float* comb) {
    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;
    const int HN = N * H;

    std::copy(streams_.begin(), streams_.end(), hc_flat_.begin());
    rmsnorm(hc_flat_.data(), hc_flat_.data(), nullptr, HN, cfg_.rms_eps);

    const float* fn = store_.get(prefix + "fn.weight");
    const float* base = store_.get(prefix + "base.weight");
    const float* scale = store_.get(prefix + "scale.weight");

    const int W = (2 + N) * N;  // 24
    std::vector<float> w(W, 0.0f);
    matvec(w.data(), fn, hc_flat_.data(), HN, W);

    for (int n = 0; n < N; ++n) {
        pre_[n] = sigmoidf(w[n] * scale[0] + base[n]) + cfg_.hc_eps;
        post[n] = 2.0f * sigmoidf(w[N + n] * scale[1] + base[N + n]);
    }

    // comb = softmax over j of (w * scale + base) + eps
    for (int i = 0; i < N; ++i) {
        std::vector<float> row(N);
        for (int j = 0; j < N; ++j) row[j] = w[2 * N + i * N + j] * scale[2] + base[2 * N + i * N + j];
        softmax(row, N);
        for (int j = 0; j < N; ++j) comb[i * N + j] = row[j] + cfg_.hc_eps;
    }
    // Sinkhorn: normalize columns, then rows/columns repeatedly.
    auto colsum = [&](int j) {
        float s = 0.0f;
        for (int i = 0; i < N; ++i) s += comb[i * N + j];
        return s + cfg_.hc_eps;
    };
    auto rowsum = [&](int i) {
        float s = 0.0f;
        for (int j = 0; j < N; ++j) s += comb[i * N + j];
        return s + cfg_.hc_eps;
    };
    for (int j = 0; j < N; ++j) {
        const float d = colsum(j);
        for (int i = 0; i < N; ++i) comb[i * N + j] /= d;
    }
    for (int it = 0; it < cfg_.hc_sinkhorn_iters - 1; ++it) {
        for (int i = 0; i < N; ++i) {
            const float d = rowsum(i);
            for (int j = 0; j < N; ++j) comb[i * N + j] /= d;
        }
        for (int j = 0; j < N; ++j) {
            const float d = colsum(j);
            for (int i = 0; i < N; ++i) comb[i * N + j] /= d;
        }
    }

    // collapsed = sum_n pre[n] * streams[n]
    std::fill(mixed_.begin(), mixed_.end(), 0.0f);
    for (int n = 0; n < N; ++n) {
        const float pre = pre_[n];
        for (int d = 0; d < H; ++d) mixed_[d] += pre * streams_[static_cast<size_t>(n) * H + d];
    }
}

void Model::layer_forward(int layer, int pos) {
    const int H = cfg_.hidden;
    const int N = cfg_.hc_mult;
    const double t0 = now_s();

    std::copy(streams_.begin(), streams_.end(), res_in_.begin());

    std::string ap = "blk." + std::to_string(layer) + ".hc_attn_";
    hc_forward(ap, post_.data(), comb_.data());
    const float* an = store_.get(blk(layer, "attn_norm.weight"));
    rmsnorm(mixed_.data(), mixed_.data(), an, H, cfg_.rms_eps);

    const double t1 = now_s();
    attention_forward(layer, pos);
    const double t2 = now_s();

    // streams = post[n] * attn_out + comb^T @ res_in
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < H; ++d) {
            float acc = post_[n] * attn_out_[d];
            for (int m = 0; m < N; ++m) acc += comb_[m * N + n] * res_in_[static_cast<size_t>(m) * H + d];
            streams_[static_cast<size_t>(n) * H + d] = acc;
        }
    }

    std::copy(streams_.begin(), streams_.end(), res_in_.begin());

    std::string fp = "blk." + std::to_string(layer) + ".hc_ffn_";
    hc_forward(fp, post_.data(), comb_.data());
    const float* fn_ = store_.get(blk(layer, "ffn_norm.weight"));
    rmsnorm(mixed_.data(), mixed_.data(), fn_, H, cfg_.rms_eps);

    const double t3 = now_s();
    if (cfg_.layer_mlp[layer] == MlpKind::Dense) {
        dense_mlp(blk(layer, "ffn_gate.weight").c_str(), blk(layer, "ffn_up.weight").c_str(),
                  blk(layer, "ffn_down.weight").c_str());
    } else {
        moe_forward(layer);
    }
    const double t4 = now_s();
    if (prof().on) {
        prof().hc += (t1 - t0) + (t3 - t2);
        prof().attn += t2 - t1;
        prof().mlp += t4 - t3;
    }

    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < H; ++d) {
            float acc = post_[n] * mlp_out_[d];
            for (int m = 0; m < N; ++m) acc += comb_[m * N + n] * res_in_[static_cast<size_t>(m) * H + d];
            streams_[static_cast<size_t>(n) * H + d] = acc;
        }
    }
}

void Model::attention_forward(int layer, int pos) {
    if (is_kda_[layer]) {
        kda_forward(layer, pos);
    } else {
        mla_forward(layer, pos);
    }
}

void Model::kda_forward(int layer, int pos, const KdaProjView* pv, float* gated_out) {
    (void)pos;
    const int H = cfg_.hidden;
    const int NH = cfg_.kda_heads;
    const int D = cfg_.kda_head_dim;
    const int QKV = NH * D;  // 8192
    const int CD = 3 * QKV;

    std::vector<float> raw_own;
    const float* raw = nullptr;
    if (pv && pv->qkv) {
        raw = pv->qkv;
    } else {
        raw_own.resize(CD);
        raw = raw_own.data();
        store_.matvec_quant(blk(layer, "attn_q.weight"), 0, H, QKV, mixed_.data(), raw_own.data());
        store_.matvec_quant(blk(layer, "attn_k.weight"), 0, H, QKV, mixed_.data(), raw_own.data() + QKV);
        store_.matvec_quant(blk(layer, "attn_v.weight"), 0, H, QKV, mixed_.data(), raw_own.data() + 2 * QKV);
    }

    // Depthwise causal conv (kernel = cfg_.kda_conv) + SiLU, split per q/k/v tensor.
    const float* cw[3] = {store_.get(blk(layer, "ssm_conv1d_q.weight")),
                          store_.get(blk(layer, "ssm_conv1d_k.weight")),
                          store_.get(blk(layer, "ssm_conv1d_v.weight"))};

    std::vector<float> conv_out(CD);
    float* state = kda_[layer].conv.data();
    const int K = cfg_.kda_conv;
    for (int c = 0; c < CD; ++c) {
        const float* wc = cw[c / QKV] + static_cast<size_t>(K) * (c % QKV);
        float acc = 0.0f;
        for (int j = 0; j < K - 1; ++j) acc += wc[j] * state[static_cast<size_t>(c) * (K - 1) + j];
        acc += wc[K - 1] * raw[c];
        for (int j = 0; j < K - 2; ++j) state[static_cast<size_t>(c) * (K - 1) + j] = state[static_cast<size_t>(c) * (K - 1) + j + 1];
        if (K >= 2) state[static_cast<size_t>(c) * (K - 1) + (K - 2)] = raw[c];
        conv_out[c] = siluf(acc);
    }

    const float* q = conv_out.data();
    const float* k = conv_out.data() + QKV;
    const float* v = conv_out.data() + 2 * QKV;

    // Forget gate: g = lower_bound * sigmoid(-A * (f_b(f_a(x)) + dt_bias))
    const float* dt = store_.get(blk(layer, "ssm_dt.bias"));
    const float* A = store_.get(blk(layer, "ssm_a"));
    std::vector<float> fg_own, beta_own;
    const float* fg = nullptr;
    const float* beta_raw = nullptr;
    if (pv && pv->fg) {
        fg = pv->fg;
        beta_raw = pv->beta;
    } else {
        const float* fa = store_.get(blk(layer, "ssm_f_a.weight"));
        const float* fb = store_.get(blk(layer, "ssm_f_b.weight"));
        std::vector<float> fa_out(128);
        fg_own.assign(QKV, 0.0f);
        beta_own.assign(NH, 0.0f);
        matvec(fa_out.data(), fa, mixed_.data(), H, 128);
        matvec(fg_own.data(), fb, fa_out.data(), 128, QKV);
        matvec(beta_own.data(), store_.get(blk(layer, "ssm_beta.weight")), mixed_.data(), H, NH);
        fg = fg_own.data();
        beta_raw = beta_own.data();
    }
    std::vector<float> g(QKV);
    for (int h = 0; h < NH; ++h) {
        const float decay = -A[h];
        for (int d = 0; d < D; ++d) {
            const float gg = fg[h * D + d] + dt[h * D + d];
            g[h * D + d] = cfg_.kda_gate_lower_bound * sigmoidf(decay * gg);
        }
    }

    // beta = sigmoid(b_proj(x))
    std::vector<float> beta_g(NH);
    for (int h = 0; h < NH; ++h) beta_g[h] = sigmoidf(beta_raw[h]);

    // l2norm q,k and scale q
    const float qscale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> qq(QKV), kk(QKV);
    for (int h = 0; h < NH; ++h) {
        float sq = 1e-6f, sk = 1e-6f;
        for (int d = 0; d < D; ++d) {
            sq += q[h * D + d] * q[h * D + d];
            sk += k[h * D + d] * k[h * D + d];
        }
        const float iq = 1.0f / std::sqrt(sq) * qscale;
        const float ik = 1.0f / std::sqrt(sk);
        for (int d = 0; d < D; ++d) {
            qq[h * D + d] = q[h * D + d] * iq;
            kk[h * D + d] = k[h * D + d] * ik;
        }
    }

    float* st = kda_[layer].rec.data();
    std::vector<float> out(QKV);
    for (int h = 0; h < NH; ++h) {
        float* Sh = st + static_cast<size_t>(h) * D * D;
        const float* qh = qq.data() + h * D;
        const float* kh = kk.data() + h * D;
        const float* vh = v + h * D;
        const float* gh = g.data() + h * D;
        const float bh = beta_g[h];
        for (int a = 0; a < D; ++a) {
            const float decay = std::exp(gh[a]);
            float* row = Sh + static_cast<size_t>(a) * D;
            for (int b = 0; b < D; ++b) row[b] *= decay;
        }
        std::vector<float> kv_mem(D, 0.0f);
        for (int a = 0; a < D; ++a) {
            const float ka = kh[a];
            const float* row = Sh + static_cast<size_t>(a) * D;
            for (int b = 0; b < D; ++b) kv_mem[b] += row[b] * ka;
        }
        std::vector<float> delta(D);
        for (int b = 0; b < D; ++b) delta[b] = (vh[b] - kv_mem[b]) * bh;
        for (int a = 0; a < D; ++a) {
            const float ka = kh[a];
            float* row = Sh + static_cast<size_t>(a) * D;
            for (int b = 0; b < D; ++b) row[b] += ka * delta[b];
        }
        float* oh = out.data() + h * D;
        for (int b = 0; b < D; ++b) {
            float acc = 0.0f;
            for (int a = 0; a < D; ++a) acc += Sh[static_cast<size_t>(a) * D + b] * qh[a];
            oh[b] = acc;
        }
    }

    // gate = g_b(g_a(x)); gated RMSNorm per head
    const float* onw = store_.get(blk(layer, "ssm_norm.weight"));
    std::vector<float> gate_own;
    const float* gate = nullptr;
    if (pv && pv->ggate) {
        gate = pv->ggate;
    } else {
        const float* ga = store_.get(blk(layer, "ssm_g_a.weight"));
        const float* gb = store_.get(blk(layer, "ssm_g_b.weight"));
        std::vector<float> ga_out(128);
        gate_own.assign(QKV, 0.0f);
        matvec(ga_out.data(), ga, mixed_.data(), H, 128);
        matvec(gate_own.data(), gb, ga_out.data(), 128, QKV);
        gate = gate_own.data();
    }
    float* gated = gated_out;
    std::vector<float> gated_own;
    if (!gated) {
        gated_own.assign(QKV, 0.0f);
        gated = gated_own.data();
    }
    for (int h = 0; h < NH; ++h) {
        float ss = 0.0f;
        for (int d = 0; d < D; ++d) ss += out[h * D + d] * out[h * D + d];
        const float rstd = 1.0f / std::sqrt(ss / D + cfg_.rms_eps);
        for (int d = 0; d < D; ++d) {
            const float y = out[h * D + d] * rstd * onw[d];
            gated[h * D + d] = y * sigmoidf(gate[h * D + d]);
        }
    }

    // The chunked path projects all tokens at once and leaves `gated` for the caller.
    if (gated_out) return;
    store_.matvec_quant(blk(layer, "attn_output.weight"), 0, QKV, H, gated, attn_out_.data());
}

void Model::indexer_append(int layer, const float* k_val, const float* g_val) {
    const int D = cfg_.index_head_dim;
    const int KP = cfg_.index_kpool;
    MlaCache& cache = mla_[layer];

    cache.idx_key.append(k_val);
    cache.idx_gate.append(g_val);

    const size_t len = cache.idx_key.size();
    if (len % static_cast<size_t>(KP) == 0) {
        const int g = static_cast<int>(len / static_cast<size_t>(KP)) - 1;
        const float* ape = store_.get(blk(layer, "indexer_compressor_ape.weight"));  // [KP][D]
        std::vector<float> gk(static_cast<size_t>(KP) * D), gg(static_cast<size_t>(KP) * D);
        for (int pp = 0; pp < KP; ++pp) {
            cache.idx_key.dequant_row(len - KP + pp, gk.data() + static_cast<size_t>(pp) * D);
            cache.idx_gate.dequant_row(len - KP + pp, gg.data() + static_cast<size_t>(pp) * D);
        }
        indexer_pool_key(indexer_params(), gk.data(), gg.data(), ape,
                         cache.pool_keys.data() + static_cast<size_t>(g) * D);
        cache.n_pools = g + 1;
    }
}

IndexerParams Model::indexer_params() const {
    IndexerParams p;
    p.n_heads = cfg_.index_n_heads;
    p.head_dim = cfg_.index_head_dim;
    p.kpool = cfg_.index_kpool;
    p.topk = cfg_.index_topk;
    p.always_select_tail = cfg_.index_kpool_always_select_tail;
    return p;
}

void Model::indexer_select_vals(int layer, const float* idx_q, const float* idx_w,
                                std::vector<int>& sel) {
    MlaCache& cache = mla_[layer];
    const int kv_len = static_cast<int>(cache.idx_key.size());
    sel = glm5::indexer_select(indexer_params(), cache.pool_keys.data(), cache.n_pools, idx_q,
                               idx_w, kv_len);
}

void Model::mla_forward(int layer, int pos, const MlaProjView* pv, float* out_proj) {
    (void)pos;
    const int H = cfg_.hidden;
    const int NH = cfg_.n_heads;
    const int DK = cfg_.qk_nope;   // 256
    const int DV = cfg_.v_head;    // 256
    const int LR = cfg_.q_lora;    // 1536
    const int KVL = cfg_.kv_lora;  // 512

    // Projections (batched over the chunk when `pv` is supplied).
    const float* q = nullptr;
    const float* k_pass = nullptr;
    const float* idx_k = nullptr;
    const float* idx_gate = nullptr;
    const float* idx_q = nullptr;
    const float* idx_w = nullptr;
    std::vector<float> q_own, kpass_own;
    std::vector<float> q_resid(LR);
    if (pv && pv->q) {
        q = pv->q;
        k_pass = pv->k_pass;
        idx_k = pv->idx_k;
        idx_gate = pv->idx_gate;
        idx_q = pv->idx_q;
        idx_w = pv->idx_w;
    } else {
        std::vector<float> qa(LR);
        store_.matvec_quant(blk(layer, "attn_q_a.weight"), 0, H, LR, mixed_.data(), qa.data());
        rmsnorm(q_resid.data(), qa.data(), store_.get(blk(layer, "attn_q_a_norm.weight")), LR,
                cfg_.rms_eps);
        q_own.resize(static_cast<size_t>(NH) * DK);
        store_.matvec_quant(blk(layer, "attn_q_b.weight"), 0, LR, NH * DK, q_resid.data(),
                            q_own.data());
        q = q_own.data();

        std::vector<float> kv(KVL);
        store_.matvec_quant(blk(layer, "attn_kv_a_mqa.weight"), 0, H, KVL, mixed_.data(), kv.data());
        kpass_own.resize(KVL);
        rmsnorm(kpass_own.data(), kv.data(), store_.get(blk(layer, "attn_kv_a_norm.weight")), KVL,
                cfg_.rms_eps);
        k_pass = kpass_own.data();

        const int D = cfg_.index_head_dim;
        const int IH = cfg_.index_n_heads;
        store_.matvec_quant(blk(layer, "indexer.attn_k.weight"), 0, H, D, mixed_.data(),
                            idx_k_.data());
        layernorm(idx_k_.data(), idx_k_.data(), store_.get(blk(layer, "indexer.k_norm.weight")),
                  store_.get(blk(layer, "indexer.k_norm.bias")), D, cfg_.norm_eps);
        store_.matvec_quant(blk(layer, "indexer_compressor_gate.weight"), 0, H, D, mixed_.data(),
                            idx_gate_.data());
        store_.matvec_quant(blk(layer, "indexer.attn_q_b.weight"), 0, LR, IH * D, q_resid.data(),
                            idx_q_.data());
        store_.matvec_quant(blk(layer, "indexer.proj.weight"), 0, H, IH, mixed_.data(),
                            idx_w_.data());
        const float wscale = 1.0f / std::sqrt(static_cast<float>(IH));
        for (int h = 0; h < IH; ++h) idx_w_[static_cast<size_t>(h)] *= wscale;
        idx_k = idx_k_.data();
        idx_gate = idx_gate_.data();
        idx_q = idx_q_.data();
        idx_w = idx_w_.data();
    }

    MlaCache& cache = mla_[layer];
    cache.latent.append(k_pass);
    const int kv_len = static_cast<int>(cache.latent.size());

    if (layer == 3 && std::getenv("GLM5_DEBUG")) {
        double ss = 0.0;
        float am = 0.0f;
        for (int i = 0; i < KVL; ++i) {
            ss += static_cast<double>(k_pass[i]) * k_pass[i];
            am = std::max(am, std::fabs(k_pass[i]));
        }
        const double rms = std::sqrt(ss / KVL);
        std::fprintf(stderr, "[dbg] layer3 latent rms=%.3f amax=%.3f amax/rms=%.2f |", rms, am,
                     am / rms);
        std::vector<float> dq(KVL);
        for (KvDtype dt : {KvDtype::BF16, KvDtype::Q8_0, KvDtype::Q4_0, KvDtype::Q2_0}) {
            KvCache tmp(KVL, 1, dt);
            tmp.append(k_pass);
            tmp.dequant_row(0, dq.data());
            double es = 0.0;
            for (int i = 0; i < KVL; ++i) {
                const double e = dq[i] - k_pass[i];
                es += e * e;
            }
            std::fprintf(stderr, " %s=%.2f%%", kv_dtype_name(dt),
                         100.0 * std::sqrt(es / KVL) / rms);
        }
        std::fprintf(stderr, "\n");
    }

    // Key selection (DSA indexer, or full causal when disabled).
    std::vector<int>& sel = idx_sel_;
    if (opts_.use_indexer) {
        indexer_append(layer, idx_k, idx_gate);
        indexer_select_vals(layer, idx_q, idx_w, sel);
    } else {
        sel.resize(static_cast<size_t>(kv_len));
        for (int p = 0; p < kv_len; ++p) sel[static_cast<size_t>(p)] = p;
    }
    const int nsel = static_cast<int>(sel.size());

    // Absorbed query: q_abs[h] = W_k[h]^T q_nope[h]   (512 per head)
    for (int h = 0; h < NH; ++h) {
        store_.matvec_quant(blk(layer, "attn_k_b.weight"), static_cast<uint64_t>(h) * KVL, DK, KVL,
                            q + static_cast<size_t>(h) * DK,
                            q_abs_.data() + static_cast<size_t>(h) * KVL);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(DK + cfg_.qk_rope));
    std::vector<float> scores(static_cast<size_t>(nsel));
    float* out = out_proj;
    std::vector<float> out_own;
    if (!out) {
        out_own.assign(static_cast<size_t>(NH) * DV, 0.0f);
        out = out_own.data();
    } else {
        std::fill(out, out + static_cast<size_t>(NH) * DV, 0.0f);
    }
    for (int h = 0; h < NH; ++h) {
        const float* qa_h = q_abs_.data() + static_cast<size_t>(h) * KVL;
        for (int s = 0; s < nsel; ++s) {
            float acc = 0.0f;
            cache.latent.dot_row(static_cast<size_t>(sel[static_cast<size_t>(s)]), qa_h, &acc);
            scores[static_cast<size_t>(s)] = acc * scale;
        }
        softmax(scores, nsel);

        std::fill(c_comb_.begin(), c_comb_.end(), 0.0f);
        for (int s = 0; s < nsel; ++s)
            cache.latent.axpy_row(static_cast<size_t>(sel[static_cast<size_t>(s)]), scores[static_cast<size_t>(s)], c_comb_.data());

        store_.matvec_quant(blk(layer, "attn_v_b.weight"), static_cast<uint64_t>(h) * DV, KVL, DV,
                            c_comb_.data(), out + static_cast<size_t>(h) * DV);
    }

    // The chunked path projects all tokens at once and leaves `out` for the caller.
    if (out_proj) return;
    store_.matvec_quant(blk(layer, "attn_output.weight"), 0, NH * DV, H, out, attn_out_.data());
}

void Model::dense_mlp(const char* gate, const char* up, const char* down) {
    const int H = cfg_.hidden;
    const int I = cfg_.intermediate;
    std::vector<float> g(I), u(I), h(I);
    store_.matvec_quant(gate, 0, H, I, mixed_.data(), g.data());
    store_.matvec_quant(up, 0, H, I, mixed_.data(), u.data());
    const float lim = cfg_.swiglu_limit;
    for (int i = 0; i < I; ++i) {
        float gi = std::min(g[i], lim);
        float ui = std::clamp(u[i], -lim, lim);
        h[i] = siluf(gi) * ui;
    }
    store_.matvec_quant(down, 0, I, H, h.data(), mlp_out_.data());
}

void Model::moe_forward(int layer) {
    const int H = cfg_.hidden;
    const int I = cfg_.moe_inter;
    const int E = cfg_.n_experts;
    const int K = cfg_.top_k;

    std::copy(mixed_.begin(), mixed_.end(), moe_x_.begin());

    // Router
    std::vector<float> logits(E), scores(E), choice(E);
    matvec(logits.data(), store_.get(blk(layer, "ffn_gate_inp.weight")), moe_x_.data(), H, E);
    const float* corr = store_.get(blk(layer, "exp_probs_b.bias"));
    for (int e = 0; e < E; ++e) {
        scores[e] = sigmoidf(logits[e]);
        choice[e] = scores[e] + corr[e];
    }

    // Top-k by `choice` (group masking is a no-op with n_group == 1).
    std::vector<int> idx(E);
    for (int e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return choice[a] > choice[b]; });

    std::vector<float> wsum(K);
    float denom = 1e-20f;
    for (int k = 0; k < K; ++k) {
        wsum[k] = scores[idx[k]];
        denom += wsum[k];
    }
    if (cfg_.norm_topk) {
        for (int k = 0; k < K; ++k) wsum[k] /= denom;
    }
    for (int k = 0; k < K; ++k) wsum[k] *= cfg_.routed_scale;

    std::fill(mlp_out_.begin(), mlp_out_.end(), 0.0f);
    const float lim = cfg_.swiglu_limit;
    std::vector<float> g(I), u(I), h(I);
    for (int k = 0; k < K; ++k) {
        const int e = idx[k];
        const uint64_t e_in = static_cast<uint64_t>(e) * I;   // gate/up: I rows per expert
        const uint64_t e_out = static_cast<uint64_t>(e) * H;  // down: H rows per expert
        store_.matvec_quant(blk(layer, "ffn_gate_exps.weight"), e_in, H, I, moe_x_.data(), g.data());
        store_.matvec_quant(blk(layer, "ffn_up_exps.weight"), e_in, H, I, moe_x_.data(), u.data());
        for (int i = 0; i < I; ++i) {
            const float gi = std::min(g[i], lim);
            const float ui = std::clamp(u[i], -lim, lim);
            h[i] = siluf(gi) * ui;
        }
        store_.matvec_quant(blk(layer, "ffn_down_exps.weight"), e_out, I, H, h.data(), e_dn_.data());
        const float w = wsum[k];
        axpy_f32(mlp_out_.data(), e_dn_.data(), w, H);
    }

    // Shared expert (runs on the same input).
    std::vector<float> sg(I), su(I), sh(I), sd(H);
    store_.matvec_quant(blk(layer, "ffn_gate_shexp.weight"), 0, H, I, moe_x_.data(), sg.data());
    store_.matvec_quant(blk(layer, "ffn_up_shexp.weight"), 0, H, I, moe_x_.data(), su.data());
    for (int i = 0; i < I; ++i) {
        const float gi = std::min(sg[i], lim);
        const float ui = std::clamp(su[i], -lim, lim);
        sh[i] = siluf(gi) * ui;
    }
    store_.matvec_quant(blk(layer, "ffn_down_shexp.weight"), 0, I, H, sh.data(), sd.data());
    add_f32(mlp_out_.data(), sd.data(), H);
}

}  // namespace glm5
