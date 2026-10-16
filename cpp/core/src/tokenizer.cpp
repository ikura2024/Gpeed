#include "glm5/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace glm5 {
namespace {

// ---- UTF-8 -----------------------------------------------------------------
int utf8_decode(const std::string& s, size_t i, uint32_t* cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) { *cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0) {
        *cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        *cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---- GPT-2 byte <-> unicode -------------------------------------------------
struct ByteMap {
    uint32_t b2u[256];
    int u2b[0x200];  // codepoint (up to 0x1FF) -> byte, -1 if unused

    ByteMap() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) b2u[bs[i]] = static_cast<uint32_t>(cs[i]);
        std::fill(std::begin(u2b), std::end(u2b), -1);
        for (size_t i = 0; i < bs.size(); ++i) {
            if (cs[i] < 0x200) u2b[cs[i]] = bs[i];
        }
    }
};

const ByteMap& byte_map() {
    static ByteMap m;
    return m;
}

std::string byte_encode(const std::string& raw) {
    const ByteMap& m = byte_map();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) utf8_append(out, m.b2u[c]);
    return out;
}

// ---- character classes (approximation of \s / \p{L} / \p{N}) -----------------
bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

bool is_space(uint32_t c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C) return true;
    if (c == 0x85 || c == 0xA0 || c == 0x1680) return true;
    if (c >= 0x2000 && c <= 0x200A) return true;
    if (c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) return true;
    return false;
}

bool is_num(uint32_t c) { return c >= '0' && c <= '9'; }

bool is_letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c < 0x80) return false;
    return !is_space(c) && !is_num(c);
}

// ---- pre-tokenizer regex (LLAMA_VOCAB_PRE_TYPE_CHATGLM4) -------------------
// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}{1,3}
// | ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
size_t match_alt(const std::vector<uint32_t>& c, size_t i) {
    const size_t n = c.size();
    const uint32_t ch = c[i];

    // alt 1
    if (ch == '\'' && i + 1 < n) {
        const uint32_t a = c[i + 1];
        if (a == 's' || a == 'S' || a == 't' || a == 'T' || a == 'm' || a == 'M' || a == 'd' || a == 'D')
            return i + 2;
        if ((a == 'r' || a == 'R' || a == 'v' || a == 'V') && i + 2 < n) {
            const uint32_t b = c[i + 2];
            if (b == 'e' || b == 'E') return i + 3;
        }
        if ((a == 'l' || a == 'L') && i + 2 < n && (c[i + 2] == 'l' || c[i + 2] == 'L')) return i + 3;
    }

    // alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
    for (int opt = 1; opt >= 0; --opt) {
        size_t j = i;
        if (opt) {
            if (j >= n || is_crlf(c[j]) || is_letter(c[j]) || is_num(c[j])) continue;
            ++j;
        }
        if (j < n && is_letter(c[j])) {
            while (j < n && is_letter(c[j])) ++j;
            return j;
        }
    }

    // alt 3: \p{N}{1,3}
    if (is_num(ch)) {
        size_t j = i, cnt = 0;
        while (j < n && is_num(c[j]) && cnt < 3) {
            ++j;
            ++cnt;
        }
        return j;
    }

    // alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        size_t j = i;
        if (j < n && c[j] == ' ') ++j;
        size_t k = j;
        while (k < n && !is_space(c[k]) && !is_letter(c[k]) && !is_num(c[k])) ++k;
        if (k > j) {
            while (k < n && is_crlf(c[k])) ++k;
            return k;
        }
    }

    // alt 5: \s*[\r\n]+
    {
        size_t j = i;
        while (j < n && is_space(c[j]) && !is_crlf(c[j])) ++j;
        if (j < n && is_crlf(c[j])) {
            while (j < n && is_crlf(c[j])) ++j;
            return j;
        }
    }

    // alt 6: \s+(?!\S)
    {
        size_t j = i;
        while (j < n && is_space(c[j])) ++j;
        if (j > i) {
            for (size_t e = j; e > i; --e) {
                if (e >= n || is_space(c[e])) return e;
            }
        }
    }

    // alt 7: \s+
    if (is_space(ch)) {
        size_t j = i;
        while (j < n && is_space(c[j])) ++j;
        return j;
    }

    return i + 1;  // safety: never stall
}

std::vector<std::pair<size_t, size_t>> regex_split(const std::string& text) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        i += utf8_decode(text, i, &cp);
        cps.push_back(cp);
    }
    // Byte offsets per codepoint.
    std::vector<size_t> off(cps.size() + 1, 0);
    {
        size_t k = 0;
        for (size_t i = 0; i < text.size() && k < cps.size();) {
            off[k] = i;
            uint32_t cp = 0;
            i += utf8_decode(text, i, &cp);
            ++k;
        }
        off[cps.size()] = text.size();
    }

    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < cps.size();) {
        const size_t e = match_alt(cps, i);
        out.emplace_back(off[i], off[std::min(e, cps.size())]);
        i = e;
    }
    return out;
}

}  // namespace

Tokenizer::Tokenizer(const GgufModel& model) {
    const GgufValue* tokens = model.find("tokenizer.ggml.tokens");
    if (!tokens || !tokens->is_array()) throw std::runtime_error("tokenizer: no tokens");
    for (const auto& v : tokens->array) id_to_token_.push_back(v.as_str());
    for (size_t i = 0; i < id_to_token_.size(); ++i) token_to_id_[id_to_token_[i]] = static_cast<int>(i);

    const GgufValue* merges = model.find("tokenizer.ggml.merges");
    if (merges && merges->is_array()) {
        int rank = 0;
        for (const auto& v : merges->array) {
            const std::string& s = v.as_str();
            const size_t sp = s.find(' ');
            if (sp == std::string::npos) continue;
            merge_rank_[s.substr(0, sp) + std::string(1, '\x01') + s.substr(sp + 1)] = rank++;
        }
    }

    is_special_.assign(id_to_token_.size(), 0);
    const GgufValue* types = model.find("tokenizer.ggml.token_type");
    if (types && types->is_array()) {
        for (size_t i = 0; i < types->array.size() && i < id_to_token_.size(); ++i) {
            const int64_t t = types->array[i].as_i64();
            if (t == 3 || t == 4) {  // CONTROL / USER_DEFINED
                is_special_[i] = 1;
                specials_[id_to_token_[i]] = static_cast<int>(i);
            }
        }
    }

    auto get_id = [&](const char* key, int def) {
        const GgufValue* v = model.find(key);
        return v ? static_cast<int>(v->as_i64()) : def;
    };
    eos_ = get_id("tokenizer.ggml.eos_token_id", -1);
    eot_ = get_id("tokenizer.ggml.eot_token_id", -1);
    eom_ = get_id("tokenizer.ggml.eom_token_id", -1);
}

const std::string& Tokenizer::id_to_token(int id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return empty;
    return id_to_token_[id];
}

int Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int id) const {
    return id >= 0 && id < static_cast<int>(is_special_.size()) && is_special_[id];
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;

    // Special-token-aware scan.
    std::vector<std::pair<std::string, int>> sp;
    for (const auto& kv : specials_) sp.push_back({kv.first, kv.second});
    std::sort(sp.begin(), sp.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    auto bpe = [&](const std::string& chunk) {
        if (chunk.empty()) return;
        for (auto [a, b] : regex_split(chunk)) {
            const std::string raw = chunk.substr(a, b - a);
            const std::string word = byte_encode(raw);
            if (word.empty()) continue;

            // ignore_merges: whole word hit
            auto whole = token_to_id_.find(word);
            if (whole != token_to_id_.end()) {
                out.push_back(whole->second);
                continue;
            }

            // Split into codepoint symbols.
            std::vector<std::string> syms;
            for (size_t i = 0; i < word.size();) {
                uint32_t cp = 0;
                const int len = utf8_decode(word, i, &cp);
                syms.push_back(word.substr(i, len));
                i += len;
            }

            while (syms.size() > 1) {
                int best_rank = INT32_MAX, best = -1;
                for (size_t i = 0; i + 1 < syms.size(); ++i) {
                    auto it = merge_rank_.find(syms[i] + std::string(1, '\x01') + syms[i + 1]);
                    if (it != merge_rank_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best = static_cast<int>(i);
                    }
                }
                if (best < 0) break;
                syms[best] += syms[best + 1];
                syms.erase(syms.begin() + best + 1);
            }

            for (const auto& s : syms) {
                auto it = token_to_id_.find(s);
                out.push_back(it == token_to_id_.end() ? 0 : it->second);
            }
        }
    };

    std::string pending;
    for (size_t i = 0; i < text.size();) {
        int matched = -1;
        size_t matched_len = 0;
        for (const auto& [s, id] : sp) {
            if (i + s.size() <= text.size() && text.compare(i, s.size(), s) == 0) {
                matched = id;
                matched_len = s.size();
                break;
            }
        }
        if (matched >= 0) {
            bpe(pending);
            pending.clear();
            out.push_back(matched);
            i += matched_len;
        } else {
            uint32_t cp = 0;
            const int len = utf8_decode(text, i, &cp);
            pending.append(text, i, len);
            i += static_cast<size_t>(len);
        }
    }
    bpe(pending);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    const ByteMap& m = byte_map();
    std::string out;
    for (int id : ids) {
        if (is_special(id)) {
            out += id_to_token(id);
            continue;
        }
        const std::string& tok = id_to_token(id);
        for (size_t i = 0; i < tok.size();) {
            uint32_t cp = 0;
            const int len = utf8_decode(tok, i, &cp);
            const int b = (cp < 0x200) ? m.u2b[cp] : -1;
            if (b >= 0) {
                out.push_back(static_cast<char>(b));
            } else {
                out.append(tok, i, len);
            }
            i += static_cast<size_t>(len);
        }
    }
    return out;
}

}  // namespace glm5
