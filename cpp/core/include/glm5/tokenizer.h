#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glm5/gguf.h"

namespace glm5 {

// GPT-2 byte-level BPE tokenizer with the GLM4 ("chatglm4") pre-tokenizer.
class Tokenizer {
public:
    explicit Tokenizer(const GgufModel& model);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int token_to_id(const std::string& tok) const;
    const std::string& id_to_token(int id) const;

    int eos_id() const { return eos_; }
    int eot_id() const { return eot_; }
    int eom_id() const { return eom_; }

    bool is_special(int id) const;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_;
    std::unordered_map<std::string, int> specials_;  // special token text -> id
    std::vector<char> is_special_;
    int eos_ = -1, eot_ = -1, eom_ = -1;
};

}  // namespace glm5
