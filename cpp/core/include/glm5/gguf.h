#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "glm5/common.h"

namespace glm5 {

// GGUF metadata value kinds.
enum class GgufValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

struct GgufValue {
    GgufValueType type = GgufValueType::UINT8;

    // Scalar payload (interpreted per `type`).
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
    } scalar{};

    std::string str;                     // STRING
    GgufValueType array_type = GgufValueType::UINT8;
    std::vector<GgufValue> array;        // ARRAY

    bool is_array() const { return type == GgufValueType::ARRAY; }

    // Convenience accessors with permissive integer/float coercion.
    uint64_t as_u64() const;
    int64_t as_i64() const;
    double as_f64() const;
    bool as_bool() const;
    const std::string& as_str() const;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // as stored in the file (ggml ne order)
    DType type = DType::F32;
    uint64_t offset = 0;         // offset into the tensor-data section
    int shard = 0;               // index of the source GgufFile

    uint64_t n_elements() const;
    uint64_t row_bytes() const;  // bytes of one row (dims[0] elements)
};

// A single GGUF file (one shard).
class GgufFile {
public:
    static std::shared_ptr<GgufFile> open(const std::string& path);

    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    uint64_t data_offset() const { return data_offset_; }
    size_t n_tensors() const { return tensors_.size(); }

    bool has(const std::string& key) const;
    const GgufValue* find(const std::string& key) const;
    const GgufValue& get(const std::string& key) const;
    std::vector<std::string> keys() const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find_tensor(const std::string& name) const;

    // Tags every tensor with the shard index (used when merging multiple files).
    void set_shard(int shard);

    // Reads the full tensor payload into `dst` (must be at least row_bytes*nrows).
    void read_tensor(const TensorInfo& t, void* dst) const;

    // Read a raw byte range inside the tensor-data section.
    void read_at(uint64_t offset, size_t bytes, void* dst) const;

    // Zero-copy pointer to the start of the tensor-data section (nullptr when the
    // file could not be memory-mapped; fall back to read_at in that case).
    const uint8_t* data_ptr() const {
        ensure_mapped();
        return map_view_ ? static_cast<const uint8_t*>(map_view_) + data_offset_ : nullptr;
    }

private:
    GgufFile() = default;
    void parse();
    void ensure_mapped() const;

    std::string path_;
    uint32_t version_ = 0;
    int shard_ = 0;
    uint64_t data_offset_ = 0;
    std::vector<uint8_t> header_;  // header + metadata + tensor infos
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    std::unordered_map<std::string, GgufValue> metadata_;
    mutable std::ifstream stream_;  // opened on the tensor-data section
    mutable void* map_file_ = nullptr;
    mutable void* map_view_ = nullptr;
    std::string open_error_;
};

// One or more shards forming a logical model.
class GgufModel {
public:
    // `paths` are the shard files in order.
    void load(const std::vector<std::string>& paths);

    const GgufValue* find(const std::string& key) const;
    const GgufFile& file(size_t i) const { return *files_.at(i); }
    const std::vector<std::shared_ptr<GgufFile>>& files() const { return files_; }
    std::vector<std::string> keys() const;

    const TensorInfo* find_tensor(const std::string& name) const;
    const std::vector<std::string>& tensor_names() const { return tensor_names_; }
    void read_tensor(const TensorInfo& t, void* dst) const;

    // Glob-discovery of "*-00001-of-NNNNN.gguf" shards given any one shard path.
    static std::vector<std::string> discover_shards(const std::string& any_shard);

private:
    std::vector<std::shared_ptr<GgufFile>> files_;
    std::unordered_map<std::string, const TensorInfo*> tensor_map_;
    std::vector<std::string> tensor_names_;
};

}  // namespace glm5
