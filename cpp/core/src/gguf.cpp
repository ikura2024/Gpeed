#include "glm5/gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace glm5 {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u;  // "GGUF"

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    void need(size_t n) const {
        if (p + n > end) throw std::runtime_error("GGUF: unexpected end of buffer");
    }
    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string read_str() {
        uint64_t n = read<uint64_t>();
        need(static_cast<size_t>(n));
        std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
        p += n;
        return s;
    }
};

GgufValue read_value(Cursor& c, GgufValueType type);

GgufValue read_array(Cursor& c) {
    GgufValue v;
    v.type = GgufValueType::ARRAY;
    v.array_type = static_cast<GgufValueType>(c.read<uint32_t>());
    uint64_t n = c.read<uint64_t>();
    v.array.reserve(static_cast<size_t>(std::min<uint64_t>(n, 1u << 20)));
    for (uint64_t i = 0; i < n; ++i) v.array.push_back(read_value(c, v.array_type));
    return v;
}

GgufValue read_value(Cursor& c, GgufValueType type) {
    GgufValue v;
    v.type = type;
    switch (type) {
        case GgufValueType::UINT8:   v.scalar.u8 = c.read<uint8_t>(); break;
        case GgufValueType::INT8:    v.scalar.i8 = c.read<int8_t>(); break;
        case GgufValueType::UINT16:  v.scalar.u16 = c.read<uint16_t>(); break;
        case GgufValueType::INT16:   v.scalar.i16 = c.read<int16_t>(); break;
        case GgufValueType::UINT32:  v.scalar.u32 = c.read<uint32_t>(); break;
        case GgufValueType::INT32:   v.scalar.i32 = c.read<int32_t>(); break;
        case GgufValueType::FLOAT32: v.scalar.f32 = c.read<float>(); break;
        case GgufValueType::BOOL:    v.scalar.b = c.read<uint8_t>() != 0; break;
        case GgufValueType::STRING:  v.str = c.read_str(); break;
        case GgufValueType::UINT64:  v.scalar.u64 = c.read<uint64_t>(); break;
        case GgufValueType::INT64:   v.scalar.i64 = c.read<int64_t>(); break;
        case GgufValueType::FLOAT64: v.scalar.f64 = c.read<double>(); break;
        case GgufValueType::ARRAY:   return read_array(c);
        default: throw std::runtime_error("GGUF: bad value type");
    }
    return v;
}

}  // namespace

uint64_t GgufValue::as_u64() const {
    switch (type) {
        case GgufValueType::UINT8: return scalar.u8;
        case GgufValueType::INT8: return static_cast<uint64_t>(scalar.i8);
        case GgufValueType::UINT16: return scalar.u16;
        case GgufValueType::INT16: return static_cast<uint64_t>(scalar.i16);
        case GgufValueType::UINT32: return scalar.u32;
        case GgufValueType::INT32: return static_cast<uint64_t>(scalar.i32);
        case GgufValueType::UINT64: return scalar.u64;
        case GgufValueType::INT64: return static_cast<uint64_t>(scalar.i64);
        case GgufValueType::BOOL: return scalar.b ? 1u : 0u;
        case GgufValueType::FLOAT32: return static_cast<uint64_t>(scalar.f32);
        case GgufValueType::FLOAT64: return static_cast<uint64_t>(scalar.f64);
        default: throw std::runtime_error("GGUF: value is not numeric");
    }
}

int64_t GgufValue::as_i64() const { return static_cast<int64_t>(as_u64()); }
double GgufValue::as_f64() const {
    if (type == GgufValueType::FLOAT32) return scalar.f32;
    if (type == GgufValueType::FLOAT64) return scalar.f64;
    return static_cast<double>(as_u64());
}
bool GgufValue::as_bool() const {
    if (type == GgufValueType::BOOL) return scalar.b;
    return as_u64() != 0;
}
const std::string& GgufValue::as_str() const {
    if (type != GgufValueType::STRING) throw std::runtime_error("GGUF: value is not a string");
    return str;
}

uint64_t TensorInfo::n_elements() const {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
}

uint64_t TensorInfo::row_bytes() const {
    if (dims.empty()) return 0;
    return dtype_row_bytes(type, dims[0]);
}

std::shared_ptr<GgufFile> GgufFile::open(const std::string& path) {
    auto f = std::shared_ptr<GgufFile>(new GgufFile());
    f->path_ = path;
    f->parse();
    return f;
}

void GgufFile::parse() {
    std::ifstream in(path_, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("GGUF: cannot open " + path_);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());

    // The header (metadata + tensor infos) is small relative to tensor data, but a
    // 300B MoE has tens of thousands of tensor infos; read a generous prefix.
    const uint64_t prefix = std::min<uint64_t>(file_size, 512ull << 20);
    header_.resize(static_cast<size_t>(prefix));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(header_.data()), static_cast<std::streamsize>(prefix));
    if (in.gcount() != static_cast<std::streamsize>(prefix))
        throw std::runtime_error("GGUF: short read of header");

    Cursor c{header_.data(), header_.data() + header_.size()};
    uint32_t magic = c.read<uint32_t>();
    if (magic != kGgufMagic) throw std::runtime_error("GGUF: bad magic");
    version_ = c.read<uint32_t>();
    if (version_ < 2 || version_ > 3) throw std::runtime_error("GGUF: unsupported version");

    uint64_t tensor_count = c.read<uint64_t>();
    uint64_t kv_count = c.read<uint64_t>();

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.read_str();
        auto vtype = static_cast<GgufValueType>(c.read<uint32_t>());
        GgufValue val = read_value(c, vtype);
        if (key == "general.alignment") alignment = static_cast<uint32_t>(val.as_u64());
        metadata_.emplace(std::move(key), std::move(val));
    }

    tensors_.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = c.read_str();
        uint32_t n_dims = c.read<uint32_t>();
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.dims[d] = c.read<uint64_t>();
        t.type = static_cast<DType>(c.read<uint32_t>());
        t.offset = c.read<uint64_t>();
        t.shard = 0;
        tensor_index_.emplace(t.name, tensors_.size());
        tensors_.push_back(std::move(t));
    }

    // Tensor data starts at the next `alignment` boundary after the parsed header.
    const uint64_t consumed = static_cast<uint64_t>(c.p - header_.data());
    data_offset_ = (consumed + alignment - 1) / alignment * alignment;

    stream_.open(path_, std::ios::binary);
    if (!stream_) throw std::runtime_error("GGUF: cannot reopen " + path_);
}

bool GgufFile::has(const std::string& key) const { return metadata_.count(key) != 0; }

const GgufValue* GgufFile::find(const std::string& key) const {
    auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

std::vector<std::string> GgufFile::keys() const {
    std::vector<std::string> out;
    out.reserve(metadata_.size());
    for (const auto& [k, v] : metadata_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

const GgufValue& GgufFile::get(const std::string& key) const {
    const GgufValue* v = find(key);
    if (!v) throw std::runtime_error("GGUF: missing metadata key " + key);
    return *v;
}

const TensorInfo* GgufFile::find_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

void GgufFile::set_shard(int shard) {
    shard_ = shard;
    for (auto& t : tensors_) t.shard = shard;
}

void GgufFile::ensure_mapped() const {
    if (map_view_) return;
#ifdef _WIN32
    void* fh = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return;
    void* mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        return;
    }
    void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mh);
        CloseHandle(fh);
        return;
    }
    map_file_ = fh;
    map_view_ = view;
#endif
}

void GgufFile::read_at(uint64_t offset, size_t bytes, void* dst) const {
    ensure_mapped();
    if (map_view_) {
        std::memcpy(dst, static_cast<const uint8_t*>(map_view_) + data_offset_ + offset, bytes);
        return;
    }
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(data_offset_ + offset));
    stream_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (stream_.gcount() != static_cast<std::streamsize>(bytes))
        throw std::runtime_error("GGUF: short read of tensor data in " + path_);
}

void GgufFile::read_tensor(const TensorInfo& t, void* dst) const {
    if (t.shard != shard_)
        throw std::runtime_error("GGUF: tensor belongs to a different shard");
    const uint64_t nrows = t.n_elements() / (t.dims.empty() ? 1 : t.dims[0]);
    read_at(t.offset, static_cast<size_t>(t.row_bytes() * nrows), dst);
}

void GgufModel::load(const std::vector<std::string>& paths) {
    files_.clear();
    tensor_map_.clear();
    tensor_names_.clear();
    for (size_t i = 0; i < paths.size(); ++i) {
        auto f = GgufFile::open(paths[i]);
        f->set_shard(static_cast<int>(i));
        files_.push_back(std::move(f));
    }
    for (const auto& f : files_) {
        for (const auto& t : f->tensors()) {
            if (tensor_map_.emplace(t.name, &t).second) tensor_names_.push_back(t.name);
        }
    }
}

const GgufValue* GgufModel::find(const std::string& key) const {
    for (const auto& f : files_) {
        if (const GgufValue* v = f->find(key)) return v;
    }
    return nullptr;
}

std::vector<std::string> GgufModel::keys() const {
    return files_.empty() ? std::vector<std::string>{} : files_.front()->keys();
}

const TensorInfo* GgufModel::find_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    return it == tensor_map_.end() ? nullptr : it->second;
}

void GgufModel::read_tensor(const TensorInfo& t, void* dst) const {
    files_.at(static_cast<size_t>(t.shard))->read_tensor(t, dst);
}

std::vector<std::string> GgufModel::discover_shards(const std::string& any_shard) {
    namespace fs = std::filesystem;
    fs::path p(any_shard);
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();

    const size_t ofpos = stem.rfind("-of-");
    if (ofpos == std::string::npos) return {any_shard};
    const size_t dash = stem.rfind('-', ofpos - 1);
    if (dash == std::string::npos) return {any_shard};

    const std::string shard_str = stem.substr(dash + 1, ofpos - dash - 1);  // "00002"
    const std::string total_str = stem.substr(ofpos + 4);                   // "00004"
    const std::string prefix = stem.substr(0, dash);                        // "..-UD-IQ3_XXS"
    const int width = static_cast<int>(shard_str.size());
    const int total = std::stoi(total_str);

    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(total));
    for (int i = 1; i <= total; ++i) {
        char num[32];
        std::snprintf(num, sizeof(num), "%0*d", width, i);
        fs::path cand = p.parent_path() / (prefix + "-" + num + "-of-" + total_str + ext);
        out.push_back(cand.string());
    }
    return out;
}

}  // namespace glm5
