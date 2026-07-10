#ifndef FORGE_GGUF_READER_HPP
#define FORGE_GGUF_READER_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace forge {

// ============================================================
//  GGUF format constants
// ============================================================
constexpr uint32_t GGUF_MAGIC = 0x46554747u; // "GGUF" little-endian

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
};

// ============================================================
//  GGUF tensor info
// ============================================================
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims; // [0] = fastest-changing
    GGMLType type;
    uint64_t offset; // from start of file
    uint64_t n_elems; // total elements

    uint64_t element_size() const;
    bool is_quantized() const;
    uint32_t block_size() const; // for quantized types, elements per block
    uint32_t block_bytes() const; // bytes per block
};

// ============================================================
//  GGUF Reader
// ============================================================
class GGUFFile {
public:
    GGUFFile();
    ~GGUFFile();

    // Open a GGUF file (memory-mapped)
    bool open(const std::string& path);

    // Get metadata string value by key
    std::string metadata_str(const std::string& key) const;

    // Get metadata int value by key
    int64_t metadata_int(const std::string& key, int64_t default_val = 0) const;

    // Get metadata float value by key
    float metadata_float(const std::string& key, float default_val = 0.0f) const;

    // Get tensor info by name
    const GGUFTensorInfo* tensor_info(const std::string& name) const;

    // Read a tensor's raw data into a buffer
    // Returns the number of bytes read, or 0 on failure
    size_t read_tensor_data(const std::string& name, std::vector<uint8_t>& buffer) const;

    // Dequantize a tensor to float32
    std::vector<float> dequantize_tensor(const std::string& name) const;

    // List all tensor names
    std::vector<std::string> list_tensors() const;

    // Get token string by id (for decoding)
    std::string token_str(int32_t id) const {
        if (id >= 0 && id < (int32_t)token_strings_.size()) return token_strings_[id];
        return "";
    }
    bool has_token_strings() const { return !token_strings_.empty(); }
    size_t token_strings_size() const { return token_strings_.size(); }
    const std::vector<std::string>& get_token_strings() const { return token_strings_; }
    const std::vector<std::string>& get_merge_strings() const { return merge_strings_; }

    // Populate ModelConfig from GGUF metadata
    ModelConfig read_config() const;

    // Close
    void close();

    bool is_open() const { return fd_ >= 0; }

private:
    int fd_ = -1;
    void* mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    // Parsed data
    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0;
    uint64_t n_metadata_ = 0;
    std::unordered_map<std::string, std::string> metadata_str_;
    std::unordered_map<std::string, int64_t> metadata_int_;
    std::unordered_map<std::string, float> metadata_float_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;
    std::vector<std::string> token_strings_;  // tokenizer.ggml.tokens
    std::vector<std::string> merge_strings_;  // tokenizer.ggml.merges

    // Parsing helpers
    // memcpy-based read (safe on ARM32 where unaligned access causes SIGBUS)
    template<typename T>
    T read_at(size_t offset, const T& default_val = T{}) const {
        T val = default_val;
        if (offset + sizeof(T) <= mmap_size_) {
            std::memcpy(&val, (uint8_t*)mmap_addr_ + offset, sizeof(T));
        }
        return val;
    }

    template<typename T>
    const T* ptr_at(size_t offset) const {
        if (offset + sizeof(T) > mmap_size_) return nullptr;
        return reinterpret_cast<const T*>((uint8_t*)mmap_addr_ + offset);
    }

    std::string read_string(size_t& offset) const;
    bool skip_value(size_t& offset, GGUFValueType type) const;
    uint64_t type_size(GGUFValueType type) const;
};

} // namespace forge
#endif // FORGE_GGUF_READER_HPP
