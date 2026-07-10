#include "forge/gguf_reader.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace forge {

// ============================================================
//  GGUFTensorInfo helpers
// ============================================================

uint64_t GGUFTensorInfo::element_size() const {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_I8:   return 1;
        case GGML_TYPE_I16:  return 2;
        case GGML_TYPE_I32:  return 4;
        case GGML_TYPE_Q8_0: return 1; // 1 byte per element (with block metadata)
        case GGML_TYPE_Q4_0: return 1; // 4-bit (use block_size/block_bytes for actual size)
        default: return 4;
    }
}

bool GGUFTensorInfo::is_quantized() const {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
        case GGML_TYPE_BF16:
            return false;
        default:
            return true;
    }
}

uint32_t GGUFTensorInfo::block_size() const {
    switch (type) {
        case GGML_TYPE_Q4_0:    return 32;
        case GGML_TYPE_Q4_1:    return 32;
        case GGML_TYPE_Q5_0:    return 32;
        case GGML_TYPE_Q5_1:    return 32;
        case GGML_TYPE_Q8_0:    return 32;
        case GGML_TYPE_Q8_1:    return 32;
        case GGML_TYPE_Q2_K:    return 256;
        case GGML_TYPE_Q3_K:    return 256;
        case GGML_TYPE_Q4_K:    return 256;
        case GGML_TYPE_Q5_K:    return 256;
        case GGML_TYPE_Q6_K:    return 256;
        case GGML_TYPE_Q8_K:    return 256;
        case GGML_TYPE_IQ2_XXS: return 256;
        case GGML_TYPE_IQ2_XS:  return 256;
        case GGML_TYPE_IQ3_XXS: return 256;
        case GGML_TYPE_IQ1_S:   return 256;
        case GGML_TYPE_IQ4_NL:  return 32;
        case GGML_TYPE_IQ3_S:   return 256;
        case GGML_TYPE_IQ2_S:   return 256;
        case GGML_TYPE_IQ4_XS:  return 256;
        case GGML_TYPE_IQ1_M:   return 256;
        default: return 1;
    }
}

uint32_t GGUFTensorInfo::block_bytes() const {
    switch (type) {
        case GGML_TYPE_Q4_0:    return 18;   // d(float16) + 32*q4_0(16 bytes)
        case GGML_TYPE_Q4_1:    return 20;
        case GGML_TYPE_Q5_0:    return 22;
        case GGML_TYPE_Q5_1:    return 24;
        case GGML_TYPE_Q8_0:    return 34;   // d(float16) + 32*int8(32 bytes)
        case GGML_TYPE_Q8_1:    return 40;
        case GGML_TYPE_Q2_K:    return 0x1C; // 28
        case GGML_TYPE_Q3_K:    return 0x2A; // 42
        case GGML_TYPE_Q4_K:    return 0x30; // 48
        case GGML_TYPE_Q5_K:    return 0x3E; // 62
        case GGML_TYPE_Q6_K:    return 0x4C; // 76
        case GGML_TYPE_Q8_K:    return 0x50; // 80
        default: return 0;
    }
}

// ============================================================
//  GGUFFile
// ============================================================

GGUFFile::GGUFFile() {}
GGUFFile::~GGUFFile() { close(); }

bool GGUFFile::open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "GGUF: Failed to open " << path << "\n";
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) { close(); return false; }
    mmap_size_ = st.st_size;

    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) { close(); return false; }

    // Read header
    size_t offset = 0;
    auto magic = read_at<uint32_t>(offset); offset += sizeof(uint32_t);
    if (magic != GGUF_MAGIC) {
        std::cerr << "GGUF: Invalid magic\n";
        close();
        return false;
    }

    version_ = read_at<uint32_t>(offset); offset += sizeof(uint32_t);

    n_tensors_ = read_at<uint64_t>(offset); offset += sizeof(uint64_t);

    n_metadata_ = read_at<uint64_t>(offset); offset += sizeof(uint64_t);

    // Read metadata KV pairs
    for (uint64_t i = 0; i < n_metadata_; i++) {
        std::string key = read_string(offset);

        GGUFValueType val_type = static_cast<GGUFValueType>(read_at<uint32_t>(offset)); offset += sizeof(uint32_t);

        switch (val_type) {
            case GGUF_TYPE_UINT8: {
                metadata_int_[key] = read_at<uint8_t>(offset); offset += sizeof(uint8_t); break;
            }
            case GGUF_TYPE_INT8: {
                metadata_int_[key] = read_at<int8_t>(offset); offset += sizeof(int8_t); break;
            }
            case GGUF_TYPE_UINT16: {
                metadata_int_[key] = read_at<uint16_t>(offset); offset += sizeof(uint16_t); break;
            }
            case GGUF_TYPE_INT16: {
                metadata_int_[key] = read_at<int16_t>(offset); offset += sizeof(int16_t); break;
            }
            case GGUF_TYPE_UINT32: {
                metadata_int_[key] = read_at<uint32_t>(offset); offset += sizeof(uint32_t); break;
            }
            case GGUF_TYPE_INT32: {
                metadata_int_[key] = read_at<int32_t>(offset); offset += sizeof(int32_t); break;
            }
            case GGUF_TYPE_FLOAT32: {
                metadata_float_[key] = read_at<float>(offset); offset += sizeof(float); break;
            }
            case GGUF_TYPE_BOOL: {
                metadata_int_[key] = read_at<bool>(offset) ? 1 : 0; offset += sizeof(bool); break;
            }
            case GGUF_TYPE_STRING: {
                std::string val = read_string(offset);
                metadata_str_[key] = val;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                GGUFValueType arr_type = static_cast<GGUFValueType>(read_at<uint32_t>(offset)); offset += sizeof(uint32_t);
                uint64_t arr_len = read_at<uint64_t>(offset); offset += sizeof(uint64_t);
                // Store tokenizer.ggml.tokens array for decoding
                bool store_tokens = (key == "tokenizer.ggml.tokens" && arr_type == GGUF_TYPE_STRING);
                bool store_merges = (key == "tokenizer.ggml.merges" && arr_type == GGUF_TYPE_STRING);
                if (store_tokens) {
                    token_strings_.reserve((size_t)arr_len);
                }
                if (store_merges) {
                    merge_strings_.reserve((size_t)arr_len);
                }
                for (uint64_t j = 0; j < arr_len; j++) {
                    if (store_tokens && arr_type == GGUF_TYPE_STRING) {
                        std::string tok = read_string(offset);
                        token_strings_.push_back(tok);
                    } else if (store_merges && arr_type == GGUF_TYPE_STRING) {
                        std::string merge = read_string(offset);
                        merge_strings_.push_back(merge);
                    } else {
                        skip_value(offset, arr_type);
                    }
                }
                break;
            }
            case GGUF_TYPE_UINT64: {
                metadata_int_[key] = (int64_t)read_at<uint64_t>(offset); offset += sizeof(uint64_t); break;
            }
            case GGUF_TYPE_INT64: {
                metadata_int_[key] = read_at<int64_t>(offset); offset += sizeof(int64_t); break;
            }
            case GGUF_TYPE_FLOAT64: {
                metadata_float_[key] = (float)read_at<double>(offset); offset += sizeof(double); break;
            }
            default:
                break;
        }
    }

    // Tensor info starts immediately after metadata (GGUF spec - no alignment padding)

    // Read tensor info
    // uint64_t tensor_data_offset = offset + n_tensors_ * 48;
    for (uint64_t i = 0; i < n_tensors_; i++) {
        GGUFTensorInfo info;
        info.name = read_string(offset);

        info.n_dims = read_at<uint32_t>(offset); offset += sizeof(uint32_t);
        if (info.n_dims == 0 || info.n_dims > 4) {
            std::cerr << "GGUF: Invalid tensor n_dims=" << info.n_dims << " at tensor " << i << "\n";
            close();
            return false;
        }

        // Read dimensions BEFORE type (GGUF v1/v2 compatibility - many models store dims before type)
        info.dims.resize(info.n_dims);
        info.n_elems = 1;
        for (uint32_t d = 0; d < info.n_dims; d++) {
            info.dims[d] = read_at<uint64_t>(offset); offset += sizeof(uint64_t);
            info.n_elems *= info.dims[d];
        }

        info.type = static_cast<GGMLType>(read_at<uint32_t>(offset)); offset += sizeof(uint32_t);

        // Skip the data_offset field (GGUF v1/v2 compatibility - uint64 after type)
        // Some converters use v1-style format even with version=3 in header
        uint64_t data_offset_field = read_at<uint64_t>(offset); offset += sizeof(uint64_t);
        (void)data_offset_field;

        // Store tensor info (offset will be set when we know data start)
        tensors_[info.name] = info;
    }

    // Calculate tensor data offsets
    // GGUF stores tensors sequentially after the index
    uint64_t data_offset = (offset + 31) & ~31;  // Align tensor data to 32 bytes (GGUF spec)
    for (auto& [name, info] : tensors_) {
        info.offset = data_offset;
        // Calculate size based on type
        uint64_t tensor_size = 0;
        if (info.is_quantized()) {
            uint32_t bs = info.block_size();
            uint32_t bb = info.block_bytes();
            uint64_t n_blocks = (info.n_elems + bs - 1) / bs;
            tensor_size = n_blocks * bb;
        } else {
            tensor_size = info.n_elems * info.element_size();
        }
        // Align each tensor to 32 bytes
        tensor_size = (tensor_size + 31) & ~31;
        data_offset += tensor_size;
    }

    return true;
}

std::string GGUFFile::read_string(size_t& offset) const {
    uint64_t len = read_at<uint64_t>(offset); offset += sizeof(uint64_t);
    if (offset + len > mmap_size_ || len > 1048576) return "";
    const char* data_p = (const char*)mmap_addr_ + offset;
    std::string result(data_p, (size_t)len);
    offset += len;
    return result;
}

bool GGUFFile::skip_value(size_t& offset, GGUFValueType type) const {
    switch (type) {
        case GGUF_TYPE_STRING: {
            // Read string length and skip the data
            uint64_t len = read_at<uint64_t>(offset);
            offset += sizeof(uint64_t) + len;
            return true;
        }
        case GGUF_TYPE_ARRAY: {
            // Skip array header then recursively skip elements
            GGUFValueType arr_type = static_cast<GGUFValueType>(read_at<uint32_t>(offset));
            offset += sizeof(uint32_t);
            uint64_t arr_len = read_at<uint64_t>(offset);
            offset += sizeof(uint64_t);
            for (uint64_t j = 0; j < arr_len; j++) {
                skip_value(offset, arr_type);
            }
            return true;
        }
        default: {
            uint64_t sz = type_size(type);
            if (sz == 0) return false;
            offset += sz;
            return true;
        }
    }
}

uint64_t GGUFFile::type_size(GGUFValueType type) const {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_STRING:  return sizeof(uint64_t); // length prefix only
        case GGUF_TYPE_ARRAY:   return sizeof(uint32_t) + sizeof(uint64_t); // type + length
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

std::string GGUFFile::metadata_str(const std::string& key) const {
    auto it = metadata_str_.find(key);
    return it != metadata_str_.end() ? it->second : "";
}

int64_t GGUFFile::metadata_int(const std::string& key, int64_t default_val) const {
    auto it = metadata_int_.find(key);
    return it != metadata_int_.end() ? it->second : default_val;
}

float GGUFFile::metadata_float(const std::string& key, float default_val) const {
    auto it = metadata_float_.find(key);
    return it != metadata_float_.end() ? it->second : default_val;
}

const GGUFTensorInfo* GGUFFile::tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

size_t GGUFFile::read_tensor_data(const std::string& name, std::vector<uint8_t>& buffer) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return 0;

    const auto& info = it->second;
    uint64_t size;
    if (info.is_quantized()) {
        uint64_t n_blocks = (info.n_elems + info.block_size() - 1) / info.block_size();
        size = n_blocks * info.block_bytes();
    } else {
        size = info.n_elems * info.element_size();
    }
    // Aligned
    size = (size + 31) & ~31;

    if (info.offset + size > mmap_size_) {
        std::cerr << "GGUF: Tensor '" << name << "' data out of bounds\n";
        return 0;
    }

    buffer.resize(size);
    std::memcpy(buffer.data(), (uint8_t*)mmap_addr_ + info.offset, size);
    return size;
}

static void dequantize_q8_0_row(const uint8_t* src, float* dst, uint64_t n) {
    // Q8_0 block: float16 d + 32 * int8
    constexpr uint32_t BLK = 32;
    uint64_t n_blocks = (n + BLK - 1) / BLK;
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint16_t d16;
        std::memcpy(&d16, src + b * 34, sizeof(uint16_t));
        // float d = (float)(int16_t)d16 / (float)(1 << 8); // fp16 to fp32
        // Actually GGUF Q8_0 stores float16 scale, let's handle properly
        // For now: read scale as half precision
        float scale;
        {   // Convert from IEEE half (16-bit) to float (32-bit)
            uint16_t h = d16;
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                // Subnormal
                f = (sign << 31) | (0x7F - 15 + 1) << 23 | (mant << 13);
                // Normalize
                while (!(f & 0x7F800000) && mant) { mant <<= 1; exp--; }
            } else if (exp == 0x1F) {
                // NaN or Inf
                f = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                // Normal
                f = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
            }
            std::memcpy(&scale, &f, sizeof(float));
        }
        const int8_t* q = (const int8_t*)(src + b * 34 + 2);
        uint64_t start = b * BLK;
        uint64_t end = std::min(start + BLK, (uint64_t)n);
        for (uint64_t i = start; i < end; i++) {
            dst[i] = (float)q[i - start] * scale;
        }
    }
}

std::vector<float> GGUFFile::dequantize_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return {};

    const auto& info = it->second;
    std::vector<float> result(info.n_elems, 0.0f);

    std::vector<uint8_t> raw;
    size_t n_read = read_tensor_data(name, raw);
    if (n_read == 0) return {};

    switch (info.type) {
        case GGML_TYPE_F32: {
            std::memcpy(result.data(), raw.data(), info.n_elems * sizeof(float));
            break;
        }
        case GGML_TYPE_F16: {
            // Convert fp16 to fp32
            for (uint64_t i = 0; i < info.n_elems; i++) {
                uint16_t h;
                std::memcpy(&h, raw.data() + i * 2, 2);
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    f = (sign << 31) | (0x7F - 15 + 1) << 23 | (mant << 13);
                } else if (exp == 0x1F) {
                    f = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
                }
                float val;
                std::memcpy(&val, &f, sizeof(float));
                result[i] = val;
            }
            break;
        }
        case GGML_TYPE_Q8_0: {
            dequantize_q8_0_row(raw.data(), result.data(), info.n_elems);
            break;
        }
        case GGML_TYPE_Q4_0: {
            // Q4_0 block: float16 d + 32*q4_0 (16 bytes)
            constexpr uint32_t BLK = 32;
            uint64_t n_blocks = (info.n_elems + BLK - 1) / BLK;
            for (uint64_t b = 0; b < n_blocks; b++) {
                uint16_t d16;
                std::memcpy(&d16, raw.data() + b * 18, 2);
                uint32_t sign = (d16 >> 15) & 1;
                uint32_t exp = (d16 >> 10) & 0x1F;
                uint32_t mant = d16 & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    f = (sign << 31) | (0x7F - 15 + 1) << 23 | (mant << 13);
                } else if (exp == 0x1F) {
                    f = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f = (sign << 31) | ((exp + 0x70) << 23) | (mant << 13);
                }
                float scale;
                std::memcpy(&scale, &f, sizeof(float));

                const uint8_t* q = raw.data() + b * 18 + 2;
                uint64_t start = b * BLK;
                uint64_t end = std::min(start + BLK, info.n_elems);
                for (uint64_t i = start; i < end; i++) {
                    int8_t val = (q[(i - start) / 2] >> ((i % 2) ? 4 : 0)) & 0x0F;
                    result[i] = ((float)val - 8.0f) * scale;
                }
            }
            break;
        }
        default: {
            std::cerr << "GGUF: Unsupported tensor type " << info.type
                      << " for '" << name << "'\n";
            return {};
        }
    }
    return result;
}

std::vector<std::string> GGUFFile::list_tensors() const {
    std::vector<std::string> names;
    for (const auto& [name, info] : tensors_) {
        names.push_back(name);
    }
    return names;
}

ModelConfig GGUFFile::read_config() const {
    ModelConfig cfg;

    // Architecture
    std::string arch = metadata_str("general.architecture");
    // Try various metadata key formats
    cfg.n_layers = (uint32_t)metadata_int("llama.block_count",
                   metadata_int("minicpm.block_count",
                   metadata_int("general.block_count", 24)));

    cfg.n_embd = (uint32_t)metadata_int("llama.embedding_length",
                metadata_int("minicpm.embedding_length",
                metadata_int("general.embedding_length", 1536)));

    cfg.n_heads = (uint32_t)metadata_int("llama.attention.head_count",
                 metadata_int("minicpm.attention.head_count",
                 metadata_int("general.attention.head_count", 16)));

    cfg.n_kv_heads = (uint32_t)metadata_int("llama.attention.head_count_kv",
                    metadata_int("minicpm.attention.head_count_kv",
                    metadata_int("general.attention.head_count_kv", 2)));

    cfg.n_ff = (uint32_t)metadata_int("llama.feed_forward_length",
              metadata_int("minicpm.feed_forward_length",
              metadata_int("general.feed_forward_length", 4608)));

    cfg.n_vocab = (uint32_t)metadata_int("llama.vocab_size",
                 metadata_int("minicpm.vocab_size",
                 metadata_int("general.vocab_size", 130560)));

    cfg.max_seq_len = (uint32_t)metadata_int("llama.context_length",
                     metadata_int("minicpm.context_length",
                     metadata_int("general.context_length", 131072)));

    // head_dim - check rope.dimension_count as authoritative source
    int64_t hd_rope = metadata_int("llama.rope.dimension_count",
                      metadata_int("minicpm.rope.dimension_count", 0));
    if (hd_rope > 0) {
        cfg.head_dim_ = (uint32_t)hd_rope;
    } else {
        // Fallback: check explicit head_dim metadata
        int64_t hd_explicit = metadata_int("llama.attention.head_dim",
                             metadata_int("minicpm.attention.head_dim", 0));
        if (hd_explicit > 0) {
            cfg.head_dim_ = (uint32_t)hd_explicit;
        }
    }

    // RoPE
    float rope_theta = metadata_float("llama.rope.freq_base", 5000000.0f);
    (void)rope_theta;

    // Norm epsilon
    float norm_eps = metadata_float("llama.attention.layer_norm_rms_epsilon", 1e-6f);
    (void)norm_eps;

    cfg.mtp_heads = 4; // default for our implementation
    cfg.kv_cache_size = std::min((uint32_t)8192, cfg.max_seq_len);

    return cfg;
}

void GGUFFile::close() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
        mmap_addr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

} // namespace forge
