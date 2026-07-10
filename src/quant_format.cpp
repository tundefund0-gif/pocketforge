#include "forge/matmul_neon.hpp"
#include "forge/quant_format.hpp"
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <zstd.h>
#include <cmath>
#include <algorithm>

namespace forge {

// ============================================================
//  Quantizer
// ============================================================

Quantizer::Quantizer(const ModelConfig& cfg) : cfg_(cfg) {}

void Quantizer::analyze_importance(const float* weights, size_t n) {
    // For each weight matrix, compute importance as:
    // sum of abs values / sqrt(n)
    // Higher = more important → allocate more bits
    importance_scores_.push_back(0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += std::abs(weights[i]);
    }
    importance_scores_.back() = sum / std::sqrt((float)n);
}

uint8_t Quantizer::select_quant_type(float score) const {
    // Score thresholds determine bit allocation
    // Top 20% → Q4 (4-bit)
    // Middle 70% → Q2 (2-bit)
    // Bottom 10% → Q1.5 (ternary)
    
    // Normalize: we use relative thresholds
    if (score > 0.8f) return 1;  // Q4
    if (score > 0.3f) return 2;  // Q2
    return 3;  // Q1.5 (ternary)
}

std::vector<uint8_t> Quantizer::quantize_matrix(
    const float* data, uint32_t rows, uint32_t cols, uint8_t quant_type
) const {
    uint32_t n = rows * cols;
    std::vector<uint8_t> result;

    switch (quant_type) {
    case 0: { // Q8
        // Store as int8 with per-block scale
        // Blocks of 32
        uint32_t n_blocks = (n + 31) / 32;
        result.resize(n_blocks * (sizeof(float) + 32));
        uint8_t* ptr = result.data();
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t start = b * 32;
            uint32_t end = std::min(start + 32, n);
            // Find max absolute value
            float amax = 0.0f;
            for (uint32_t i = start; i < end; i++) {
                amax = std::max(amax, std::abs(data[i]));
            }
            float scale = (amax == 0.0f) ? 0.0f : 127.0f / amax;
            std::memcpy(ptr, &scale, sizeof(float)); ptr += sizeof(float);
            for (uint32_t i = start; i < end; i++) {
                *ptr++ = (uint8_t)(int8_t)std::round(data[i] * scale + 0.5f);
            }
            // Pad if needed
            for (uint32_t i = end; i < start + 32; i++) {
                *ptr++ = 0;
            }
        }
        break;
    }
    case 1: { // Q4
        uint32_t n_blocks = (n + 31) / 32;
        result.resize(n_blocks * (sizeof(float) + 16));
        uint8_t* ptr = result.data();
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t start = b * 32;
            uint32_t end = std::min(start + 32, n);
            float amax = 0.0f;
            for (uint32_t i = start; i < end; i++) {
                amax = std::max(amax, std::abs(data[i]));
            }
            float scale = (amax == 0.0f) ? 0.0f : 7.0f / amax;
            std::memcpy(ptr, &scale, sizeof(float)); ptr += sizeof(float);
            for (uint32_t i = start; i < end; i += 2) {
                int8_t v0 = (int8_t)std::round(data[i] * scale);
                v0 = std::max((int8_t)-8, std::min((int8_t)7, v0));
                uint8_t packed = (uint8_t)(v0 + 8); // offset to [0, 15]
                if (i + 1 < end) {
                    int8_t v1 = (int8_t)std::round(data[i + 1] * scale);
                    v1 = std::max((int8_t)-8, std::min((int8_t)7, v1));
                    packed |= ((uint8_t)(v1 + 8)) << 4;
                }
                *ptr++ = packed;
            }
        }
        break;
    }
    case 2: { // Q2
        uint32_t n_blocks = (n + 31) / 32;
        result.resize(n_blocks * (sizeof(float) + 8));
        uint8_t* ptr = result.data();
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t start = b * 32;
            uint32_t end = std::min(start + 32, n);
            float amax = 0.0f;
            for (uint32_t i = start; i < end; i++) {
                amax = std::max(amax, std::abs(data[i]));
            }
            float scale = (amax == 0.0f) ? 0.0f : 1.0f / amax;
            std::memcpy(ptr, &scale, sizeof(float)); ptr += sizeof(float);
            for (uint32_t i = start; i < end; i += 4) {
                uint8_t packed = 0;
                for (uint32_t j = 0; j < 4 && i + j < end; j++) {
                    int8_t v = (int8_t)std::round(data[i + j] * scale);
                    v = std::max((int8_t)-2, std::min((int8_t)1, v));
                    packed |= (uint8_t)(v + 2) << (j * 2);
                }
                *ptr++ = packed;
            }
        }
        break;
    }
    case 3: { // Q1.5 (ternary: -1, 0, +1)
        uint32_t n_blocks = (n + 31) / 32;
        result.resize(n_blocks * 16);
        uint8_t* ptr = result.data();
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t start = b * 32;
            uint32_t end = std::min(start + 32, n);
            for (uint32_t i = start; i < end; i += 2) {
                // Map to ternary: threshold at 0.35*max
                // -1 if < -0.35*max, +1 if > 0.35*max, else 0
                uint8_t packed = 0;
                if (i < end) {
                    float v = data[i];
                    uint8_t code = (v > 0.35f) ? 2 : ((v < -0.35f) ? 0 : 1);
                    packed |= code;
                }
                if (i + 1 < end) {
                    float v = data[i + 1];
                    uint8_t code = (v > 0.35f) ? 2 : ((v < -0.35f) ? 0 : 1);
                    packed |= code << 2;
                }
                *ptr++ = packed;
            }
        }
        break;
    }
    }
    return result;
}

std::vector<uint8_t> Quantizer::compress_block(const std::vector<uint8_t>& data) const {
    size_t max_dst = ZSTD_compressBound(data.size());
    std::vector<uint8_t> compressed(max_dst);
    size_t result = ZSTD_compress(compressed.data(), max_dst,
                                   data.data(), data.size(), 3);
    if (ZSTD_isError(result)) {
        return data; // fallback: store uncompressed
    }
    compressed.resize(result);
    return compressed;
}

bool Quantizer::write_squeeze(
    const std::string& path,
    const std::vector<WeightBlock>& blocks,
    const std::vector<std::vector<uint8_t>>& compressed_data,
    const ModelConfig& cfg
) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    // Prepare header
    SqueezeHeader header;
    header.n_blocks = (uint32_t)blocks.size();
    header.cfg_n_layers = cfg.n_layers;
    header.cfg_n_embd = cfg.n_embd;
    header.cfg_n_heads = cfg.n_heads;
    header.cfg_n_kv_heads = cfg.n_kv_heads;
    header.cfg_n_ff = cfg.n_ff;
    header.cfg_n_vocab = cfg.n_vocab;
    header.cfg_max_seq_len = cfg.max_seq_len;
    header.cfg_mtp_heads = cfg.mtp_heads;

    // Write header
    fwrite(&header, sizeof(header), 1, f);

    // Write compressed data blocks, collect index
    std::vector<WeightBlock> written_blocks = blocks;
    uint64_t data_offset = sizeof(header);
    
    for (uint32_t i = 0; i < blocks.size(); i++) {
        written_blocks[i].offset = data_offset;
        written_blocks[i].compressed_size = (uint32_t)compressed_data[i].size();
        
        fwrite(compressed_data[i].data(), 1, compressed_data[i].size(), f);
        data_offset += compressed_data[i].size();
    }

    // Write index at end
    header.index_offset = data_offset;
    header.index_size = blocks.size() * sizeof(WeightBlock);
    
    fwrite(written_blocks.data(), sizeof(WeightBlock), blocks.size(), f);

    // Rewrite header with updated index info
    fseek(f, 0, SEEK_SET);
    fwrite(&header, sizeof(header), 1, f);

    fclose(f);
    return true;
}

// ============================================================
//  WeightLoader
// ============================================================

WeightLoader::WeightLoader() {}

WeightLoader::~WeightLoader() {
    close();
}

bool WeightLoader::open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    struct stat st;
    if (fstat(fd_, &st) < 0) { close(); return false; }
    mmap_size_ = st.st_size;

    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) { close(); return false; }

    // Read header
    std::memcpy(&header_, mmap_addr_, sizeof(SqueezeHeader));

    // Validate magic
    if (std::memcmp(header_.magic, "FORGE", 5) != 0) {
        close();
        return false;
    }

    // Read index
    n_blocks_ = header_.n_blocks;
    index_.resize(n_blocks_);
    std::memcpy(index_.data(),
                (uint8_t*)mmap_addr_ + header_.index_offset,
                n_blocks_ * sizeof(WeightBlock));

    // Populate config
    config_.n_layers = header_.cfg_n_layers;
    config_.n_embd = header_.cfg_n_embd;
    config_.n_heads = header_.cfg_n_heads;
    config_.n_kv_heads = header_.cfg_n_kv_heads;
    config_.n_ff = header_.cfg_n_ff;
    config_.n_vocab = header_.cfg_n_vocab;
    config_.max_seq_len = header_.cfg_max_seq_len;
    config_.mtp_heads = header_.cfg_mtp_heads;

    // Pre-allocate scratch buffer for decompression (largest possible block)
    scratch_.resize(1024 * 1024); // 1 MB scratch

    return true;
}

ModelConfig WeightLoader::config() const {
    return config_;
}

std::vector<int8_t> WeightLoader::load_block(uint32_t layer_id, uint32_t matrix_id,
                                               float* scale_out)
{
    // Find block in index
    for (const auto& blk : index_) {
        if (blk.layer_id == layer_id && blk.matrix_id == matrix_id) {
            // Block found
            const uint8_t* compressed = (const uint8_t*)mmap_addr_ + blk.offset;
            size_t compressed_size = blk.compressed_size;

            // Decompress
            size_t dst_size = blk.original_size;
            std::vector<uint8_t> decompressed(dst_size);

            size_t result = ZSTD_decompress(
                decompressed.data(), dst_size,
                compressed, compressed_size
            );

            if (ZSTD_isError(result)) {
                return {}; // decompression failed
            }

            // Convert from quantized format to int8
            std::vector<int8_t> result_data(blk.n_rows * blk.n_cols);
            float scale = 1.0f;

            switch (blk.quant_type) {
            case 0: { // Q8
                const uint8_t* ptr = decompressed.data();
                uint32_t n = blk.n_rows * blk.n_cols;
                uint32_t n_blocks = (n + 31) / 32;
                for (uint32_t b = 0; b < n_blocks; b++) {
                    float blk_scale;
                    std::memcpy(&blk_scale, ptr, sizeof(float)); ptr += sizeof(float);
                    uint32_t start = b * 32;
                    uint32_t end = std::min(start + 32, n);
                    for (uint32_t i = start; i < end; i++) {
                        result_data[i] = (int8_t)(*ptr++);
                    }
                    ptr += (32 - (end - start)); // skip padding
                }
                scale = 1.0f; // weights already int8, scale applied later
                break;
            }
            case 1: { // Q4
                dequant_q4_to_int8(decompressed.data(), result_data.data(), 1.0f,
                                   blk.n_rows * blk.n_cols);
                scale = 0.0f; // TODO: load scale from block
                break;
            }
            case 2: { // Q2
                dequant_q2_to_int8(decompressed.data(), result_data.data(), 1.0f,
                                   blk.n_rows * blk.n_cols);
                break;
            }
            case 3: { // Q1.5 (ternary)
                dequant_ternary_to_int8(decompressed.data(), result_data.data(),
                                       blk.n_rows * blk.n_cols);
                break;
            }
            }

            if (scale_out) *scale_out = scale;
            return result_data;
        }
    }
    return {}; // not found
}

WeightLoader::LoadResult WeightLoader::load_block_with_scales(
    uint32_t layer_id, uint32_t matrix_id)
{
    LoadResult result;
    for (const auto& blk : index_) {
        if (blk.layer_id == layer_id && blk.matrix_id == matrix_id) {
            const uint8_t* compressed = (const uint8_t*)mmap_addr_ + blk.offset;
            size_t compressed_size = blk.compressed_size;
            size_t dst_size = blk.original_size;
            std::vector<uint8_t> decompressed(dst_size);
            size_t dec_result = ZSTD_decompress(
                decompressed.data(), dst_size,
                compressed, compressed_size);
            if (ZSTD_isError(dec_result)) return result;

            result.data.resize(blk.n_rows * blk.n_cols);
            result.global_scale = 1.0f;

            uint32_t n = blk.n_rows * blk.n_cols;

            switch (blk.quant_type) {
            case 0: { // Q8
                uint32_t n_blocks = (n + 31) / 32;
                uint32_t cols = blk.n_cols;
                uint32_t rows = blk.n_rows;
                uint32_t blocks_per_row = (cols + 31) / 32;
                result.row_scales.resize(rows, 0.0f);

                const uint8_t* ptr = decompressed.data();
                for (uint32_t b = 0; b < n_blocks; b++) {
                    float blk_scale;
                    std::memcpy(&blk_scale, ptr, sizeof(float)); ptr += sizeof(float);
                    uint32_t row = b / blocks_per_row;
                    result.row_scales[row] += blk_scale;
                    uint32_t start = b * 32;
                    uint32_t end = std::min(start + 32, n);
                    for (uint32_t i = start; i < end; i++) {
                        result.data[i] = (int8_t)(*ptr++);
                    }
                    // Skip padding
                    for (uint32_t i = end; i < start + 32; i++) ptr++;
                }
                // Average row scales
                float total_scale = 0.0f;
                for (uint32_t r = 0; r < rows; r++) {
                    result.row_scales[r] /= (float)blocks_per_row;
                    total_scale += result.row_scales[r];
                }
                result.global_scale = total_scale / (float)rows;
                break;
            }
            case 1: { // Q4
                dequant_q4_to_int8(decompressed.data(), result.data.data(), 1.0f, n);
                uint32_t rows = blk.n_rows;
                result.row_scales.assign(rows, 1.0f);
                result.global_scale = 1.0f;
                break;
            }
            case 2: { // Q2
                dequant_q2_to_int8(decompressed.data(), result.data.data(), 1.0f, n);
                uint32_t rows = blk.n_rows;
                result.row_scales.assign(rows, 1.0f);
                result.global_scale = 1.0f;
                break;
            }
            case 3: { // Q1.5
                dequant_ternary_to_int8(decompressed.data(), result.data.data(), n);
                uint32_t rows = blk.n_rows;
                result.row_scales.assign(rows, 1.0f);
                result.global_scale = 1.0f;
                break;
            }
            }
            return result;
        }
    }
    return result;
}

bool WeightLoader::prefetch_layer(uint32_t layer_id) {
    // In a more advanced implementation, we'd madvise the pages
    // For now, we just return true (the prefetch pipeline handles async loading)
    return true;
}

void WeightLoader::close() {
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
