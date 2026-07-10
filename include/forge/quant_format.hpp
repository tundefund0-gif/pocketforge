#ifndef FORGE_QUANT_FORMAT_HPP
#define FORGE_QUANT_FORMAT_HPP

#include "types.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>

namespace forge {

// ============================================================
//  .squeeze file format
// ============================================================
// [Header]
// [WeightBlock index]
// [Compressed weight data blocks]
//
// Header (256 bytes):
//   char     magic[8]      = "FORGE\0\0\0"
//   uint32_t version
//   uint32_t n_blocks
//   ModelConfig config
//   uint64_t index_offset
//   uint64_t index_size
//   uint8_t  reserved[168]
// ============================================================

#pragma pack(push, 1)
struct SqueezeHeader {
    char     magic[8]       = {'F','O','R','G','E',0,0,0};
    uint32_t version        = 1;
    uint32_t n_blocks       = 0;
    // config follows
    uint32_t cfg_n_layers   = 0;
    uint32_t cfg_n_embd     = 0;
    uint32_t cfg_n_heads    = 0;
    uint32_t cfg_n_kv_heads = 0;
    uint32_t cfg_n_ff       = 0;
    uint32_t cfg_n_vocab    = 0;
    uint32_t cfg_max_seq_len = 0;
    uint32_t cfg_mtp_heads  = 0;
    uint64_t index_offset   = 0;
    uint64_t index_size     = 0;
    uint8_t  reserved[192]  = {0};
};
static_assert(sizeof(SqueezeHeader) == 256, "SqueezeHeader must be 256 bytes");
#pragma pack(pop)

// ============================================================
//  Quantizer: converts raw FP weights → compressed squeeze format
// ============================================================

class Quantizer {
public:
    Quantizer(const ModelConfig& cfg);

    // Analyze weight importance and assign bit width per matrix
    void analyze_importance(const float* weights, size_t n);

    // Quantize a weight matrix to target bit width
    std::vector<uint8_t> quantize_matrix(
        const float* data,
        uint32_t rows,
        uint32_t cols,
        uint8_t quant_type
    ) const;

    // Compress quantized data with zstd
    std::vector<uint8_t> compress_block(const std::vector<uint8_t>& data) const;

    // Write full model to .squeeze file
    bool write_squeeze(
        const std::string& path,
        const std::vector<WeightBlock>& blocks,
        const std::vector<std::vector<uint8_t>>& compressed_data,
        const ModelConfig& cfg
    );

    // Determine per-matrix bit allocation
    uint8_t select_quant_type(float importance_score) const;

private:
    ModelConfig cfg_;
    std::vector<float> importance_scores_;
};

// ============================================================
//  Weight loader: reads blocks from .squeeze at runtime
// ============================================================

class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    // Open a .squeeze file (mmap'd)
    bool open(const std::string& path);

    // Get model config from header
    ModelConfig config() const;

    // Decompress a specific weight block
    std::vector<int8_t> load_block(
        uint32_t layer_id,
        uint32_t matrix_id,
        float* scale_out = nullptr
    );

    // Load all blocks for a layer (returns pointers into decompression buffer)
    bool prefetch_layer(uint32_t layer_id);

    // Get number of blocks
    uint32_t num_blocks() const { return n_blocks_; }

    // Close
    void close();

private:
    int fd_ = -1;
    void* mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;
    SqueezeHeader header_;
    std::vector<WeightBlock> index_;
    ModelConfig config_;
    uint32_t n_blocks_ = 0;

    // Decompression scratch buffer
    std::vector<uint8_t> scratch_;
};

} // namespace forge
#endif // FORGE_QUANT_FORMAT_HPP
