#ifndef FORGE_TYPES_HPP
#define FORGE_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace forge {

// ============================================================
//  Quantized types
// ============================================================

struct Q8_0 {
    float d;
    int8_t q[32];
};

struct Q4_0 {
    float d;
    uint8_t q[16];
};

struct Q2_0 {
    float d;
    uint8_t q[8];
};

struct Q1_5 {
    uint8_t q[16];
};

struct SamplingConfig {
    float temperature  = 0.8f;
    float top_p        = 0.95f;
    uint32_t top_k     = 40;
    bool   greedy      = false;
};

struct GenerationConfig {
    SamplingConfig sampling;
    uint32_t max_new_tokens  = 256;
    uint32_t max_context     = 131072;
    bool     mtp_enabled     = true;
    bool     stream          = false;
    float    skip_threshold  = 0.01f;
};

struct WeightBlock {
    uint32_t layer_id;
    uint32_t matrix_id;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t compressed_size;
    uint32_t original_size;
    uint8_t  quant_type;
    uint32_t offset;
};

// ============================================================
//  Model configuration — SANE DEFAULTS for 250MB budget
// ============================================================

struct ModelConfig {
    // Architecture (1B-scale)
    uint32_t n_layers       = 24;
    uint32_t n_embd         = 2048;
    uint32_t n_heads        = 32;
    uint32_t n_kv_heads     = 4;    // GQA
    uint32_t n_ff           = 8192;
    uint32_t n_vocab        = 32000;

    // Memory-safe defaults
    uint32_t max_seq_len    = 131072; // 131K context
    bool     mtp_enabled    = true;
    uint32_t mtp_heads      = 4;

    // Layer skip
    float    skip_threshold = 0.01f;
    uint32_t skip_interval  = 4;

    // KV cache: slide window size (tokens kept)
    uint32_t kv_cache_size  = 8192;   // sliding window size

    // Hard memory cap
        size_t   max_memory     = 250 * 1024 * 1024; // 250 MB
    bool     use_bpe        = false;  // false = char-level ASCII fallback

    // Derived
    uint32_t head_dim()    const { return n_embd / n_heads; }
    uint32_t kv_head_dim() const { return n_embd / n_kv_heads; }

    // Memory budget estimate: returns OK or prints warning
    bool check_memory_budget() const;
};

// ============================================================
//  Generation result
// ============================================================

struct GenerationResult {
    std::vector<int32_t> tokens;
    float                tokens_per_second = 0.0f;
    size_t               peak_memory_bytes = 0;
    int                  mtp_accepted      = 0;
    int                  layers_skipped    = 0;
        bool                 oom_protected     = false; // set if we hit memory cap
    int                  mtp_rejected      = 0;
};

using TokenCallback = std::function<void(int32_t token, float prob)>;
using ProgressCallback = std::function<void(float progress)>;

} // namespace forge
#endif // FORGE_TYPES_HPP
