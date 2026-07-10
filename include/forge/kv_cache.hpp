#ifndef FORGE_KV_CACHE_HPP
#define FORGE_KV_CACHE_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace forge {

// ============================================================
//  High-Context KV Cache (int8, streaming attention)
// ============================================================
//
// No float32 scratch buffer — attention computed directly on int8 data.
// Supports 16K+ context within 250MB budget.
//
// Memory:
//   24L × 4KV × 16384ctx × 64dim × 2(K+V) × 1B = 192 MB (int8 data)
//   + ~2 MB working = 194 MB total
//

struct KVCacheConfig {
    uint32_t n_layers     = 24;
    uint32_t n_kv_heads   = 4;
    uint32_t head_dim     = 64;
    uint32_t max_positions = 16384;  // 16K context
    uint32_t n_sinks      = 4;       // attention sink tokens
    bool     enable_sliding_window = true;
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& cfg);

    // Store K/V for one position
    void store(uint32_t layer, uint32_t head, uint32_t pos,
               const float* key, const float* value);

    // Compute attention scores for one query head against all stored K
    // query: [head_dim] float  →  scores: [count] float
    void attention_scores(
        uint32_t layer, uint32_t head,
        const float* query,
        float* scores_out,
        uint32_t* count_out
    );

    // Compute weighted sum of values
    // scores: [count] float  →  output: [head_dim] float
    void attention_weighted_sum(
        uint32_t layer, uint32_t head,
        const float* scores,
        float* output,
        uint32_t count
    );

    uint32_t size()       const { return cached_positions_; }
    uint32_t capacity()   const { return max_positions_; }
    void clear();

    size_t memory_usage() const;

private:
    // Per-head storage: two flat arrays for K and V int8 data
    struct HeadStorage {
        std::vector<int8_t> keys;    // [max_pos * head_dim]
        std::vector<int8_t> values;  // [max_pos * head_dim]
        std::vector<float>  k_scales; // per-position
        std::vector<float>  v_scales;
        uint32_t count = 0;
        uint32_t head_dim;

        void resize(uint32_t max_pos, uint32_t hd) {
            head_dim = hd;
            keys.resize(max_pos * hd, 0);
            values.resize(max_pos * hd, 0);
            k_scales.resize(max_pos, 1.0f);
            v_scales.resize(max_pos, 1.0f);
        }
    };

    struct LayerStorage {
        std::vector<HeadStorage> heads;
    };

    KVCacheConfig cfg_;
    uint32_t max_positions_;
    uint32_t cached_positions_ = 0;
    uint32_t n_sinks_;
    std::vector<LayerStorage> layers_;

    // Quantize row to int8
    void quantize_row(const float* src, int8_t* dst, float* scale_out, uint32_t n) const;
    // Single dequantized dot product: query_float × key_int8
    float dot_product_quantized(const float* query, const int8_t* key, float scale, uint32_t n) const;
    // Weighted sum: sum(score_i * value_i_int8) then dequant
    void weighted_sum_quantized(const float* scores, const int8_t* values,
                                const float* v_scales, float* output,
                                uint32_t count, uint32_t head_dim) const;
};

} // namespace forge
#endif // FORGE_KV_CACHE_HPP
