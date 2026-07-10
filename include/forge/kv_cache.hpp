#ifndef FORGE_KV_CACHE_HPP
#define FORGE_KV_CACHE_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace forge {

// ============================================================
//  3-Tier KV Cache for 131K Context
// ============================================================
//
// Tier 1 — Sink:         first 4 tokens (never evicted, int8)
// Tier 2 — Window:       last 8K tokens (sliding, int8)
// Tier 3 — Pooled Hist:  between sink & window, averaged in blocks of 64
//
// Attention computes scores against ALL three tiers.
// Memory: ~120 MB for 131K context → well under 250 MB
//

struct KVCacheConfig {
    uint32_t n_layers       = 24;
    uint32_t n_kv_heads     = 4;
    uint32_t head_dim       = 64;
    uint32_t sink_size      = 4;
    uint32_t window_size    = 8192;
    uint32_t pool_block     = 64;
    uint32_t max_context    = 131072;
};

class KVCache {
public:
    explicit KVCache(const KVCacheConfig& cfg);

    void store(uint32_t layer, uint32_t head, uint32_t pos,
               const float* key, const float* value);

    // Returns number of scores written
    uint32_t attention_scores(
        uint32_t layer, uint32_t head,
        const float* query,
        float* scores_out
    );

    void attention_weighted_sum(
        uint32_t layer, uint32_t head,
        const float* scores,
        float* output,
        uint32_t count
    );

    uint32_t size() const { return total_stored_; }
    void clear();
    size_t memory_usage() const;

private:
    struct HeadCache {
        // Sink
        std::vector<int8_t> sink_k;
        std::vector<int8_t> sink_v;
        std::vector<float>  sink_k_scale;
        std::vector<float>  sink_v_scale;
        uint32_t sink_count = 0;

        // Window (circular)
        std::vector<int8_t> win_k;
        std::vector<int8_t> win_v;
        std::vector<float>  win_k_scale;
        std::vector<float>  win_v_scale;
        std::vector<uint32_t> win_pos;
        uint32_t win_head = 0;
        uint32_t win_count = 0;

        // Pooled history
        std::vector<int8_t> pool_k;
        std::vector<int8_t> pool_v;
        std::vector<float>  pool_k_scale;
        std::vector<float>  pool_v_scale;
        std::vector<uint32_t> pool_start;
        std::vector<uint32_t> pool_end;
        uint32_t pool_count = 0;

        // Accumulator for current pool block
        std::vector<double> acc_k;
        std::vector<double> acc_v;
        uint32_t acc_count = 0;
        uint32_t acc_start = 0;
        uint32_t head_dim = 0;

        void init(uint32_t hd, uint32_t ws, uint32_t ps);
        void store_sink(const float* key, const float* value, uint32_t pos,
                        uint32_t sink_sz);
        void store_window(const float* key, const float* value, uint32_t pos,
                          uint32_t ws);
        void evict_and_pool(uint32_t pool_block);
        void finalize_pool_block();
    };

    KVCacheConfig cfg_;
    uint32_t window_sz_;
    uint32_t pool_block_;
    uint32_t total_stored_ = 0;
    std::vector<std::vector<HeadCache>> layers_;

    void quantize_row(const float* src, int8_t* dst, float* scale_out, uint32_t n) const;
    float dot_quant(const float* q, const int8_t* kv, float s, uint32_t n) const;
    float dot_product_quantized(const float* q, const int8_t* kv, float s, uint32_t n) const;
};

} // namespace forge
#endif
