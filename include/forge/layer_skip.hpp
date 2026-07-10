#ifndef FORGE_LAYER_SKIP_HPP
#define FORGE_LAYER_SKIP_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>

namespace forge {

// ============================================================
//  Predictive Layer Skipping
// ============================================================
//
// Observation: consecutive tokens have very similar hidden states
// at middle layers. We can skip the FFN computation (and sometimes
// the attention) when the activation change from previous token
// is below a threshold.
//
// Algorithm:
//   1. Cache hidden state and per-layer outputs from token T
//   2. For token T+1, compute attention (cheap), then before FFN:
//      a. Compute delta = ||hidden_T+1_after_attn - hidden_T_after_attn||
//      b. If delta < threshold, SKIP FFN (use cached output)
//   3. Full recompute every N tokens (anchor tokens)
//
// We track which layers were skipped so the engine can avoid
// loading their FFN weights.
//

class LayerSkipController {
public:
    LayerSkipController(const ModelConfig& cfg);

    // Called after attention, before FFN
    // Returns true if FFN should be SKIPPED for this layer
    bool should_skip_ffn(
        uint32_t layer_id,
        const float* hidden_after_attn,
        uint32_t n_embd
    );

    // Called after full layer compute (for cache update)
    void record_layer_output(
        uint32_t layer_id,
        const float* hidden_after_ffn,
        uint32_t n_embd
    );

    // Called at start of each new token
    void new_token();

    // Get number of skipped layers so far
    uint32_t total_skipped() const { return skipped_count_; }

    // Get skip mask for current token
    const std::vector<bool>& skip_mask() const { return skip_mask_; }

    // Reset
    void reset();
    void reset_anchor();

    // Set threshold (higher = more skipping)
    void set_threshold(float t) { threshold_ = t; }
    float threshold() const { return threshold_; }

    // Check if we should anchor (full recompute)
    bool is_anchor_token() const { return tokens_since_anchor_ >= interval_; }

private:
    ModelConfig cfg_;
    float threshold_;
    uint32_t interval_;

    // Cached hidden states per layer (from previous token or last full compute)
    std::vector<std::vector<float>> cached_hidden_;

    // Per-layer attention output cache
    std::vector<std::vector<float>> cached_attn_out_;

    // Current skip mask
    std::vector<bool> skip_mask_;

    // Statistics
    uint32_t skipped_count_ = 0;
    uint32_t tokens_since_anchor_ = 0;

    // L2 norm of difference
    float hidden_delta(const float* a, const float* b, uint32_t n) const;
};

} // namespace forge
#endif // FORGE_LAYER_SKIP_HPP
