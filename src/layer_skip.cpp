#include "forge/layer_skip.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace forge {

LayerSkipController::LayerSkipController(const ModelConfig& cfg)
    : cfg_(cfg)
    , threshold_(cfg.skip_threshold)
    , interval_(cfg.skip_interval)
{
    cached_hidden_.resize(cfg.n_layers);
    cached_attn_out_.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        cached_hidden_[i].resize(cfg.n_embd, 0.0f);
        cached_attn_out_[i].resize(cfg.n_embd, 0.0f);
    }
    skip_mask_.resize(cfg.n_layers, false);
}

bool LayerSkipController::should_skip_ffn(
    uint32_t layer_id,
    const float* hidden_after_attn,
    uint32_t n_embd
) {
    if (layer_id >= cfg_.n_layers) return false;

    // Anchor token: always compute fully
    if (is_anchor_token()) {
        skip_mask_[layer_id] = false;
        // Record attention output for future comparison
        std::memcpy(cached_attn_out_[layer_id].data(), hidden_after_attn,
                    n_embd * sizeof(float));
        return false;
    }

    // Compare with previous token's attention output at this layer
    float delta = hidden_delta(
        hidden_after_attn,
        cached_attn_out_[layer_id].data(),
        n_embd
    );

    // Skip middle layers more aggressively
    uint32_t mid = cfg_.n_layers / 2;
    float adjusted_threshold = threshold_;
    if (layer_id > mid / 2 && layer_id < mid + mid / 2) {
        adjusted_threshold *= 2.0f; // more aggressive skip in middle
    }

    // Don't skip first 2 or last 2 layers
    if (layer_id < 2 || layer_id >= cfg_.n_layers - 2) {
        adjusted_threshold = -1.0f; // never skip
    }

    bool skip = (delta < adjusted_threshold);
    skip_mask_[layer_id] = skip;

    if (!skip) {
        // Update cache if we actually computed FFN
        std::memcpy(cached_attn_out_[layer_id].data(), hidden_after_attn,
                    n_embd * sizeof(float));
    }

    return skip;
}

void LayerSkipController::record_layer_output(
    uint32_t layer_id,
    const float* hidden_after_ffn,
    uint32_t n_embd
) {
    if (layer_id >= cfg_.n_layers) return;

    // Only update if FFN was actually computed (not skipped)
    if (!skip_mask_[layer_id]) {
        std::memcpy(cached_hidden_[layer_id].data(), hidden_after_ffn,
                    n_embd * sizeof(float));
    }
}

void LayerSkipController::new_token() {
    tokens_since_anchor_ = (tokens_since_anchor_ + 1) % (interval_ + 1);
    // Reset skip mask
    std::fill(skip_mask_.begin(), skip_mask_.end(), false);
}

void LayerSkipController::reset() {
    skipped_count_ = 0;
    tokens_since_anchor_ = 0;
    std::fill(skip_mask_.begin(), skip_mask_.end(), false);
    for (auto& h : cached_hidden_) {
        std::fill(h.begin(), h.end(), 0.0f);
    }
    for (auto& h : cached_attn_out_) {
        std::fill(h.begin(), h.end(), 0.0f);
    }
}

void LayerSkipController::reset_anchor() {
    tokens_since_anchor_ = 0;
}

float LayerSkipController::hidden_delta(
    const float* a, const float* b, uint32_t n
) const {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum / n);
}

} // namespace forge
