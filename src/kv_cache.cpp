#include "forge/kv_cache.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace forge {

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    max_positions_ = cfg.max_positions;
    n_sinks_ = std::min(cfg.max_positions, std::max(1u, cfg.n_sinks));

    layers_.resize(cfg.n_layers);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        layers_[l].heads.resize(cfg.n_kv_heads);
        for (uint32_t h = 0; h < cfg.n_kv_heads; h++) {
            layers_[l].heads[h].resize(max_positions_, cfg.head_dim);
        }
    }
}

void KVCache::store(uint32_t layer, uint32_t head, uint32_t pos,
                     const float* key, const float* value) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads) return;
    if (pos >= max_positions_) return;

    auto& hs = layers_[layer].heads[head];
    uint32_t hd = hs.head_dim;

    quantize_row(key, hs.keys.data() + pos * hd, &hs.k_scales[pos], hd);
    quantize_row(value, hs.values.data() + pos * hd, &hs.v_scales[pos], hd);

    hs.count = std::max(hs.count, pos + 1);
    if (pos >= cached_positions_) cached_positions_ = pos + 1;
}

void KVCache::attention_scores(uint32_t layer, uint32_t head,
                                const float* query,
                                float* scores_out, uint32_t* count_out) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads) {
        *count_out = 0;
        return;
    }

    auto& hs = layers_[layer].heads[head];
    uint32_t hd = hs.head_dim;
    *count_out = hs.count;

    for (uint32_t p = 0; p < hs.count; p++) {
        scores_out[p] = dot_product_quantized(
            query,
            hs.keys.data() + p * hd,
            hs.k_scales[p],
            hd
        );
    }
}

void KVCache::attention_weighted_sum(uint32_t layer, uint32_t head,
                                      const float* scores,
                                      float* output, uint32_t count) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads || count == 0) {
        std::memset(output, 0, cfg_.head_dim * sizeof(float));
        return;
    }

    auto& hs = layers_[layer].heads[head];
    uint32_t hd = hs.head_dim;

    weighted_sum_quantized(scores, hs.values.data(), hs.v_scales.data(),
                           output, count, hd);
}

void KVCache::clear() {
    for (auto& layer : layers_) {
        for (auto& hs : layer.heads) {
            hs.count = 0;
        }
    }
    cached_positions_ = 0;
}

size_t KVCache::memory_usage() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        for (const auto& hs : layer.heads) {
            total += hs.keys.capacity() * sizeof(int8_t);
            total += hs.values.capacity() * sizeof(int8_t);
            total += hs.k_scales.capacity() * sizeof(float);
            total += hs.v_scales.capacity() * sizeof(float);
        }
    }
    return total;
}

void KVCache::quantize_row(const float* src, int8_t* dst, float* scale_out, uint32_t n) const {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float absv = std::abs(src[i]);
        if (absv > amax) amax = absv;
    }
    *scale_out = (amax == 0.0f) ? 1.0f : amax / 127.0f;
    float inv_scale = (amax == 0.0f) ? 1.0f : 127.0f / amax;
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = (int8_t)std::round(src[i] * inv_scale);
    }
}

float KVCache::dot_product_quantized(const float* query, const int8_t* key,
                                      float scale, uint32_t n) const {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        sum += query[i] * ((float)key[i] * scale);
    }
    return sum;
}

void KVCache::weighted_sum_quantized(const float* scores, const int8_t* values,
                                      const float* v_scales, float* output,
                                      uint32_t count, uint32_t head_dim) const {
    std::memset(output, 0, head_dim * sizeof(float));
    for (uint32_t p = 0; p < count; p++) {
        float s = scores[p];
        float scale = v_scales[p];
        const int8_t* v = values + p * head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            output[i] += s * ((float)v[i] * scale);
        }
    }
}

} // namespace forge
