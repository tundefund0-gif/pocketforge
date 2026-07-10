#include "forge/kv_cache.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace forge {

// ============================================================
//  KVCache implementation
// ============================================================

float KVCache::dot_quant(const float* q, const int8_t* kv, float s, uint32_t n) const {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += q[i] * ((float)kv[i] * s);
    return sum;
}

void KVCache::quantize_row(const float* src, int8_t* dst, float* scale_out, uint32_t n) const {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float absv = std::abs(src[i]);
        if (absv > amax) amax = absv;
    }
    *scale_out = (amax == 0.0f) ? 1.0f : amax / 127.0f;
    float inv = (amax == 0.0f) ? 1.0f : 127.0f / amax;
    for (uint32_t i = 0; i < n; i++) dst[i] = (int8_t)std::round(src[i] * inv);
}

// ============================================================

KVCache::KVCache(const KVCacheConfig& cfg) : cfg_(cfg) {
    window_sz_ = std::min(cfg.window_size, cfg.max_context);
    pool_block_ = std::max(1u, cfg.pool_block);
    uint32_t hd = cfg.head_dim;
    uint32_t ws = window_sz_;

    layers_.resize(cfg.n_layers);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        layers_[l].resize(cfg.n_kv_heads);
        for (uint32_t h = 0; h < cfg.n_kv_heads; h++) {
            auto& c = layers_[l][h];
            c.head_dim = hd;
            c.sink_k.resize(8 * hd, 0);
            c.sink_v.resize(8 * hd, 0);
            c.sink_k_scale.resize(8, 1.0f);
            c.sink_v_scale.resize(8, 1.0f);
            c.win_k.resize(ws * hd, 0);
            c.win_v.resize(ws * hd, 0);
            c.win_k_scale.resize(ws, 1.0f);
            c.win_v_scale.resize(ws, 1.0f);
            c.win_pos.resize(ws, 0);
            c.acc_k.resize(hd, 0.0);
            c.acc_v.resize(hd, 0.0);
        }
    }
}

void KVCache::store(uint32_t layer, uint32_t head, uint32_t pos,
                     const float* key, const float* value) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads) return;
    if (pos >= cfg_.max_context) return;

    auto& c = layers_[layer][head];
    uint32_t hd = c.head_dim;

    if (pos < cfg_.sink_size) {
        // Sink storage
        if (pos < c.sink_k.size() / hd) {
            quantize_row(key, c.sink_k.data() + pos * hd, &c.sink_k_scale[pos], hd);
            quantize_row(value, c.sink_v.data() + pos * hd, &c.sink_v_scale[pos], hd);
            if (pos >= c.sink_count) c.sink_count = pos + 1;
        }
    } else {
        uint32_t ws = window_sz_;

        if (c.win_count >= ws) {
            // Evict oldest from window, pool it
            uint32_t slot = c.win_head;
            uint32_t evict_pos = c.win_pos[slot];

            if (c.acc_count == 0) c.acc_start = evict_pos;
            for (uint32_t i = 0; i < hd; i++) {
                c.acc_k[i] += (double)c.win_k[slot * hd + i] * c.win_k_scale[slot];
                c.acc_v[i] += (double)c.win_v[slot * hd + i] * c.win_v_scale[slot];
            }
            c.acc_count++;

            if (c.acc_count >= pool_block_) {
                // Finalize pooled block
                float inv = 1.0f / c.acc_count;
                uint32_t pidx = c.pool_count++;
                c.pool_start.push_back(c.acc_start);
                c.pool_end.push_back(evict_pos);
                c.pool_k.resize((pidx + 1) * hd);
                c.pool_v.resize((pidx + 1) * hd);
                c.pool_k_scale.push_back(1.0f);
                c.pool_v_scale.push_back(1.0f);

                std::vector<float> avg(hd);
                for (uint32_t i = 0; i < hd; i++) avg[i] = (float)(c.acc_k[i] * inv);
                quantize_row(avg.data(), c.pool_k.data() + pidx * hd, &c.pool_k_scale[pidx], hd);
                for (uint32_t i = 0; i < hd; i++) avg[i] = (float)(c.acc_v[i] * inv);
                quantize_row(avg.data(), c.pool_v.data() + pidx * hd, &c.pool_v_scale[pidx], hd);

                std::fill(c.acc_k.begin(), c.acc_k.end(), 0.0);
                std::fill(c.acc_v.begin(), c.acc_v.end(), 0.0);
                c.acc_count = 0;
            }

            // Write new data to evicted slot
            quantize_row(key, c.win_k.data() + slot * hd, &c.win_k_scale[slot], hd);
            quantize_row(value, c.win_v.data() + slot * hd, &c.win_v_scale[slot], hd);
            c.win_pos[slot] = pos;
            c.win_head = (c.win_head + 1) % ws;
        } else {
            uint32_t slot = c.win_head;
            quantize_row(key, c.win_k.data() + slot * hd, &c.win_k_scale[slot], hd);
            quantize_row(value, c.win_v.data() + slot * hd, &c.win_v_scale[slot], hd);
            c.win_pos[slot] = pos;
            c.win_head = (c.win_head + 1) % ws;
            c.win_count++;
        }
    }

    if (pos >= total_stored_) total_stored_ = pos + 1;
}

uint32_t KVCache::attention_scores(uint32_t layer, uint32_t head,
                                    const float* query, float* scores_out) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads) return 0;

    auto& c = layers_[layer][head];
    uint32_t hd = c.head_dim;
    uint32_t idx = 0;

    // Tier 1: Sink tokens
    for (uint32_t p = 0; p < c.sink_count; p++) {
        scores_out[idx++] = dot_quant(query,
            c.sink_k.data() + p * hd, c.sink_k_scale[p], hd);
    }

    // Tier 3: Pooled history blocks
    for (uint32_t p = 0; p < c.pool_count; p++) {
        scores_out[idx++] = dot_quant(query,
            c.pool_k.data() + p * hd, c.pool_k_scale[p], hd);
    }

    // Tier 2: Sliding window (logical order)
    uint32_t n = c.win_count;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t slot = (n < window_sz_) ? i : (c.win_head + i) % window_sz_;
        scores_out[idx++] = dot_quant(query,
            c.win_k.data() + slot * hd, c.win_k_scale[slot], hd);
    }

    return idx;
}

void KVCache::attention_weighted_sum(uint32_t layer, uint32_t head,
                                      const float* scores, float* output,
                                      uint32_t count) {
    if (layer >= cfg_.n_layers || head >= cfg_.n_kv_heads || count == 0) {
        std::memset(output, 0, cfg_.head_dim * sizeof(float));
        return;
    }

    auto& c = layers_[layer][head];
    uint32_t hd = c.head_dim;
    std::memset(output, 0, hd * sizeof(float));
    uint32_t idx = 0;

    // Tier 1: Sink
    for (uint32_t p = 0; p < c.sink_count && idx < count; p++, idx++) {
        float s = scores[idx];
        float sc = c.sink_v_scale[p];
        const int8_t* v = c.sink_v.data() + p * hd;
        for (uint32_t i = 0; i < hd; i++) output[i] += s * ((float)v[i] * sc);
    }

    // Tier 3: Pooled history
    for (uint32_t p = 0; p < c.pool_count && idx < count; p++, idx++) {
        float s = scores[idx];
        float sc = c.pool_v_scale[p];
        const int8_t* v = c.pool_v.data() + p * hd;
        for (uint32_t i = 0; i < hd; i++) output[i] += s * ((float)v[i] * sc);
    }

    // Tier 2: Window
    uint32_t n = c.win_count;
    for (uint32_t i = 0; i < n && idx < count; i++, idx++) {
        uint32_t slot = (n < window_sz_) ? i : (c.win_head + i) % window_sz_;
        float s = scores[idx];
        float sc = c.win_v_scale[slot];
        const int8_t* v = c.win_v.data() + slot * hd;
        for (uint32_t j = 0; j < hd; j++) output[j] += s * ((float)v[j] * sc);
    }
}

void KVCache::clear() {
    for (auto& layer : layers_) {
        for (auto& c : layer) {
            c.sink_count = 0;
            c.win_count = 0;
            c.win_head = 0;
            c.pool_count = 0;
            c.pool_k.clear(); c.pool_k.shrink_to_fit();
            c.pool_v.clear(); c.pool_v.shrink_to_fit();
            c.pool_k_scale.clear(); c.pool_k_scale.shrink_to_fit();
            c.pool_v_scale.clear(); c.pool_v_scale.shrink_to_fit();
            c.pool_start.clear(); c.pool_start.shrink_to_fit();
            c.pool_end.clear(); c.pool_end.shrink_to_fit();
            std::fill(c.acc_k.begin(), c.acc_k.end(), 0.0);
            std::fill(c.acc_v.begin(), c.acc_v.end(), 0.0);
            c.acc_count = 0;
        }
    }
    total_stored_ = 0;
}

size_t KVCache::memory_usage() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        for (const auto& c : layer) {
            total += c.sink_k.capacity() + c.sink_v.capacity();
            total += c.sink_k_scale.capacity() * sizeof(float);
            total += c.sink_v_scale.capacity() * sizeof(float);
            total += c.win_k.capacity() + c.win_v.capacity();
            total += c.win_k_scale.capacity() * sizeof(float);
            total += c.win_v_scale.capacity() * sizeof(float);
            total += c.win_pos.capacity() * sizeof(uint32_t);
            total += c.pool_k.capacity() + c.pool_v.capacity();
            total += c.pool_k_scale.capacity() * sizeof(float);
            total += c.pool_v_scale.capacity() * sizeof(float);
            total += c.pool_start.capacity() * sizeof(uint32_t);
            total += c.pool_end.capacity() * sizeof(uint32_t);
            total += c.acc_k.capacity() * sizeof(double);
            total += c.acc_v.capacity() * sizeof(double);
        }
    }
    return total;
}

} // namespace forge
