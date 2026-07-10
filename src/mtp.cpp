#include "forge/mtp.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

namespace forge {

MTPHeads::MTPHeads(const MTPConfig& cfg) : cfg_(cfg) {
    heads_.resize(cfg.n_mtp_heads);
    init_weights();
}

void MTPHeads::init_weights() {
    uint32_t d = cfg_.n_embd;
    uint32_t ff = cfg_.mtp_ff_dim;
    float init_scale = 1.0f / std::sqrt((float)d);

    for (uint32_t h = 0; h < cfg_.n_mtp_heads; h++) {
        auto& head = heads_[h];
        
        // Pre-allocate Q4 storage (d * ff/2 bytes per matrix)
        head.gate_q4.resize(d * (ff / 2));
        head.up_q4.resize(d * (ff / 2));
        head.down_q4.resize(ff * (d / 2));
        head.gate_scale.resize(d, 1.0f);
        head.up_scale.resize(d, 1.0f);
        head.down_scale.resize(ff, 1.0f);

        // Generate random float weights, then quantize to Q4
        auto init_q4 = [init_scale](
            std::vector<uint8_t>& q, std::vector<float>& scales,
            uint32_t rows, uint32_t cols) {
            std::vector<float> tmp(rows * cols);
            for (auto& v : tmp) {
                v = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * init_scale;
            }
            MTPHeads::quantize_q4(tmp.data(), q.data(), scales, rows, cols);
        };

        init_q4(head.gate_q4, head.gate_scale, d, ff);
        init_q4(head.up_q4, head.up_scale, d, ff);
        init_q4(head.down_q4, head.down_scale, ff, d);
    }
}

std::vector<std::vector<float>> MTPHeads::forward(
    const float* hidden,
    const std::vector<float>& embedding_table
) const {
    std::vector<std::vector<float>> all_logits(cfg_.n_mtp_heads);
    uint32_t d = cfg_.n_embd;
    uint32_t ff = cfg_.mtp_ff_dim;
    bool use_tied = embedding_table.size() >= (size_t)cfg_.n_vocab * d;

    std::vector<float> current_hidden(hidden, hidden + d);
    std::vector<float> dequant_buf(std::max(d, ff)); // reusable dequantization buffer
    std::vector<float> gate_activ(ff);
    std::vector<float> up_activ(ff);
    std::vector<float> activated(ff);
    std::vector<float> ffn_out(d);

    for (uint32_t h = 0; h < cfg_.n_mtp_heads; h++) {
        const auto& head = heads_[h];

        // Gate: dequant Q4 rows on the fly
        for (uint32_t j = 0; j < ff; j++) {
            float sum = 0.0f;
            dequant_q4_row(head.gate_q4.data() + j * (d / 2),
                           dequant_buf.data(), head.gate_scale[j], d);
            for (uint32_t i = 0; i < d; i++) {
                sum += current_hidden[i] * dequant_buf[i];
            }
            gate_activ[j] = sum;
        }

        // Up: dequant Q4 rows on the fly
        for (uint32_t j = 0; j < ff; j++) {
            float sum = 0.0f;
            dequant_q4_row(head.up_q4.data() + j * (d / 2),
                           dequant_buf.data(), head.up_scale[j], d);
            for (uint32_t i = 0; i < d; i++) {
                sum += current_hidden[i] * dequant_buf[i];
            }
            up_activ[j] = sum;
        }

        // Activate: SiLU(gate) * up
        for (uint32_t j = 0; j < ff; j++) {
            activated[j] = silu(gate_activ[j]) * up_activ[j];
        }

        // Down project
        for (uint32_t i = 0; i < d; i++) {
            float sum = 0.0f;
            dequant_q4_row(head.down_q4.data() + i * (ff / 2),
                           dequant_buf.data(), head.down_scale[i], ff);
            for (uint32_t j = 0; j < ff; j++) {
                sum += activated[j] * dequant_buf[j];
            }
            ffn_out[i] = sum;
        }

        // Residual
        for (uint32_t i = 0; i < d; i++) {
            current_hidden[i] += ffn_out[i];
        }

        // Unembed
        std::vector<float> logits(cfg_.n_vocab);
        if (use_tied) {
            for (uint32_t v = 0; v < cfg_.n_vocab; v++) {
                float sum = 0.0f;
                const float* emb_row = embedding_table.data() + v * d;
                for (uint32_t i = 0; i < d; i++) {
                    sum += current_hidden[i] * emb_row[i];
                }
                logits[v] = sum;
            }
        } else {
            for (uint32_t v = 0; v < cfg_.n_vocab; v++) {
                logits[v] = (float)((v * 2654435761ULL + (uint64_t)h) % cfg_.n_vocab) / (float)cfg_.n_vocab;
            }
        }
        all_logits[h] = std::move(logits);
    }

    return all_logits;
}

int MTPHeads::speculative_accept(
    int32_t token,
    const std::vector<std::vector<float>>& mtp_logits
) const {
    int accepted = 1;
    uint32_t max_check = std::min((uint32_t)mtp_logits.size(), cfg_.n_mtp_heads);
    for (uint32_t h = 1; h < max_check; h++) {
        int32_t predicted = argmax(mtp_logits[h].data(), cfg_.n_vocab);
        if (predicted == token) accepted++;
        else break;
    }
    return accepted;
}

int32_t MTPHeads::argmax(const float* logits, uint32_t n) {
    int32_t best = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = (int32_t)i;
        }
    }
    return best;
}

float MTPHeads::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

size_t MTPHeads::memory_usage() const {
    size_t total = 0;
    for (const auto& head : heads_) {
        total += head.gate_q4.capacity();
        total += head.up_q4.capacity();
        total += head.down_q4.capacity();
        total += head.gate_scale.capacity() * sizeof(float);
        total += head.up_scale.capacity() * sizeof(float);
        total += head.down_scale.capacity() * sizeof(float);
    }
    return total;
}

void MTPHeads::quantize_q4(const float* src, uint8_t* dst,
                            std::vector<float>& scales,
                            uint32_t rows, uint32_t cols) {
    scales.resize(rows);
    for (uint32_t r = 0; r < rows; r++) {
        float amax = 0.0f;
        for (uint32_t c = 0; c < cols; c++) {
            amax = std::max(amax, std::abs(src[r * cols + c]));
        }
        float inv_scale = (amax == 0.0f) ? 1.0f : 127.0f / amax;
        scales[r] = (amax == 0.0f) ? 1.0f : amax / 127.0f;
        for (uint32_t c = 0; c < cols; c += 2) {
            int8_t v0 = (int8_t)std::round(src[r * cols + c] * inv_scale / 16.0f * 7.0f);
            v0 = std::max((int8_t)-8, std::min((int8_t)7, v0));
            uint8_t packed = (uint8_t)(v0 + 8);
            if (c + 1 < cols) {
                                int8_t v1 = (int8_t)std::round(src[r * cols + c + 1] * inv_scale);
                v1 = std::max((int8_t)-8, std::min((int8_t)7, v1));
                packed |= ((uint8_t)(v1 + 8)) << 4;
            }
            dst[r * (cols / 2) + c / 2] = packed;
        }
    }
}

void MTPHeads::dequant_q4_row(const uint8_t* src, float* dst, float scale, uint32_t n) {
    for (uint32_t i = 0; i < n; i += 2) {
        uint8_t packed = src[i / 2];
        int8_t v0 = (int8_t)(packed & 0x0F) - 8;
        dst[i] = (float)v0 * scale;
        if (i + 1 < n) {
            int8_t v1 = (int8_t)((packed >> 4) & 0x0F) - 8;
            dst[i + 1] = (float)v1 * scale;
        }
    }
}

} // namespace forge
