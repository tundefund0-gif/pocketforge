#ifndef FORGE_MTP_HPP
#define FORGE_MTP_HPP

#include "types.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace forge {

// ============================================================
//  Low-memory Multi-Token Prediction
// ============================================================
//
// Uses tied embeddings (shares main model's embedding table for
// unembedding) + Q4-quantized FFN weights.
// Memory: 4 heads × 3 matrices × 2048×2048 × 0.5 = ~24 MB
//   vs float32: 768 MB — 32× reduction
//
// FFN dim is d×1 (no expansion) for MTP heads to save memory.
//

struct MTPConfig {
    uint32_t n_mtp_heads = 4;
    uint32_t n_embd      = 2048;
    uint32_t n_vocab     = 32000;
    // Use tiny FFN for MTP (no expansion)
    uint32_t mtp_ff_dim  = 2048; // = n_embd, no expansion
};

class MTPHeads {
public:
    explicit MTPHeads(const MTPConfig& cfg);

    // Initialize with small Q4-quantized weights
    void init_weights();

    // Forward: hidden → logits per head
    // embedding_table: tied embedding weights (n_vocab × n_embd)
    std::vector<std::vector<float>> forward(
        const float* hidden,
        const std::vector<float>& embedding_table  // flat: n_vocab * n_embd
    ) const;

    // Speculative acceptance check
    int speculative_accept(
        int32_t token,
        const std::vector<std::vector<float>>& mtp_logits
    ) const;

    static int32_t argmax(const float* logits, uint32_t n);

    size_t memory_usage() const;

private:
    MTPConfig cfg_;

    // Q4-quantized weights per head
    // Stored as int8 (raw nibbles) + float scales
    struct HeadWeights {
        // gate, up, down: each [n_embd × ff_dim]
        std::vector<uint8_t> gate_q4;   // packed Q4 nibbles
        std::vector<uint8_t> up_q4;
        std::vector<uint8_t> down_q4;
        std::vector<float>   gate_scale; // per-row scales (n_embd)
        std::vector<float>   up_scale;
        std::vector<float>   down_scale; // per-row scales (ff_dim)
    };

    std::vector<HeadWeights> heads_;

    static float silu(float x);
    static void quantize_q4(const float* src, uint8_t* dst, std::vector<float>& scales,
                            uint32_t rows, uint32_t cols);
    static void dequant_q4_row(const uint8_t* src, float* dst, float scale, uint32_t n);
};

} // namespace forge
#endif // FORGE_MTP_HPP
