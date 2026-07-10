#ifndef FORGE_ENGINE_HPP
#define FORGE_ENGINE_HPP

#include "types.hpp"
#include "quant_format.hpp"
#include "kv_cache.hpp"
#include "prefetch.hpp"
#include "mtp.hpp"
#include "layer_skip.hpp"
#include <memory>
#include <string>

namespace forge {

class Engine {
public:
    Engine();
    ~Engine();

    bool load_model(const std::string& squeezef_path);

    void set_max_context(uint32_t tokens);
    void set_mtp_enabled(bool enabled);
    void set_layer_skip_threshold(float threshold);
    void set_max_memory(size_t bytes);

    GenerationResult generate(
        const std::vector<int32_t>& prompt_tokens,
        uint32_t max_new_tokens,
        TokenCallback callback = nullptr,
        ProgressCallback progress = nullptr
    );

    std::vector<float> forward(const std::vector<int32_t>& tokens);

    struct MemoryStats {
        size_t weights_mmap = 0;
        size_t weights_decompressed = 0;
        size_t kv_cache = 0;
        size_t activations = 0;
        size_t prefetch_buffer = 0;
        size_t mtp_heads = 0;
        size_t other = 0;
        size_t total() const;
    };
    MemoryStats get_memory_stats() const;

    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;
    std::unique_ptr<WeightLoader> loader_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<PrefetchPipeline> prefetcher_;
    std::unique_ptr<MTPHeads> mtp_;
    std::unique_ptr<LayerSkipController> skipper_;

    // Pre-allocated buffers (never freed during runtime)
    std::vector<float> hidden_state_;
    std::vector<float> residual_;
    std::vector<float> attn_out_;
    std::vector<float> ffn_out_;
    std::vector<float> normed_;
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    std::vector<float> scores_;  // attention scores buffer

    void compute_attention(uint32_t layer_id, const float* input, float* output,
                           uint32_t token_pos);
    void compute_ffn(uint32_t layer_id, const float* input, float* output);
    const int8_t* load_weight_block(uint32_t layer_id, uint32_t matrix_id,
                                     float* scale_out);
    int32_t sample_token(const float* logits, uint32_t n_vocab);
};

} // namespace forge
#endif // FORGE_ENGINE_HPP
