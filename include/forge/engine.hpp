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

    // Forward returns logits. Stores hidden state internally.
    std::vector<float> forward(const std::vector<int32_t>& tokens);

    // Forward and return both logits and the final hidden state
    std::vector<float> forward_with_hidden(
        const std::vector<int32_t>& tokens,
        std::vector<float>& hidden_out
    );

    // MTP speculative decode: generate tokens with MTP verification
    // Returns (accepted_count, last_logits)
    std::pair<int, std::vector<float>> step_mtp(
        int32_t current_token,
        const std::vector<float>& hidden_state,
        const std::vector<float>& embedding_table,
        int& mtp_accepted_out,
        int& mtp_rejected_out
    );

    // Sampling methods
    int32_t sample(const float* logits, uint32_t n_vocab,
                   const SamplingConfig& cfg = SamplingConfig());
    int32_t sample_greedy(const float* logits, uint32_t n_vocab);
    int32_t sample_top_k(const float* logits, uint32_t n_vocab,
                         uint32_t k, float temp);
    int32_t sample_top_p(const float* logits, uint32_t n_vocab,
                         float p, float temp);

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

    // Embedding table (tied, shared with unembed)
    std::vector<float> embedding_table_;
    bool embeddings_loaded_ = false;

    // Last hidden state from forward pass
    std::vector<float> last_hidden_;
    bool hidden_valid_ = false;

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
    std::vector<float> attn_scores_out_;

    // MTP buffers
    std::vector<float> verify_hidden_;
    std::vector<float> verify_logits_;

    // Real weight loading
    struct CachedWeight {
        std::vector<int8_t> data;
        std::vector<float> row_scales;
        float global_scale = 1.0f;
        bool loaded = false;
    };
    std::vector<CachedWeight> weight_cache_;
    void load_layer_weights(uint32_t layer_id);

    // RMS norm weights (1D per layer)
    std::vector<float> attn_norm_weights_;
    std::vector<float> ffn_norm_weights_;
    std::vector<float> final_norm_weights_;
    // bool norm_weights_loaded_ = false;
    bool real_weights_avail_ = false;

    void compute_attention(uint32_t layer_id, const float* input, float* output,
                           uint32_t token_pos);
    void compute_ffn(uint32_t layer_id, const float* input, float* output);
    const int8_t* load_weight_block(uint32_t layer_id, uint32_t matrix_id,
                                     float* scale_out);
    int32_t sample_token(const float* logits, uint32_t n_vocab);
    void generate_placeholder_logits(int32_t token, float* logits, uint32_t n_vocab);
};

} // namespace forge
#endif // FORGE_ENGINE_HPP
