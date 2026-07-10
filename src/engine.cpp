#include "forge/engine.hpp"
#include "forge/matmul_neon.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace forge {

bool ModelConfig::check_memory_budget() const {
    // Estimate KV cache: 24L × 4KV × ctx × 64dim × 2(K+V) × 1B
    uint32_t hd = head_dim();
    size_t kv_est = (size_t)n_layers * n_kv_heads * kv_cache_size * hd * 2 * 1;
    size_t mtp_est = (size_t)mtp_heads * n_embd * n_embd * 3 / 2; // Q4 MTP
    size_t act_est = (size_t)n_embd * 20 * 4;
    size_t total = kv_est + mtp_est + act_est;
    return total < max_memory;
}

Engine::Engine()
    : loader_(std::make_unique<WeightLoader>())
    , kv_cache_(nullptr)
    , prefetcher_(nullptr)
    , mtp_(nullptr)
    , skipper_(nullptr)
{
}

Engine::~Engine() = default;

bool Engine::load_model(const std::string& squeezef_path) {
    if (!loader_->open(squeezef_path)) {
        std::cerr << "Failed to open model: " << squeezef_path << std::endl;
        return false;
    }
    config_ = loader_->config();

    if (!config_.check_memory_budget()) {
        std::cerr << "WARNING: Config may exceed budget. Reducing ctx from "
                  << config_.kv_cache_size << " to " << (config_.kv_cache_size / 2) << "\n";
        config_.kv_cache_size /= 2;
    }

    // Build KVCacheConfig from ModelConfig
    KVCacheConfig kv_cfg;
    kv_cfg.n_layers = config_.n_layers;
    kv_cfg.n_kv_heads = config_.n_kv_heads;
    kv_cfg.head_dim = config_.head_dim();
    kv_cfg.max_positions = config_.kv_cache_size;
    kv_cfg.n_sinks = 4;
    kv_cache_ = std::make_unique<KVCache>(kv_cfg);

    prefetcher_ = std::make_unique<PrefetchPipeline>(loader_.get(), config_.n_layers, 7);

    if (config_.mtp_enabled) {
        MTPConfig mtp_cfg;
        mtp_cfg.n_mtp_heads = config_.mtp_heads;
        mtp_cfg.n_embd = config_.n_embd;
        mtp_cfg.n_vocab = config_.n_vocab;
        mtp_cfg.mtp_ff_dim = config_.n_embd;
        mtp_ = std::make_unique<MTPHeads>(mtp_cfg);
    }

    skipper_ = std::make_unique<LayerSkipController>(config_);

    // Pre-allocate buffers
    hidden_state_.resize(config_.n_embd, 0.0f);
    residual_.resize(config_.n_embd, 0.0f);
    attn_out_.resize(config_.n_embd, 0.0f);
    ffn_out_.resize(config_.n_embd, 0.0f);
    normed_.resize(config_.n_embd, 0.0f);
    q_buf_.resize(config_.n_embd, 0.0f);
    k_buf_.resize(config_.n_embd, 0.0f);
    v_buf_.resize(config_.n_embd, 0.0f);

    // Score buffer for attention — sized for max context
    scores_.resize(config_.kv_cache_size, 0.0f);

    prefetcher_->start();
    return true;
}

void Engine::set_max_context(uint32_t tokens) {
    config_.max_seq_len = std::min(tokens, (uint32_t)32768);
    config_.kv_cache_size = config_.max_seq_len;
    scores_.resize(config_.kv_cache_size, 0.0f);
}

void Engine::set_mtp_enabled(bool enabled) { config_.mtp_enabled = enabled; }
void Engine::set_layer_skip_threshold(float threshold) {
    config_.skip_threshold = threshold;
    if (skipper_) skipper_->set_threshold(threshold);
}
void Engine::set_max_memory(size_t bytes) { config_.max_memory = bytes; }

GenerationResult Engine::generate(
    const std::vector<int32_t>& prompt_tokens,
    uint32_t max_new_tokens,
    TokenCallback callback,
    ProgressCallback progress
) {
    GenerationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward({prompt_tokens[i]});
        if (progress && i % 10 == 0)
            progress((float)i / (prompt_tokens.size() + max_new_tokens));
    }
    result.tokens = prompt_tokens;

    uint32_t n_generated = 0;
    int32_t current_token = prompt_tokens.empty() ? 0 : prompt_tokens.back();

    while (n_generated < max_new_tokens) {
        if (get_memory_stats().total() > config_.max_memory) {
            std::cerr << "OOM protection: hit limit\n";
            result.oom_protected = true;
            break;
        }

        auto logits = forward({current_token});
        if (progress)
            progress((float)(prompt_tokens.size() + n_generated) /
                     (prompt_tokens.size() + max_new_tokens));

        current_token = sample_token(logits.data(), config_.n_vocab);
        result.tokens.push_back(current_token);
        n_generated++;
        if (callback) callback(current_token, 1.0f);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(end_time - start_time).count();
    result.tokens_per_second = (float)result.tokens.size() / elapsed;
    result.peak_memory_bytes = get_memory_stats().total();
    result.layers_skipped = skipper_ ? skipper_->total_skipped() : 0;
    return result;
}

std::vector<float> Engine::forward(const std::vector<int32_t>& tokens) {
    uint32_t n_embd = config_.n_embd;
    int32_t token = tokens.back();

    // Embedding (placeholder hash)
    for (uint32_t i = 0; i < n_embd; i++) {
        uint64_t h = (uint64_t)token * 2654435761ULL + i;
        hidden_state_[i] = (float)(h % 1000) / 1000.0f - 0.5f;
    }

    std::vector<float> norm_weight(n_embd, 1.0f);
    uint32_t token_pos = tokens.size() - 1;

    if (prefetcher_) prefetcher_->prefetch_layer(0);

    for (uint32_t l = 0; l < config_.n_layers; l++) {
        rmsnorm(normed_.data(), hidden_state_.data(), norm_weight.data(), n_embd);

        // Attention with streaming scores
        compute_attention(l, normed_.data(), attn_out_.data(), token_pos);

        for (uint32_t i = 0; i < n_embd; i++) hidden_state_[i] += attn_out_[i];
        rmsnorm(normed_.data(), hidden_state_.data(), norm_weight.data(), n_embd);

        bool skip_ffn = false;
        if (skipper_) skip_ffn = skipper_->should_skip_ffn(l, normed_.data(), n_embd);

        if (!skip_ffn) {
            compute_ffn(l, normed_.data(), ffn_out_.data());
            for (uint32_t i = 0; i < n_embd; i++) hidden_state_[i] += ffn_out_[i];
            if (skipper_) skipper_->record_layer_output(l, hidden_state_.data(), n_embd);
        }

        if (prefetcher_ && l + 1 < config_.n_layers)
            prefetcher_->prefetch_layer(l + 1);
    }

    rmsnorm(normed_.data(), hidden_state_.data(), norm_weight.data(), n_embd);

    // Unembed (placeholder)
    std::vector<float> logits(config_.n_vocab, 0.0f);
    for (uint32_t v = 0; v < config_.n_vocab; v++) {
        uint64_t h = (uint64_t)token * 2654435761ULL + v * 7919ULL;
        logits[v] = (float)(h % 10000) / 10000.0f;
    }
    return logits;
}

void Engine::compute_attention(uint32_t layer_id, const float* input,
                                float* output, uint32_t token_pos) {
    uint32_t n_heads = config_.n_heads;
    uint32_t n_kv_heads = config_.n_kv_heads;
    uint32_t n_embd = config_.n_embd;
    uint32_t head_dim = config_.head_dim();

    std::fill(attn_out_.begin(), attn_out_.end(), 0.0f);

    // Placeholder QKV
    for (uint32_t i = 0; i < n_embd; i++) {
        q_buf_[i] = input[i] * 0.1f;
        k_buf_[i] = input[i] * 0.1f;
        v_buf_[i] = input[i] * 0.1f;
    }

    // Streaming attention per head (no scratch buffer needed)
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_head = h % n_kv_heads;
        float* q_head = q_buf_.data() + h * head_dim;

        // Get scores via streaming int8 dot products
        uint32_t kv_count = 0;
        kv_cache_->attention_scores(layer_id, kv_head, q_head,
                                     scores_.data(), &kv_count);

        // Score for current token (not yet cached)
        float self_score = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) {
            self_score += q_head[i] * k_buf_[kv_head * head_dim + i];
        }
        scores_[kv_count] = self_score;

        // Scale and softmax
        float scale = 1.0f / std::sqrt((float)head_dim);
        for (uint32_t i = 0; i <= kv_count; i++) scores_[i] *= scale;
        softmax(scores_.data(), kv_count + 1);

        // Weighted sum from cached values (streaming int8)
        kv_cache_->attention_weighted_sum(layer_id, kv_head, scores_.data(),
                                           output, kv_count);

        // Add current token's contribution
        float self_prob = scores_[kv_count];
        for (uint32_t i = 0; i < head_dim; i++) {
            attn_out_[h * head_dim + i] += self_prob * v_buf_[kv_head * head_dim + i];
        }
    }

    // Store KV for future
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        kv_cache_->store(layer_id, h, token_pos,
                         k_buf_.data() + h * head_dim,
                         v_buf_.data() + h * head_dim);
    }
    std::memcpy(output, attn_out_.data(), n_embd * sizeof(float));
}

void Engine::compute_ffn(uint32_t layer_id, const float* input, float* output) {
    uint32_t n_embd = config_.n_embd;
    (void)layer_id;
    for (uint32_t i = 0; i < n_embd; i++) {
        float val = 0.0f;
        for (uint32_t j = 0; j < n_embd; j++) {
            val += input[j] * 0.01f;
        }
        output[i] = std::max(0.0f, val);
    }
}

const int8_t* Engine::load_weight_block(uint32_t layer_id, uint32_t matrix_id,
                                         float* scale_out) {
    if (prefetcher_) {
        auto block = prefetcher_->wait_for_block(layer_id, matrix_id);
        if (block.ready) {
            static std::vector<std::vector<int8_t>> block_cache;
            block_cache.emplace_back(std::move(block.data));
            return block_cache.back().data();
        }
    }
    return nullptr;
}

int32_t Engine::sample_token(const float* logits, uint32_t n_vocab) {
    return MTPHeads::argmax(logits, n_vocab);
}

Engine::MemoryStats Engine::get_memory_stats() const {
    MemoryStats stats;
    if (kv_cache_) stats.kv_cache = kv_cache_->memory_usage();
    if (prefetcher_) stats.prefetch_buffer = prefetcher_->memory_usage();
    if (mtp_) stats.mtp_heads = mtp_->memory_usage();
    stats.activations = hidden_state_.capacity() * sizeof(float) +
                        attn_out_.capacity() * sizeof(float) +
                        ffn_out_.capacity() * sizeof(float) +
                        normed_.capacity() * sizeof(float) +
                        q_buf_.capacity() * sizeof(float) +
                        k_buf_.capacity() * sizeof(float) +
                        v_buf_.capacity() * sizeof(float) +
                        scores_.capacity() * sizeof(float);
    return stats;
}

size_t Engine::MemoryStats::total() const {
    return weights_decompressed + kv_cache + activations +
           prefetch_buffer + mtp_heads + other;
}

} // namespace forge
