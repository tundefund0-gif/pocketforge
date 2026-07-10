#include "forge/engine.hpp"
#include "forge/matmul_neon.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace forge {

// ============================================================
//  Helpers
// ============================================================

static std::mt19937 rng_engine(42);

bool ModelConfig::check_memory_budget() const {
    uint32_t hd = head_dim();
    size_t window_size = std::min(kv_cache_size, (uint32_t)8192);
    size_t pool_entries = (max_seq_len > window_size) ? (max_seq_len - window_size) / 64 : 0;
    size_t kv_est = (size_t)n_layers * n_kv_heads * (window_size + pool_entries) * hd * 2 * 1;
    size_t mtp_est = (size_t)mtp_heads * n_embd * n_embd * 3 / 2;
    size_t act_est = (size_t)n_embd * 20 * 4;
    size_t total = kv_est + mtp_est + act_est;
    return total < max_memory;
}

// ============================================================
//  Constructor / Destructor
// ============================================================

Engine::Engine()
    : loader_(std::make_unique<WeightLoader>())
    , kv_cache_(nullptr)
    , prefetcher_(nullptr)
    , mtp_(nullptr)
    , skipper_(nullptr)
{
}

Engine::~Engine() = default;

// ============================================================
//  Model loading
// ============================================================

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
    kv_cfg.window_size = std::min(config_.kv_cache_size, (uint32_t)8192);
    kv_cfg.max_context = config_.max_seq_len;
    kv_cfg.sink_size = 4;
    kv_cfg.pool_block = 64;
    kv_cache_ = std::make_unique<KVCache>(kv_cfg);

    prefetcher_ = std::make_unique<PrefetchPipeline>(loader_.get(), config_.n_layers, 7);

    if (config_.mtp_enabled) {
        MTPConfig mtp_cfg;
        mtp_cfg.n_mtp_heads = config_.mtp_heads;
        mtp_cfg.n_embd = config_.n_embd;
        mtp_cfg.n_vocab = config_.n_vocab;
        mtp_cfg.mtp_ff_dim = config_.n_embd;
        mtp_ = std::make_unique<MTPHeads>(mtp_cfg);
        mtp_->init_weights();
    }

    skipper_ = std::make_unique<LayerSkipController>(config_);

    // Pre-allocate buffers
    hidden_state_.resize(config_.n_embd, 0.0f);
    residual_.resize(config_.n_embd, 0.0f);
    {
        // Attention projection buffers need head_dim-aware sizing
        uint32_t hd = config_.head_dim();
        uint32_t attn_sz = std::max(config_.n_heads * hd, config_.n_embd);
        uint32_t kv_sz = std::max(config_.n_kv_heads * hd, config_.n_embd);
        attn_out_.resize(attn_sz, 0.0f);
        q_buf_.resize(attn_sz, 0.0f);
        k_buf_.resize(kv_sz, 0.0f);
        v_buf_.resize(kv_sz, 0.0f);
    }
    ffn_out_.resize(config_.n_embd, 0.0f);
    normed_.resize(config_.n_embd, 0.0f);

    // Precompute RoPE tables
    {
        uint32_t hd = config_.head_dim();
        rope_theta_ = 10000.0f;  // default
        // Try to read from config metadata (set during load_model)
        rope_sin_.resize(hd);
        rope_cos_.resize(hd);
        rope_initialized_ = false;
    }
    last_hidden_.resize(config_.n_embd, 0.0f);
    verify_hidden_.resize(config_.n_embd, 0.0f);
    verify_logits_.resize(config_.n_vocab, 0.0f);

    // Score buffers - sized for max context
    scores_.resize(config_.kv_cache_size + 4096, 0.0f);
    attn_scores_out_.resize(config_.kv_cache_size + 4096, 0.0f);

    // Embedding table (int8 quantized, per-row scaling)
    // Limit vocab for low-RAM devices (32-bit phones with <256 MB)
    uint32_t embed_vocab = std::min(config_.n_vocab, (uint32_t)16384);
    if (embed_vocab <= 32000 && config_.n_embd <= 2048) {
        // Check if we have budget: int8 = 1 byte per weight + 4 bytes per row for scale
        size_t emb_size = (size_t)config_.n_vocab * config_.n_embd; // int8
        size_t emb_scales = (size_t)config_.n_vocab * sizeof(float);
        size_t total_emb = emb_size + emb_scales;
        
        // If embedding table is too big, reduce vocab or skip
        size_t emb_bytes_reduced = (size_t)embed_vocab * config_.n_embd;
        if (emb_bytes_reduced + get_memory_stats().total() < config_.max_memory) {
            // Will be filled from .squeeze data below
            embedding_table_.resize((size_t)embed_vocab * config_.n_embd);
            embeddings_loaded_ = false;
        } else {
            std::cerr << "WARNING: Embedding table too large (" 
                      << (total_emb / (1024*1024)) << " MB), skipping\n";
            embedding_table_.clear();
            embeddings_loaded_ = false;
        }
    }

    prefetcher_->start();

    // Try loading embedding table from .squeeze (matrix_id=10)
    {
        auto emb_block = loader_->load_block_with_scales(0, 10);
        if (!emb_block.data.empty()) {
            uint32_t emb_vocab = std::min(config_.n_vocab, (uint32_t)16384);
            uint32_t emb_dim = config_.n_embd;
            embedding_table_.resize((size_t)emb_vocab * emb_dim);
            float scale = emb_block.global_scale;
            size_t max_vals = emb_block.data.size();
            for (uint32_t v = 0; v < emb_vocab; v++) {
                for (uint32_t d = 0; d < emb_dim; d++) {
                    size_t idx = (size_t)v * emb_dim + d;
                    if (idx < max_vals) {
                        embedding_table_[idx] = (float)emb_block.data[idx] * scale;
                    }
                }
            }
            embeddings_loaded_ = true;
            std::cout << "Loaded embedding table (" << emb_block.data.size() << " bytes, " 
                      << emb_vocab << " tokens)\n";
        }
    }

    // Pre-allocate weight cache (7 matrices per layer, initially empty)
    // Only cache 2 layers worth of weights to save RAM (sliding window eviction)
    weight_cache_.resize(std::max((uint32_t)2, config_.n_layers) * 7);
    // Also pre-allocate norm weight arrays (loaded from .squeeze)
    attn_norm_weights_.resize(config_.n_layers * config_.n_embd, 1.0f);
    ffn_norm_weights_.resize(config_.n_layers * config_.n_embd, 1.0f);
    final_norm_weights_.resize(config_.n_embd, 1.0f);

    // Load norm weights from .squeeze (or skip if not stored)
    if (loader_) {
        // Output norm (layer 0, matrix_id=9)
        {
            auto block = loader_->load_block_with_scales(0, 9);
            if (!block.data.empty()) {
                uint32_t n = std::min((uint32_t)final_norm_weights_.size(), block.n_cols);
                for (uint32_t i = 0; i < n; i++) {
                    final_norm_weights_[i] = (float)block.data[i] * block.global_scale;
                }
                std::cout << "Loaded output norm (" << n << " weights)\n";
            }
        }
        // Per-layer norm weights
        for (uint32_t l = 0; l < config_.n_layers; l++) {
            auto block_a = loader_->load_block_with_scales(l, 7);
            if (!block_a.data.empty()) {
                uint32_t n = std::min(config_.n_embd, block_a.n_cols);
                for (uint32_t i = 0; i < n; i++)
                    attn_norm_weights_[l * config_.n_embd + i] = (float)block_a.data[i] * block_a.global_scale;
            }
            auto block_f = loader_->load_block_with_scales(l, 8);
            if (!block_f.data.empty()) {
                uint32_t n = std::min(config_.n_embd, block_f.n_cols);
                for (uint32_t i = 0; i < n; i++)
                    ffn_norm_weights_[l * config_.n_embd + i] = (float)block_f.data[i] * block_f.global_scale;
            }
        }
    }

    return true;
}

bool Engine::load_norm_weights_from_gguf(const std::string& gguf_path) {
    forge::GGUFFile gguf;
    if (!gguf.open(gguf_path)) {
        std::cerr << "Failed to open GGUF for norm weights: " << gguf_path << "\n";
        return false;
    }
    
    uint32_t n_embd = config_.n_embd;
    uint32_t n_layers = config_.n_layers;
    
    // Resize norm arrays if needed
    attn_norm_weights_.resize(n_layers * n_embd, 1.0f);
    ffn_norm_weights_.resize(n_layers * n_embd, 1.0f);
    final_norm_weights_.resize(n_embd, 1.0f);
    
    // Output norm
    {
        auto data = gguf.dequantize_tensor("output_norm.weight");
        if (data.empty()) data = gguf.dequantize_tensor("model.norm.weight");
        if (data.empty()) data = gguf.dequantize_tensor("gpt_neox.final_layer_norm.weight");
        if (!data.empty()) {
            uint32_t n = std::min(n_embd, (uint32_t)data.size());
            std::memcpy(final_norm_weights_.data(), data.data(), n * sizeof(float));
            std::cout << "Loaded output_norm from GGUF\n";
        }
    }
    
    // Per-layer norms
    for (uint32_t l = 0; l < n_layers; l++) {
        std::string attn_key = "blk." + std::to_string(l) + ".attn_norm.weight";
        std::string ffn_key = "blk." + std::to_string(l) + ".ffn_norm.weight";
        
        auto attn_data = gguf.dequantize_tensor(attn_key);
        if (!attn_data.empty()) {
            uint32_t n = std::min(n_embd, (uint32_t)attn_data.size());
            std::memcpy(attn_norm_weights_.data() + l * n_embd, attn_data.data(), n * sizeof(float));
        }
        
        auto ffn_data = gguf.dequantize_tensor(ffn_key);
        if (!ffn_data.empty()) {
            uint32_t n = std::min(n_embd, (uint32_t)ffn_data.size());
            std::memcpy(ffn_norm_weights_.data() + l * n_embd, ffn_data.data(), n * sizeof(float));
        }
    }
    
    std::cout << "Loaded norm weights from GGUF: " << gguf_path << "\n";
    gguf.close();
    return true;
}

void Engine::set_max_context(uint32_t tokens) {
    config_.max_seq_len = std::min(tokens, (uint32_t)131072);
    config_.kv_cache_size = std::min(tokens, (uint32_t)8192);
    scores_.resize(config_.kv_cache_size + 4096, 0.0f);
    attn_scores_out_.resize(config_.kv_cache_size + 4096, 0.0f);
}

void Engine::set_mtp_enabled(bool enabled) { config_.mtp_enabled = enabled; }
void Engine::set_layer_skip_threshold(float threshold) {
    config_.skip_threshold = threshold;
    if (skipper_) skipper_->set_threshold(threshold);
}
void Engine::set_max_memory(size_t bytes) { config_.max_memory = bytes; }

// ============================================================
//  Generation loop with MTP speculative decoding
// ============================================================

GenerationResult Engine::generate(
    const std::vector<int32_t>& prompt_tokens,
    uint32_t max_new_tokens,
    TokenCallback callback,
    ProgressCallback progress
) {
    GenerationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process prompt tokens
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

        // ——— MTP Speculative Decoding ———
        if (config_.mtp_enabled && mtp_ && hidden_valid_ && embeddings_loaded_
            && n_generated + 4 <= max_new_tokens) {

            // Step 1: Main forward at current token
            auto base_logits = forward({current_token});

            // Sample first token
            int32_t T1 = sample(base_logits.data(), config_.n_vocab);
            result.tokens.push_back(T1);
            n_generated++;
            if (callback) callback(T1, 1.0f);

            // Step 2: Get MTP predictions from last hidden state
            if (!last_hidden_.empty() && !embedding_table_.empty()) {
                auto mtp_logits_all = mtp_->forward(
                    last_hidden_.data(), embedding_table_);

                // Sample predictions from each MTP head
                std::vector<int32_t> mtp_preds;
                for (uint32_t h = 0; h < mtp_logits_all.size(); h++) {
                    int32_t pred = sample(mtp_logits_all[h].data(), config_.n_vocab);
                    mtp_preds.push_back(pred);
                }

                // Step 3: Verify each prediction
                int accepted = 0;
                int rejected = 0;
                int32_t last_verified = T1;

                for (uint32_t h = 0; h < mtp_preds.size(); h++) {
                    // Forward the last verified token
                    std::vector<float> verify_out;
                    int32_t verify_input = (h == 0) ? T1 : mtp_preds[h - 1];
                    auto verify_logits = forward_with_hidden(
                        {verify_input}, verify_hidden_);

                    // Check if main model agrees with MTP prediction
                    int32_t main_pred = sample_greedy(verify_logits.data(), config_.n_vocab);
                    int32_t mtp_pred = mtp_preds[h];

                    if (main_pred == mtp_pred) {
                        // ACCEPT: MTP prediction matches main model
                        last_verified = mtp_pred;
                        result.tokens.push_back(mtp_pred);
                        n_generated++;
                        accepted++;
                        if (callback) callback(mtp_pred, 1.0f);
                        // Update hidden state
                        last_hidden_ = verify_hidden_;
                    } else {
                        // REJECT: sample from main model's distribution
                        int32_t corrective = sample(verify_logits.data(), config_.n_vocab);
                        if (corrective != mtp_pred) {
                            // Only add if different from rejected prediction
                            result.tokens.push_back(corrective);
                            n_generated++;
                            rejected++;
                            if (callback) callback(corrective, 1.0f);
                        }
                        break;
                    }
                }

                result.mtp_accepted += accepted;
                result.mtp_rejected += rejected;

                // Continue the main token loop
                current_token = result.tokens.back();
                continue;
            }
        }

        // ——— Non-MTP path (fallback) ———
        auto logits = forward({current_token});

        if (progress)
            progress((float)(prompt_tokens.size() + n_generated) /
                     (prompt_tokens.size() + max_new_tokens));

        current_token = sample(logits.data(), config_.n_vocab);
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

// ============================================================
//  MTP step helper
// ============================================================

std::pair<int, std::vector<float>> Engine::step_mtp(
    int32_t current_token,
    const std::vector<float>& hidden_state,
    const std::vector<float>& embedding_table,
    int& mtp_accepted_out,
    int& mtp_rejected_out
) {
    mtp_accepted_out = 0;
    mtp_rejected_out = 0;

    // Get MTP predictions
    if (!mtp_ || embedding_table.empty()) {
        auto logits = forward({current_token});
        return {0, logits};
    }

    auto mtp_logits_all = mtp_->forward(hidden_state.data(), embedding_table);

    // Sample predictions from each MTP head
    std::vector<int32_t> mtp_preds;
    for (uint32_t h = 0; h < mtp_logits_all.size(); h++) {
        int32_t pred = sample(mtp_logits_all[h].data(), config_.n_vocab);
        mtp_preds.push_back(pred);
    }

    // Verify predictions
    int accepted = 0;
    for (uint32_t h = 0; h < mtp_preds.size(); h++) {
        int32_t verify_input = (h == 0) ? current_token : mtp_preds[h - 1];
        std::vector<float> tmp_hidden;
        auto verify_logits = forward_with_hidden({verify_input}, tmp_hidden);

        int32_t main_pred = sample_greedy(verify_logits.data(), config_.n_vocab);
        int32_t mtp_pred = mtp_preds[h];

        if (main_pred == mtp_pred) {
            accepted++;
        } else {
            mtp_rejected_out = (int)mtp_preds.size() - accepted;
            return {accepted, verify_logits};
        }
    }

    mtp_accepted_out = accepted;
    std::vector<float> empty;
    return {accepted, empty};
}

// ============================================================
//  Forward pass (full transformer)
// ============================================================

std::vector<float> Engine::forward(const std::vector<int32_t>& tokens) {
    std::vector<float> tmp_hidden;
    return forward_with_hidden(tokens, tmp_hidden);
}

std::vector<float> Engine::forward_with_hidden(
    const std::vector<int32_t>& tokens,
    std::vector<float>& hidden_out
) {
    uint32_t n_embd = config_.n_embd;
    uint32_t n_vocab = config_.n_vocab;

    if (tokens.empty()) {
        hidden_out.assign(n_embd, 0.0f);
        return std::vector<float>(n_vocab, 0.0f);
    }

    int32_t token = tokens.back();

    // ============================================================
    //  Token embedding (with placeholder fallback)
    // ============================================================
    if (embeddings_loaded_ && (uint32_t)token < config_.n_vocab) {
        uint32_t embed_vocab = std::min(config_.n_vocab, (uint32_t)16384);
        if ((uint32_t)token < embed_vocab) {
            const float* emb_row = embedding_table_.data() + token * n_embd;
            std::memcpy(hidden_state_.data(), emb_row, n_embd * sizeof(float));
        } else {
            // Out-of-range token: use zero embedding (better than random/UB)
            std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
        }
    } else {
        // Placeholder embedding
        uint64_t seed = (uint64_t)token * 2654435761ULL;
        for (uint32_t i = 0; i < n_embd; i++) {
            hidden_state_[i] = (float)((seed + i * 7919ULL) % 10000) / 10000.0f;
        }
    }

    // ============================================================
    //  Transformer layers
    // ============================================================
    for (uint32_t l = 0; l < config_.n_layers; l++) {
        // Use learned norm weights (or fallback to 1.0 if not loaded)
        const float* attn_norm_w = (l * config_.n_embd < attn_norm_weights_.size()) 
            ? attn_norm_weights_.data() + l * config_.n_embd : nullptr;
        const float* ffn_norm_w = (l * config_.n_embd < ffn_norm_weights_.size()) 
            ? ffn_norm_weights_.data() + l * config_.n_embd : nullptr;
        
        // Dummy fallback if norm weights not loaded
        std::vector<float> fallback_norm;
        if (!attn_norm_w) {
            fallback_norm.assign(n_embd, 1.0f);
            attn_norm_w = fallback_norm.data();
        }
        if (!ffn_norm_w) {
            if (fallback_norm.empty()) fallback_norm.assign(n_embd, 1.0f);
            ffn_norm_w = fallback_norm.data();
        }
        
        // RMSNorm before attention (use learned attn_norm weights)
        rmsnorm(normed_.data(), hidden_state_.data(), attn_norm_w, n_embd);

        // Attention
        compute_attention(l, normed_.data(), attn_out_.data(), l);

        // Residual
        for (uint32_t i = 0; i < n_embd; i++) {
            hidden_state_[i] += attn_out_[i];
        }

        // RMSNorm before FFN (use learned ffn_norm weights)
        rmsnorm(normed_.data(), hidden_state_.data(), ffn_norm_w, n_embd);

        // Layer skip
        bool skip_ffn = false;
        if (skipper_) {
            skip_ffn = skipper_->should_skip_ffn(l, normed_.data(), n_embd);
        }

        if (!skip_ffn) {
            compute_ffn(l, normed_.data(), ffn_out_.data());
            for (uint32_t i = 0; i < n_embd; i++) {
                hidden_state_[i] += ffn_out_[i];
            }
            if (skipper_) {
                skipper_->record_layer_output(l, hidden_state_.data(), n_embd);
            }
        }

        if (prefetcher_ && l + 1 < config_.n_layers) {
            prefetcher_->prefetch_layer(l + 1);
        }
    }

    // Save hidden state for MTP
    last_hidden_ = hidden_state_;
    hidden_valid_ = true;
    if (!hidden_out.empty()) {
        std::memcpy(hidden_out.data(), hidden_state_.data(), n_embd * sizeof(float));
    }

    // Final RMSNorm (use learned output_norm weights)
    {
        const float* final_w = (final_norm_weights_.size() >= n_embd) 
            ? final_norm_weights_.data() : nullptr;
        std::vector<float> fallback;
        if (!final_w) {
            fallback.assign(n_embd, 1.0f);
            final_w = fallback.data();
        }
        rmsnorm(normed_.data(), hidden_state_.data(), final_w, n_embd);
    }

    // ============================================================
    //  Unembed (logits)
    // ============================================================
    std::vector<float> logits(n_vocab, 0.0f);

    if (embeddings_loaded_ && n_vocab <= 32000) {
        // Use tied embeddings for unembedding
        for (uint32_t v = 0; v < n_vocab; v++) {
            float sum = 0.0f;
            const float* emb_row = embedding_table_.data() + v * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) {
                sum += hidden_state_[i] * emb_row[i];
            }
            logits[v] = sum;
        }
    } else {
        // Placeholder logits (hash-based for testing)
        generate_placeholder_logits(token, logits.data(), n_vocab);
    }

    return logits;
}

// ============================================================
//  Placeholder logits (for testing without real model)
// ============================================================

void Engine::generate_placeholder_logits(int32_t token, float* logits, uint32_t n_vocab) {
    for (uint32_t v = 0; v < n_vocab; v++) {
        uint64_t h = (uint64_t)token * 2654435761ULL + v * 7919ULL;
        logits[v] = (float)(h % 10000) / 10000.0f;
    }
}

// ============================================================
//  Layer weight loading
// ============================================================

void Engine::load_layer_weights(uint32_t layer_id) {
    if (!loader_ || loader_->num_blocks() == 0) return;
    uint32_t max_cached = weight_cache_.size();
    if (max_cached == 0) return;
    
    // Evict weights from ALL layers (keep only current layer)
    // This keeps RAM usage to ~1 layer worth of weights instead of all 24
    for (uint32_t i = 0; i < max_cached; i++) {
        if (weight_cache_[i].loaded && (i / 7) != layer_id) {
            weight_cache_[i].data.clear();
            weight_cache_[i].data.shrink_to_fit();
            weight_cache_[i].row_scales.clear();
            weight_cache_[i].row_scales.shrink_to_fit();
            weight_cache_[i].loaded = false;
        }
    }
    
    uint32_t base = layer_id * 7;
    if (base + 7 > max_cached) return;
    bool all_loaded = true;
    for (uint32_t m = 0; m < 7; m++) {
        if (!weight_cache_[base + m].loaded) {
            auto result = loader_->load_block_with_scales(layer_id, m);
            if (!result.data.empty()) {
                weight_cache_[base + m].data = std::move(result.data);
                weight_cache_[base + m].row_scales = std::move(result.row_scales);
                weight_cache_[base + m].global_scale = result.global_scale;
                weight_cache_[base + m].loaded = true;
            } else {
                all_loaded = false;
            }
        }
    }
    if (all_loaded) real_weights_avail_ = true;
}

// ============================================================
//  Attention (with real weight support)
// ============================================================

void Engine::compute_attention(uint32_t layer_id, const float* input,
                                float* output, uint32_t token_pos) {
    uint32_t n_heads = config_.n_heads;
    uint32_t n_kv_heads = config_.n_kv_heads;
    uint32_t n_embd = config_.n_embd;
    uint32_t head_dim = config_.head_dim();

    // Compute RoPE sin/cos for current position on-the-fly
    // Precompute freqs once, cache them for efficiency
    // For now, compute on every call (inexpensive for single token)
    float rope_theta = rope_theta_;
    {
        if (!rope_initialized_) {
            rope_theta_ = 5000000.0f; // default for MiniCPM5
            for (uint32_t i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / std::pow(rope_theta_, (float)i / (float)head_dim);
                rope_sin_[i] = std::sin((float)token_pos * freq);
                rope_cos_[i] = std::cos((float)token_pos * freq);
                if (i + 1 < head_dim) {
                    rope_sin_[i + 1] = rope_sin_[i];
                    rope_cos_[i + 1] = rope_cos_[i];
                }
            }
            rope_initialized_ = true;
        } else {
            // Recompute for this position (different from previous)
            for (uint32_t i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / std::pow(rope_theta_, (float)i / (float)head_dim);
                rope_sin_[i] = std::sin((float)token_pos * freq);
                rope_cos_[i] = std::cos((float)token_pos * freq);
                if (i + 1 < head_dim) {
                    rope_sin_[i + 1] = rope_sin_[i];
                    rope_cos_[i + 1] = rope_cos_[i];
                }
            }
        }
    }

    std::fill(attn_out_.begin(), attn_out_.end(), 0.0f);

    // Try to load real weights
    uint32_t base = layer_id * 7;
    bool have_weights = (base + 7 <= weight_cache_.size() &&
                         weight_cache_[base + 0].loaded &&  // Q
                         weight_cache_[base + 1].loaded &&  // K
                         weight_cache_[base + 2].loaded);   // V

    if (have_weights) {
        // Real QKV projection using matmul_quantized
        // Q projection: q = input @ Wq^T  (input [1 x n_embd], Wq [n_embd x n_embd])
        const auto& wq = weight_cache_[base + 0];
        const auto& wk = weight_cache_[base + 1];
        const auto& wv = weight_cache_[base + 2];
        const auto& wo = weight_cache_[base + 3];

        // Q = input @ Wq (input is 1xK, Wq is NxK, output is 1xN)
        matmul_quantized(input, wq.data.data(), q_buf_.data(),
                         1, config_.n_heads * head_dim, config_.n_embd,
                         wq.row_scales.data(), nullptr);

        // K = input @ Wk (GQA: Wk is [n_kv_heads*head_dim x n_embd])
        matmul_quantized(input, wk.data.data(), k_buf_.data(),
                         1, n_kv_heads * head_dim, n_embd,
                         wk.row_scales.data(), nullptr);

        // V = input @ Wv
        matmul_quantized(input, wv.data.data(), v_buf_.data(),
                         1, n_kv_heads * head_dim, n_embd,
                         wv.row_scales.data(), nullptr);

        // Prefetch output weights for attention output projection
        (void)wo; // used later after attention computes
    } else {
        // Placeholder QKV projection (for testing)
        for (uint32_t i = 0; i < n_embd; i++) {
            q_buf_[i] = input[i] * 0.1f;
            k_buf_[i] = input[i] * 0.1f;
            v_buf_[i] = input[i] * 0.1f;
        }
    }

    // Apply RoPE to Q and K vectors
    {
        // Use the precomputed sin/cos for the current position
        // q_buf_ has n_heads * head_dim elements
        // k_buf_ has n_kv_heads * head_dim elements
        for (uint32_t h = 0; h < n_heads; h++) {
            float* q_head = q_buf_.data() + h * head_dim;
            uint32_t kv_head = h % n_kv_heads;
            float* k_head = k_buf_.data() + kv_head * head_dim;
            
            for (uint32_t i = 0; i < head_dim; i += 2) {
                float q0 = q_head[i];
                float q1 = (i + 1 < head_dim) ? q_head[i + 1] : 0.0f;
                float k0 = k_head[i];
                float k1 = (i + 1 < head_dim) ? k_head[i + 1] : 0.0f;
                float c = rope_cos_[i];
                float s = rope_sin_[i];
                q_head[i] = q0 * c - q1 * s;
                if (i + 1 < head_dim) q_head[i + 1] = q0 * s + q1 * c;
                k_head[i] = k0 * c - k1 * s;
                if (i + 1 < head_dim) k_head[i + 1] = k0 * s + k1 * c;
            }
        }
    }

    // Streaming attention per head
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_head = h % n_kv_heads;
        float* q_head = q_buf_.data() + h * head_dim;

        uint32_t kv_count = kv_cache_->attention_scores(
            layer_id, kv_head, q_head, scores_.data());

        float self_score = 0.0f;
        const float* k_head = k_buf_.data() + kv_head * head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            self_score += q_head[i] * k_head[i];
        }
        scores_[kv_count] = self_score;

        float scale = 1.0f / std::sqrt((float)head_dim);
        for (uint32_t i = 0; i <= kv_count; i++) scores_[i] *= scale;
        softmax(scores_.data(), kv_count + 1);

        kv_cache_->attention_weighted_sum(
            layer_id, kv_head, scores_.data(), output, kv_count);

        float self_prob = scores_[kv_count];
        const float* v_head = v_buf_.data() + kv_head * head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            attn_out_[h * head_dim + i] += self_prob * v_head[i];
        }
    }

    // Store KV for future tokens
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        kv_cache_->store(layer_id, h, token_pos,
                         k_buf_.data() + h * head_dim,
                         v_buf_.data() + h * head_dim);
    }

    // Apply output projection if weights available
    if (have_weights) {
        const auto& wo = weight_cache_[base + 3];
        // O = attn_out @ Wo^T
        matmul_quantized(attn_out_.data(), wo.data.data(), output,
                         1, config_.n_embd, config_.n_heads * head_dim,
                         wo.row_scales.data(), nullptr);
        std::memcpy(attn_out_.data(), output, n_embd * sizeof(float));
    }

    std::memcpy(output, attn_out_.data(), n_embd * sizeof(float));
}

// ============================================================
//  FFN (with real weight + SiLU support)
// ============================================================

void Engine::compute_ffn(uint32_t layer_id, const float* input, float* output) {
    uint32_t n_embd = config_.n_embd;
    uint32_t n_ff = config_.n_ff;

    uint32_t base = layer_id * 7;
    bool have_weights = (base + 7 <= weight_cache_.size() &&
                         weight_cache_[base + 4].loaded &&  // gate
                         weight_cache_[base + 5].loaded &&  // up
                         weight_cache_[base + 6].loaded);   // down

    if (have_weights) {
        const auto& w_gate = weight_cache_[base + 4];
        const auto& w_up   = weight_cache_[base + 5];
        const auto& w_down = weight_cache_[base + 6];

        // gate = input @ Wgate (n_ff x n_embd)
        std::vector<float> gate_buf(n_ff, 0.0f);
        matmul_quantized(input, w_gate.data.data(), gate_buf.data(),
                         1, n_ff, n_embd,
                         w_gate.row_scales.data(), nullptr);

        // up = input @ Wup (n_ff x n_embd)
        std::vector<float> up_buf(n_ff, 0.0f);
        matmul_quantized(input, w_up.data.data(), up_buf.data(),
                         1, n_ff, n_embd,
                         w_up.row_scales.data(), nullptr);

        // SiLU activation on gate: silu(x) = x * sigmoid(x)
        std::vector<float> ffn_hidden(n_ff, 0.0f);
        for (uint32_t i = 0; i < n_ff; i++) {
            float g = gate_buf[i];
            float sig = 1.0f / (1.0f + std::exp(-g));
            ffn_hidden[i] = g * sig * up_buf[i];
        }

        // down = ffn_hidden @ Wdown (n_embd x n_ff)
        matmul_quantized(ffn_hidden.data(), w_down.data.data(), output,
                         1, n_embd, n_ff,
                         w_down.row_scales.data(), nullptr);
    } else {
        // Placeholder FFN (for testing)
        for (uint32_t i = 0; i < n_embd; i++) {
            float val = 0.0f;
            for (uint32_t j = 0; j < n_embd; j++) {
                val += input[j] * 0.01f;
            }
            output[i] = std::max(0.0f, val);
        }
    }
}

// ============================================================
//  Weight loading (for real models)
// ============================================================

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

// ============================================================
//  Sampling methods
// ============================================================

int32_t Engine::sample_token(const float* logits, uint32_t n_vocab) {
    return sample(logits, n_vocab);
}

int32_t Engine::sample(const float* logits, uint32_t n_vocab,
                       const SamplingConfig& cfg) {
    if (cfg.greedy || cfg.temperature <= 0.0f) {
        return sample_greedy(logits, n_vocab);
    }

    if (cfg.top_p < 1.0f && cfg.top_p > 0.0f) {
        return sample_top_p(logits, n_vocab, cfg.top_p, cfg.temperature);
    }

    if (cfg.top_k > 0 && cfg.top_k < n_vocab) {
        return sample_top_k(logits, n_vocab, cfg.top_k, cfg.temperature);
    }

    // Default: temperature sampling
    std::vector<float> scaled(n_vocab);
    float temp = std::max(cfg.temperature, 0.001f);
    float max_val = logits[0];
    for (uint32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        scaled[i] = std::exp((logits[i] - max_val) / temp);
        sum += scaled[i];
    }

    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_engine);
    float cumulative = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        cumulative += scaled[i] / sum;
        if (r <= cumulative) return (int32_t)i;
    }
    return (int32_t)(n_vocab - 1);
}

int32_t Engine::sample_greedy(const float* logits, uint32_t n_vocab) {
    int32_t best = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = (int32_t)i;
        }
    }
    return best;
}

int32_t Engine::sample_top_k(const float* logits, uint32_t n_vocab,
                              uint32_t k, float temp) {
    // Get top-k indices
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
        scored.emplace_back(logits[i], (int32_t)i);
    }

    std::partial_sort(scored.begin(), scored.begin() + std::min((uint32_t)k, n_vocab),
                      scored.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });

    uint32_t n = std::min(k, n_vocab);
    float max_val = scored[0].first;
    std::vector<float> probs(n);
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = std::exp((scored[i].first - max_val) / std::max(temp, 0.001f));
        sum += probs[i];
    }

    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_engine);
    float cumulative = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        cumulative += probs[i] / sum;
        if (r <= cumulative) return scored[i].second;
    }
    return scored[n - 1].second;
}

int32_t Engine::sample_top_p(const float* logits, uint32_t n_vocab,
                              float p, float temp) {
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
        scored.emplace_back(logits[i], (int32_t)i);
    }

    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    float max_val = scored[0].first;
    std::vector<float> probs(n_vocab);
    float sum_all = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        probs[i] = std::exp((scored[i].first - max_val) / std::max(temp, 0.001f));
        sum_all += probs[i];
    }

    // Normalize
    for (uint32_t i = 0; i < n_vocab; i++) {
        probs[i] /= sum_all;
    }

    // Find cutoff
    float cumulative = 0.0f;
    uint32_t cutoff = n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
        cumulative += probs[i];
        if (cumulative >= p) {
            cutoff = i + 1;
            break;
        }
    }

    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_engine);
    cumulative = 0.0f;
    for (uint32_t i = 0; i < cutoff; i++) {
        cumulative += probs[i];
        if (r <= cumulative) return scored[i].second;
    }
    return scored[cutoff - 1].second;
}

// ============================================================
//  Memory stats
// ============================================================

Engine::MemoryStats Engine::get_memory_stats() const {
    MemoryStats stats;
    if (kv_cache_) stats.kv_cache = kv_cache_->memory_usage();
    if (prefetcher_) stats.prefetch_buffer = prefetcher_->memory_usage();
    if (mtp_) stats.mtp_heads = mtp_->memory_usage();

    stats.activations =
        hidden_state_.capacity() * sizeof(float) +
        residual_.capacity() * sizeof(float) +
        attn_out_.capacity() * sizeof(float) +
        ffn_out_.capacity() * sizeof(float) +
        normed_.capacity() * sizeof(float) +
        q_buf_.capacity() * sizeof(float) +
        k_buf_.capacity() * sizeof(float) +
        v_buf_.capacity() * sizeof(float) +
        scores_.capacity() * sizeof(float) +
        attn_scores_out_.capacity() * sizeof(float) +
        last_hidden_.capacity() * sizeof(float) +
        verify_hidden_.capacity() * sizeof(float) +
        verify_logits_.capacity() * sizeof(float);

    if (embeddings_loaded_) {
        // Embedding table: int8 weights + float scales
        size_t emb_bytes = embedding_table_.capacity() * sizeof(float);
        // But we also track the scale memory: n_vocab * sizeof(float)
        stats.other += emb_bytes + (size_t)config_.n_vocab * sizeof(float);
    }

    return stats;
}

size_t Engine::MemoryStats::total() const {
    return weights_decompressed + kv_cache + activations +
           prefetch_buffer + mtp_heads + other;
}

} // namespace forge
