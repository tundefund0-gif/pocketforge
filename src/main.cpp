#include "forge/engine.hpp"
#include "forge/gguf_reader.hpp"
#include "forge/matmul_neon.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace forge;

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║      PocketForge Inference Engine    ║\n";
    std::cout << "║     Custom 1B LLM on 32-bit ARM      ║\n";
    std::cout << "║         < 250 MB RAM, 8K+ ctx        ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    if (argc < 2) {
        std::cerr << "Usage: forge <model.squeeze> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --prompt TEXT   Input prompt\n";
        std::cerr << "  --tokens N      Max tokens to generate (default: 128)\n";
        std::cerr << "  --mtp BOOL      Enable MTP (default: 1)\n";
        std::cerr << "  --skip FLOAT    Layer skip threshold (default: 0.01)\n";
        std::cerr << "  --benchmark     Run benchmark mode\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "Hello";
    uint32_t max_tokens = 128;
    bool mtp_enabled = true;
    float skip_threshold = 0.01f;
    bool benchmark = false;
    forge::SamplingConfig sampling;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--mtp" && i + 1 < argc) mtp_enabled = (bool)std::atoi(argv[++i]);
        else if (arg == "--skip" && i + 1 < argc) skip_threshold = (float)std::atof(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) sampling.temperature = (float)std::atof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) sampling.top_k = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) sampling.top_p = (float)std::atof(argv[++i]);
        else if (arg == "--gguf" && i + 1 < argc) { /* skip, handled later */ }
        else if (arg == "--greedy") sampling.greedy = true;
        else if (arg == "--benchmark") benchmark = true;
    }

    // Initialize engine
    Engine engine;
    std::cout << "Loading model: " << model_path << "...\n";

    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model!\n";
        return 1;
    }

    auto cfg = engine.config();
    std::cout << "Model: " << cfg.n_layers << " layers, "
              << cfg.n_embd << " embd, " << cfg.n_heads << " heads\n";
    std::cout << "KV cache: " << cfg.n_kv_heads << " heads, "
              << cfg.max_seq_len << " max ctx\n";
    std::cout << "MTP: " << (mtp_enabled ? "enabled" : "disabled") << "\n";
    std::cout << "Layer skip threshold: " << skip_threshold << "\n";
    std::cout << "Sampling: temp=" << sampling.temperature << " top-k=" << sampling.top_k
              << " top-p=" << sampling.top_p;
    if (sampling.greedy) std::cout << " (greedy)";
    std::cout << "\n\n";

    engine.set_mtp_enabled(mtp_enabled);
    engine.set_layer_skip_threshold(skip_threshold);

    // Load tokenizer from GGUF for text decoding
    forge::GGUFFile tokenizer_file;
    std::string gguf_src;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--gguf" && i + 1 < argc) gguf_src = argv[++i];
    }
    if (gguf_src.empty()) {
        // Try replacing .squeeze with .q8.gguf
        gguf_src = model_path;
        if (gguf_src.size() > 8 && gguf_src.substr(gguf_src.size()-8) == ".squeeze") {
            gguf_src = gguf_src.substr(0, gguf_src.size()-8) + ".q8.gguf";
        }
    }
    bool has_token_decoder = false;
    if (!gguf_src.empty() && tokenizer_file.open(gguf_src)) {
        has_token_decoder = tokenizer_file.has_token_strings();
        if (has_token_decoder) {
            std::cout << "Tokenizer: " << (int)tokenizer_file.token_strings_size() << " tokens loaded\n";
        }
    }
    if (!has_token_decoder) {
        std::cout << "Tokenizer: not loaded (use --gguf <file.gguf>)\n";
    }

    // Tokenize prompt (placeholder: simple char-level for testing)
    std::vector<int32_t> prompt_tokens;
    for (char c : prompt) {
        prompt_tokens.push_back((int32_t)(unsigned char)c);
    }

    if (benchmark) {
        std::cout << "=== Benchmark Mode ===\n";
        auto stats = engine.get_memory_stats();
        std::cout << "Memory stats:\n";
        std::cout << "  KV cache:      " << stats.kv_cache / 1024 << " KB\n";
        std::cout << "  Prefetch buf:  " << stats.prefetch_buffer / 1024 << " KB\n";
        std::cout << "  Activations:   " << stats.activations / 1024 << " KB\n";
        if (stats.mtp_heads > 0)
            std::cout << "  MTP heads:     " << stats.mtp_heads / 1024 << " KB\n";
        std::cout << "  Estimated RAM: " << stats.total() / 1024 << " KB\n";

        // Run warmup
        std::cout << "\nWarmup...\n";
        auto warmup_result = engine.generate(prompt_tokens, 4, nullptr, nullptr);

        // Benchmark
        std::cout << "Benchmarking " << max_tokens << " tokens...\n";
        int num_runs = 3;
        float avg_tps = 0.0f;

        for (int run = 0; run < num_runs; run++) {
            auto result = engine.generate(prompt_tokens, max_tokens, nullptr, nullptr);
            avg_tps += result.tokens_per_second;
            std::cout << "  Run " << (run + 1) << ": "
                      << result.tokens_per_second << " tok/s, "
                      << result.layers_skipped << " layers skipped, "
                      << result.mtp_accepted << " MTP accepted\n";
        }
        avg_tps /= num_runs;

        std::cout << "\n=== Results ===\n";
        std::cout << "Average throughput: " << avg_tps << " tokens/second\n";
        std::cout << "Peak memory: " << engine.get_memory_stats().total() / 1024 << " KB\n";
        std::cout << "Target: 250 MB max → "
                  << (engine.get_memory_stats().total() < 250 * 1024 * 1024 ? "✓ PASS" : "✗ FAIL")
                  << "\n";
    } else {
        // Interactive generation
        std::cout << "Prompt: " << prompt << "\n\n";
        std::cout << "Generating...\n\n";

        auto result = engine.generate(prompt_tokens, max_tokens,
            [&](int32_t token, float prob) {
                // Decode token using loaded tokenizer
                std::string decoded;
                if (has_token_decoder) {
                    decoded = tokenizer_file.token_str(token);
                    // Output raw token string (terminal handles UTF-8/byte display)
                    std::cout << decoded;
                } else {
                    if (token >= 32 && token < 127) {
                        std::cout << (char)token;
                    } else {
                        std::cout << "[" << token << "]";
                    }
                }
                std::cout.flush();
            },
            [](float progress) {
                // Progress bar
                static int last_pct = -1;
                int pct = (int)(progress * 100);
                if (pct > last_pct) {
                    std::cout << "\rProgress: " << pct << "%" << std::flush;
                    last_pct = pct;
                }
            }
        );

        std::cout << "\n\n=== Generation Stats ===\n";
        std::cout << "Tokens generated: " << result.tokens.size() << "\n";
        std::cout << "Throughput: " << result.tokens_per_second << " tok/s\n";
        std::cout << "Layers skipped: " << result.layers_skipped << "\n";
        std::cout << "MTP accepted: " << result.mtp_accepted << "\n";
        std::cout << "Peak memory: " << result.peak_memory_bytes / 1024 << " KB\n";
    }

    return 0;
}
