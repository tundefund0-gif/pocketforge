#include "forge/engine.hpp"
#include "forge/gguf_reader.hpp"
#include "forge/tokenizer.hpp"
#include "forge/matmul_neon.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>

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
        std::cerr << "  --prompt TEXT    Input prompt (default: 'Hello')\n";
        std::cerr << "  --system TEXT    System prompt (default: 'You are a helpful AI assistant')\n";
        std::cerr << "  --tokens N       Max tokens to generate (default: 256)\n";
        std::cerr << "  --temp F         Temperature (0.0 = greedy, default: 0.0)\n";
        std::cerr << "  --greedy         Greedy sampling (default: on)\n";
        std::cerr << "  --no-greedy      Enable temperature sampling\n";
        std::cerr << "  --top-k N        Top-K sampling (default: 40)\n";
        std::cerr << "  --top-p F        Top-P sampling (default: 0.9)\n";
        std::cerr << "  --ctx N          Context length (default: 4096)\n";
        std::cerr << "  --mtp BOOL       MTP speculative decode (default: 1)\n";
        std::cerr << "  --gguf PATH      GGUF file for tokenizer\n";
        std::cerr << "  --benchmark      Run benchmark mode\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "What is 2+2?";
    std::string system_prompt = "You are a helpful AI assistant. Always respond in plain English.";
    uint32_t max_tokens = 256;
    bool mtp_enabled = true;
    float skip_threshold = 0.01f;
    bool benchmark = false;
    uint32_t context_size = 4096;
    forge::SamplingConfig sampling;
    sampling.greedy = true;       // Default to greedy for clean output
    sampling.temperature = 0.0f;  // Greedy
    sampling.top_k = 40;
    sampling.top_p = 0.9f;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--system" && i + 1 < argc) system_prompt = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--mtp" && i + 1 < argc) mtp_enabled = (bool)std::atoi(argv[++i]);
        else if (arg == "--skip" && i + 1 < argc) skip_threshold = (float)std::atof(argv[++i]);
        else if (arg == "--ctx" && i + 1 < argc) context_size = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) { sampling.temperature = (float)std::atof(argv[++i]); sampling.greedy = (sampling.temperature == 0.0f); }
        else if (arg == "--top-k" && i + 1 < argc) sampling.top_k = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) sampling.top_p = (float)std::atof(argv[++i]);
        else if (arg == "--gguf" && i + 1 < argc) { /* skip */ }
        else if (arg == "--greedy") { sampling.greedy = true; sampling.temperature = 0.0f; }
        else if (arg == "--no-greedy") sampling.greedy = false;
        else if (arg == "--benchmark") benchmark = true;
    }

    // Initialize engine with reduced context to save RAM
    Engine engine;
    engine.set_max_context(context_size);
    std::cout << "Loading model: " << model_path << "...\n";

    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model!\n";
        return 1;
    }

    auto cfg = engine.config();
    std::cout << "Model: " << cfg.n_layers << " layers, "
              << cfg.n_embd << " embd, " << cfg.n_heads << " heads\n";
    std::cout << "KV cache: " << cfg.n_kv_heads << " heads, "
              << cfg.kv_cache_size << " ctx\n";
    std::cout << "MTP: " << (mtp_enabled ? "enabled" : "disabled") << "\n";
    std::cout << "Sampling: " << (sampling.greedy ? "greedy" : 
        ("temp=" + std::to_string(sampling.temperature) + " top-k=" + std::to_string(sampling.top_k) + " top-p=" + std::to_string(sampling.top_p)));
    std::cout << "\n\n";

    engine.set_mtp_enabled(mtp_enabled);
    engine.set_layer_skip_threshold(skip_threshold);

    // Load BPE tokenizer from GGUF
    forge::BPETokenizer tokenizer;
    {
        forge::GGUFFile tok_file;
        std::string gguf_src;
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--gguf" && i + 1 < argc) gguf_src = argv[++i];
        }
        if (gguf_src.empty()) {
            gguf_src = model_path;
            if (gguf_src.size() > 8 && gguf_src.substr(gguf_src.size()-8) == ".squeeze") {
                gguf_src = gguf_src.substr(0, gguf_src.size()-8) + ".q8.gguf";
            }
        }
        if (!gguf_src.empty() && tok_file.open(gguf_src)) {
            if (tok_file.has_token_strings()) {
                tokenizer.load_tokens(tok_file.get_token_strings());
                if (!tok_file.get_merge_strings().empty()) {
                    tokenizer.load_merges(tok_file.get_merge_strings());
                }
                std::cout << "Tokenizer: " << tok_file.token_strings_size() << " tokens loaded\n";
            }
        }
    }
    if (!tokenizer.is_loaded()) {
        std::cout << "Tokenizer: not loaded (use --gguf <file.gguf>)\n";
    }

    // Build MiniCPM5 chat-formatted prompt
    std::string formatted_prompt;
    if (tokenizer.is_loaded()) {
        // Use the MiniCPM5 chat format with system prompt
        formatted_prompt = "<system>" + system_prompt + "</system>\n"
                          "<user>" + prompt + "</user>\n"
                          "<assistant>";
    } else {
        formatted_prompt = prompt;
    }

    // Tokenize prompt using BPE tokenizer
    std::vector<int32_t> prompt_tokens;
    if (tokenizer.is_loaded()) {
        prompt_tokens = tokenizer.encode(formatted_prompt);
    }
    if (prompt_tokens.empty()) {
        // Fallback to char-level
        for (char c : formatted_prompt) {
            prompt_tokens.push_back((int32_t)(unsigned char)c);
        }
    }

    std::cout << "Chat format:\n";
    std::cout << "  System: " << system_prompt << "\n";
    std::cout << "  User: " << prompt << "\n\n";
    std::cout << "Response:\n ";

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

        std::cout << "\nWarmup...\n";
        engine.generate(prompt_tokens, 4, nullptr, nullptr);

        std::cout << "Benchmarking " << max_tokens << " tokens...\n";
        int num_runs = 3;
        float avg_tps = 0.0f;

        for (int run = 0; run < num_runs; run++) {
            auto result = engine.generate(prompt_tokens, max_tokens, nullptr, nullptr);
            avg_tps += result.tokens_per_second;
            std::cout << "  Run " << (run + 1) << ": "
                      << result.tokens_per_second << " tok/s, "
                      << result.layers_skipped << " layers skipped\n";
        }
        avg_tps /= num_runs;

        std::cout << "\n=== Results ===\n";
        std::cout << "Average throughput: " << avg_tps << " tokens/second\n";
        std::cout << "Peak memory: " << engine.get_memory_stats().total() / 1024 << " KB\n";
        std::cout << "Target: 250 MB max → "
                  << (engine.get_memory_stats().total() < 250 * 1024 * 1024 ? "✓ PASS" : "✗ FAIL")
                  << "\n";
    } else {
        // Generate response
        auto result = engine.generate(prompt_tokens, max_tokens,
            [&](int32_t token, float prob) {
                if (tokenizer.is_loaded()) {
                    std::string decoded = tokenizer.decode(token);
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
        std::cout << "Peak memory: " << result.peak_memory_bytes / 1024 << " KB ("
                  << (result.peak_memory_bytes / (1024*1024)) << " MB)\n";
        std::cout << "Target: 250 MB max → "
                  << (result.peak_memory_bytes < 250 * 1024 * 1024 ? "✓ PASS" : "✗ FAIL")
                  << "\n";
    }

    return 0;
}
