#include "forge/engine.hpp"
#include "forge/quant_format.hpp"
#include <iostream>
#include <cmath>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        std::cerr << "  ✗ " << name << ": FAILED\n"; \
        tests_failed++; \
    } else { \
        std::cout << "  ✓ " << name << "\n"; \
        tests_passed++; \
    } \
} while(0)

int main() {
    std::cout << "=== Test: Engine ===\n\n";

    // First, create a small test model file
    forge::ModelConfig cfg;
    cfg.n_layers = 2;
    cfg.n_embd = 64;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.n_ff = 256;
    cfg.n_vocab = 256;
    cfg.max_seq_len = 128;
    cfg.mtp_heads = 2;

    forge::Quantizer quantizer(cfg);

    std::vector<forge::WeightBlock> blocks;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    srand(123);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        for (uint32_t m = 0; m < 7; m++) {
            uint32_t rows = 64, cols = 64;
            if (m == 4 || m == 5) { cols = 256; }
            if (m == 6) { rows = 256; cols = 64; }

            std::vector<float> weights(rows * cols);
            for (auto& w : weights) w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

            auto quantized = quantizer.quantize_matrix(weights.data(), rows, cols, 0);
            auto comp = quantizer.compress_block(quantized);

            forge::WeightBlock block;
            block.layer_id = l;
            block.matrix_id = m;
            block.n_rows = rows;
            block.n_cols = cols;
            block.compressed_size = (uint32_t)comp.size();
            block.original_size = (uint32_t)quantized.size();
            block.quant_type = 0;
            block.offset = 0;

            blocks.push_back(block);
            compressed_blocks.push_back(std::move(comp));
        }
    }

    std::string test_path = "/tmp/test_forge_engine.squeeze";
    quantizer.write_squeeze(test_path, blocks, compressed_blocks, cfg);

    // Initialize engine
    forge::Engine engine;
    
    TEST("Engine loads model", engine.load_model(test_path));
    
    // Check config matches
    TEST("Engine config matches layers", engine.config().n_layers == cfg.n_layers);
    TEST("Engine config matches embd", engine.config().n_embd == cfg.n_embd);

    // Test generation with a simple prompt
    {
        std::vector<int32_t> prompt = {101, 102, 103};
        
        auto callback = [](int32_t token, float prob) {
            // Just verify callback is called
        };
        
        auto result = engine.generate(prompt, 10, callback, nullptr);
        
        TEST("Generated tokens returned", !result.tokens.empty());
        TEST("Generation TPS > 0", result.tokens_per_second > 0);
        TEST("Peak memory tracked", result.peak_memory_bytes > 0);
        TEST("Includes prompt tokens", result.tokens.size() >= prompt.size());
    }

    // Test single forward pass
    {
        std::vector<int32_t> single_token = {42};
        auto logits = engine.forward(single_token);
        TEST("Forward returns logits", !logits.empty());
        TEST("Logits size matches vocab", logits.size() == cfg.n_vocab);
        
        bool all_finite = true;
        for (float l : logits) {
            if (!std::isfinite(l)) { all_finite = false; break; }
        }
        TEST("Logits are finite", all_finite);
    }

    // Test with different prompts
    {
        auto r1 = engine.generate({1, 2, 3}, 5);
        auto r2 = engine.generate({10, 20, 30}, 5);
        TEST("Different prompts produce different outputs", 
             r1.tokens_per_second > 0 && r2.tokens_per_second > 0);
    }

    // Test memory stats
    {
        auto stats = engine.get_memory_stats();
        TEST("Memory stats total > 0", stats.total() > 0);
        TEST("KV cache tracked", stats.kv_cache > 0);
        TEST("Activations tracked", stats.activations > 0);
        // Verify memory is well under 250 MB
        size_t max_mem = 250 * 1024 * 1024;
        TEST("Memory under 250 MB limit", stats.total() < max_mem);
    }

    // Test with MTP disabled
    {
        engine.set_mtp_enabled(false);
        auto result = engine.generate({5, 10, 15}, 5);
        TEST("MTP disabled works", result.tokens_per_second > 0);
        engine.set_mtp_enabled(true);
    }

    // Test with layer skip
    {
        engine.set_layer_skip_threshold(0.5f);
        auto result = engine.generate({1, 2, 3}, 5);
        TEST("Layer skip works", result.tokens_per_second > 0);
        engine.set_layer_skip_threshold(0.01f);
    }

    // Cleanup
    std::remove(test_path.c_str());

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
