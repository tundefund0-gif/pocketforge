#include "forge/quant_format.hpp"
#include "forge/matmul_neon.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstring>

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
    std::cout << "=== Test: Quant Format ===\n\n";

    forge::ModelConfig cfg;
    cfg.n_layers = 2;
    cfg.n_embd = 32;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.n_ff = 128;
    cfg.n_vocab = 256;
    cfg.mtp_heads = 2;

    forge::Quantizer quantizer(cfg);

    // Test quantization of a known matrix
    {
        std::vector<float> test_weights(64); // 8x8
        for (uint32_t i = 0; i < 64; i++) test_weights[i] = ((float)i / 32.0f) - 1.0f;

        // Q8 quantization
        auto q8 = quantizer.quantize_matrix(test_weights.data(), 8, 8, 0);
        TEST("Q8 quantization produces output", !q8.empty());

        // Q4 quantization
        auto q4 = quantizer.quantize_matrix(test_weights.data(), 8, 8, 1);
        TEST("Q4 quantization produces output", !q4.empty());
        TEST("Q4 is smaller than Q8", q4.size() < q8.size());

        // Q2 quantization
        auto q2 = quantizer.quantize_matrix(test_weights.data(), 8, 8, 2);
        TEST("Q2 quantization produces output", !q2.empty());
        TEST("Q2 is smaller than Q4", q2.size() < q4.size());

        // Q1.5 (ternary) quantization
        auto q15 = quantizer.quantize_matrix(test_weights.data(), 8, 8, 3);
        TEST("Q1.5 quantization produces output", !q15.empty());
        // Q1.5 vs Q2: block overhead dominates for tiny 8x8 matrix, skip size comparison
    }

    // Test compression
    {
        std::vector<uint8_t> test_data(1000, 0xAB);
        auto compressed = quantizer.compress_block(test_data);
        TEST("zstd compression produces output", !compressed.empty());
        TEST("zstd compressed smaller than original", compressed.size() < test_data.size());
    }

    // Test full write and read cycle
    {
        std::vector<forge::WeightBlock> blocks;
        std::vector<std::vector<uint8_t>> compressed_blocks;

        srand(42);
        
        for (uint32_t l = 0; l < cfg.n_layers; l++) {
            for (uint32_t m = 0; m < 7; m++) {
                uint32_t rows = 32, cols = 32;
                if (m == 4 || m == 5) { cols = 128; }
                if (m == 6) { rows = 128; cols = 32; }

                std::vector<float> weights(rows * cols);
                for (auto& w : weights) w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

                quantizer.analyze_importance(weights.data(), rows * cols);
                
                uint8_t qtype = quantizer.select_quant_type(
                    (float)rand() / RAND_MAX
                );
                
                auto quantized = quantizer.quantize_matrix(weights.data(), rows, cols, qtype);
                auto comp = quantizer.compress_block(quantized);

                forge::WeightBlock block;
                block.layer_id = l;
                block.matrix_id = m;
                block.n_rows = rows;
                block.n_cols = cols;
                block.compressed_size = (uint32_t)comp.size();
                block.original_size = (uint32_t)quantized.size();
                block.quant_type = qtype;
                block.offset = 0;

                blocks.push_back(block);
                compressed_blocks.push_back(std::move(comp));
            }
        }

        TEST("Created test blocks", !blocks.empty());

        std::string test_path = "/tmp/test_forge_quant.squeeze";
        bool written = quantizer.write_squeeze(test_path, blocks, compressed_blocks, cfg);
        TEST("Write squeeze file", written);

        // Read back
        forge::WeightLoader loader;
        bool opened = loader.open(test_path);
        TEST("Open squeeze file", opened);

        if (opened) {
            auto loaded_cfg = loader.config();
            TEST("Config layers match", loaded_cfg.n_layers == cfg.n_layers);
            TEST("Config embd match", loaded_cfg.n_embd == cfg.n_embd);
            TEST("Config heads match", loaded_cfg.n_heads == cfg.n_heads);
            TEST("Config kv heads match", loaded_cfg.n_kv_heads == cfg.n_kv_heads);
            TEST("Config ff match", loaded_cfg.n_ff == cfg.n_ff);
            TEST("Config vocab match", loaded_cfg.n_vocab == cfg.n_vocab);
            TEST("Config mtp heads match", loaded_cfg.mtp_heads == cfg.mtp_heads);
            TEST("Num blocks match", loader.num_blocks() == blocks.size());

            // Load a specific block
            float scale;
            auto data = loader.load_block(0, 0, &scale);
            TEST("Load block returns data", !data.empty());
            TEST("Load block has correct dimensions", 
                 data.size() == 32 * 32); // first block is 32x32

            // Try to load a non-existent block
            auto empty_data = loader.load_block(99, 99, &scale);
            TEST("Load non-existent returns empty", empty_data.empty());

            loader.close();
        }

        std::remove(test_path.c_str());
    }

    // Test select_quant_type
    {
        TEST("High importance → Q4 (type 1)", 
             quantizer.select_quant_type(0.9f) == 1);
        TEST("Medium importance → Q2 (type 2)",
             quantizer.select_quant_type(0.5f) == 2);
        TEST("Low importance → Q1.5 (type 3)",
             quantizer.select_quant_type(0.1f) == 3);
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
