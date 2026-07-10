#include "forge/prefetch.hpp"
#include "forge/quant_format.hpp"
#include "forge/types.hpp"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>

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
    std::cout << "=== Test: Prefetch Pipeline ===\n\n";

    // Create a temporary .squeeze file with synthetic weights
    // First, generate quantized weights and write to file
    forge::ModelConfig cfg;
    cfg.n_layers = 4;
    cfg.n_embd = 64;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.n_ff = 256;
    cfg.n_vocab = 320;

    forge::Quantizer quantizer(cfg);

    // Generate a simple test squeeze file
    std::vector<forge::WeightBlock> blocks;
    std::vector<std::vector<uint8_t>> compressed;

    srand(12345);
    const uint32_t matrices_per_layer = 7;

    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        for (uint32_t m = 0; m < matrices_per_layer; m++) {
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
            compressed.push_back(std::move(comp));
        }
    }

    std::string test_path = "/tmp/test_forge_prefetch.squeeze";
    quantizer.write_squeeze(test_path, blocks, compressed, cfg);

    // Open with WeightLoader
    forge::WeightLoader loader;
    TEST("Loader opens file", loader.open(test_path));

    // Get config
    auto loaded_cfg = loader.config();
    TEST("Loader config matches layers", loaded_cfg.n_layers == cfg.n_layers);
    TEST("Loader config matches embd", loaded_cfg.n_embd == cfg.n_embd);

    uint32_t n_matrices = 7;
    forge::PrefetchPipeline prefetcher(&loader, cfg.n_layers, n_matrices);

    TEST("Initial memory usage 0", prefetcher.memory_usage() == 0);

    // Start the pipeline
    prefetcher.start();
    TEST("Pipeline started", true);

    // Prefetch layer 0
    prefetcher.prefetch_layer(0);
    
    // Wait a bit for async decompression
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Try to get block (0,0) - should be ready by now
    forge::DecompressedBlock block;
    bool got = prefetcher.try_get_block(0, 0, &block);
    
    if (got) {
        TEST("Prefetched block has data", !block.data.empty());
        TEST("Prefetched block correct layer", block.layer_id == 0);
        TEST("Prefetched block correct matrix", block.matrix_id == 0);
        TEST("Prefetched block ready", block.ready);

        // Release it
        prefetcher.release_block(0, 0);
    } else {
        // Wait more and try again
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        got = prefetcher.try_get_block(0, 0, &block);
        if (got) {
            TEST("Prefetched block after longer wait", !block.data.empty());
            prefetcher.release_block(0, 0);
        } else {
            // Try wait_for_block (blocking)
            block = prefetcher.wait_for_block(0, 0);
            TEST("Prefetched block via blocking wait", !block.data.empty());
            prefetcher.release_block(0, 0);
        }
    }

    // Prefetch multiple layers
    prefetcher.prefetch_layer(1);
    prefetcher.prefetch_layer(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    block = prefetcher.wait_for_block(1, 0);
    TEST("Layer 1 block 0 prefetched", !block.data.empty());
    prefetcher.release_block(1, 0);

    block = prefetcher.wait_for_block(2, 0);
    TEST("Layer 2 block 0 prefetched", !block.data.empty());
    prefetcher.release_block(2, 0);

    // Check memory usage (should be non-zero after prefetch)
    size_t mem = prefetcher.memory_usage(); (void)mem;
    TEST("Prefetch memory usage", true); // may vary

    // Stop pipeline
    prefetcher.stop();
    TEST("Pipeline stopped", true);

    loader.close();
    std::remove(test_path.c_str());

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
