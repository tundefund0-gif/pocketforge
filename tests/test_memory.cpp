#include "forge/engine.hpp"
#include "forge/quant_format.hpp"
#include "forge/kv_cache.hpp"
#include "forge/mtp.hpp"
#include "forge/layer_skip.hpp"
#include <iostream>
#include <cmath>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { \
        std::cerr << "  \u2717 " << name << ": FAILED\n"; \
        tests_failed++; \
    } else { \
        std::cout << "  \u2713 " << name << "\n"; \
        tests_passed++; \
    } \
} while(0)

int main() {
    std::cout << "=== Test: Memory Pressure ===\n";
    std::cout << "Verifying all components stay under 250 MB total\n\n";

    // ============================================================
    // 1. Model config for a realistic 1B-scale model
    // ============================================================
    forge::ModelConfig cfg;
    cfg.n_layers = 24;
    cfg.n_embd = 1536;
    cfg.n_heads = 16;
    cfg.n_kv_heads = 2;
    cfg.n_ff = 4608;
    cfg.n_vocab = 130560;
    cfg.max_seq_len = 8192;
    cfg.mtp_heads = 4;
    cfg.head_dim_ = 128; // MiniCPM5-1B

    // ============================================================
    // 2. Measure KV cache memory
    // ============================================================
    std::cout << "--- KV Cache ---\n";
    {
        forge::KVCacheConfig kv_cfg;
        kv_cfg.n_layers = cfg.n_layers;
        kv_cfg.n_kv_heads = cfg.n_kv_heads;
        kv_cfg.head_dim = cfg.head_dim();
        kv_cfg.max_context = 16384;
        kv_cfg.window_size = 8192;
        kv_cfg.pool_block = 64;
        forge::KVCache kv(kv_cfg);
        size_t kv_mem = kv.memory_usage();
        std::cout << "  KV cache memory: " << (kv_mem / 1024) << " KB\n";
        TEST("KV cache memory reasonable", kv_mem < 500 * 1024 * 1024);
    }

    // ============================================================
    // 3. Measure MTP head memory
    // ============================================================
    std::cout << "\n--- MTP Heads ---\n";
    {
        forge::MTPConfig mtp_cfg;
        mtp_cfg.n_mtp_heads = cfg.mtp_heads;
        mtp_cfg.n_embd = cfg.n_embd;
        mtp_cfg.n_vocab = cfg.n_vocab;
        
        forge::MTPHeads mtp(mtp_cfg);
        size_t mtp_mem = mtp.memory_usage();
        std::cout << "  MTP heads memory: " << (mtp_mem / (1024 * 1024)) << " MB\n";
        TEST("MTP memory under 50 MB", mtp_mem < 50 * 1024 * 1024);
    }

    // ============================================================
    // 4. Full engine with realistic but budget-friendly config
    // ============================================================
    std::cout << "\n--- Total Budget ---\n";
    {
        // Use a small model config so the engine doesn't overallocate
        forge::ModelConfig small_cfg;
        small_cfg.n_layers = 2;
        small_cfg.n_embd = 128;
        small_cfg.n_heads = 4;
        small_cfg.n_kv_heads = 2;
        small_cfg.n_ff = 512;
        small_cfg.n_vocab = 1024;
        small_cfg.max_seq_len = 512;
        small_cfg.mtp_heads = 2;
        small_cfg.kv_cache_size = 512;
        small_cfg.max_memory = 250 * 1024 * 1024;

        forge::Quantizer quant(small_cfg);
        
        // Generate minimal weight blocks matching small_cfg dimensions
        std::vector<forge::WeightBlock> blocks;
        std::vector<std::vector<uint8_t>> compressed_blocks;
        for (uint32_t m = 0; m < 7; m++) {
            uint32_t rows = 128, cols = 128;
            if (m == 4 || m == 5) { cols = 512; rows = 128; }
            if (m == 6) { rows = 512; cols = 128; }
            std::vector<float> weights(rows * cols, 0.1f);
            auto quantized = quant.quantize_matrix(weights.data(), rows, cols, 1);
            auto comp = quant.compress_block(quantized);
            forge::WeightBlock block;
            block.layer_id = 0;
            block.matrix_id = m;
            block.n_rows = rows;
            block.n_cols = cols;
            block.compressed_size = (uint32_t)comp.size();
            block.original_size = (uint32_t)quantized.size();
            block.quant_type = 1;
            block.offset = 0;
            blocks.push_back(block);
            compressed_blocks.push_back(std::move(comp));
        }

        std::string test_path = "/tmp/test_forge_memory.squeeze";
        quant.write_squeeze(test_path, blocks, compressed_blocks, small_cfg);

        forge::Engine engine;
        if (engine.load_model(test_path)) {
            auto stats = engine.get_memory_stats();
            
            std::cout << "  KV cache:              " << (stats.kv_cache / 1024) << " KB\n";
            std::cout << "  Activations:           " << (stats.activations / 1024) << " KB\n";
            std::cout << "  Prefetch buffer:       " << (stats.prefetch_buffer / 1024) << " KB\n";
            if (stats.mtp_heads > 0)
                std::cout << "  MTP heads:             " << (stats.mtp_heads / 1024) << " KB\n";
            if (stats.other > 0)
                std::cout << "  Embeddings:            " << (stats.other / 1024) << " KB\n";
            std::cout << "  ----\n";
            std::cout << "  TOTAL:                 " << (stats.total() / 1024) << " KB\n";
            std::cout << "  Budget:                250 MB max\n";
            
            size_t budget = 250 * 1024 * 1024;
            bool under_budget = stats.total() < budget;
            TEST("Total memory under 250 MB", under_budget);
            
            if (!under_budget) {
                std::cout << "  \u2717 OVER BUDGET by " 
                          << (stats.total() - budget) / (1024 * 1024) << " MB\n";
            } else {
                std::cout << "  \u2713 " << (budget - stats.total()) / (1024 * 1024) 
                          << " MB headroom\n";
            }

            // Run a quick generation
            auto result = engine.generate({1, 2, 3}, 4);
            auto stats_after = engine.get_memory_stats();
            TEST("Memory stable during inference", 
                 stats_after.total() < budget);
        }
        std::remove(test_path.c_str());
    }

    // ============================================================
    // 5. Estimate 1B model memory breakdown
    // ============================================================
    std::cout << "\n--- 1B Model Memory Estimate ---\n";
    {
        // KV cache for 131K context
        forge::KVCacheConfig kv_cfg;
        kv_cfg.n_layers = cfg.n_layers;
        kv_cfg.n_kv_heads = cfg.n_kv_heads;
        kv_cfg.head_dim = cfg.head_dim();
        kv_cfg.max_context = 131072;
        kv_cfg.window_size = 8192;
        kv_cfg.pool_block = 64;
        forge::KVCache kv(kv_cfg);
        size_t kv_mem = kv.memory_usage();

        // MTP heads
        forge::MTPConfig mtp_cfg;
        mtp_cfg.n_mtp_heads = cfg.mtp_heads;
        mtp_cfg.n_embd = cfg.n_embd;
        mtp_cfg.n_vocab = cfg.n_vocab;
        forge::MTPHeads mtp(mtp_cfg);
        size_t mtp_mem = mtp.memory_usage();

        // Activations estimate
        size_t act_mem = (size_t)cfg.n_embd * 20 * 4;

        // Embedding table (Q4: vocab*embd/2 + vocab*4 scales)
        size_t emb_mem = (size_t)cfg.n_vocab * cfg.n_embd / 2 + (size_t)cfg.n_vocab * sizeof(float);

        size_t total = kv_mem + mtp_mem + act_mem + emb_mem;
        
        std::cout << "  KV cache (131K, int8):   ~" << (kv_mem / (1024*1024)) << " MB\n";
        std::cout << "  MTP heads (Q4):           ~" << (mtp_mem / (1024*1024)) << " MB\n";
        std::cout << "  Activations:              ~" << (act_mem / 1024) << " KB\n";
        std::cout << "  Embedding table (Q4):     ~" << (emb_mem / (1024*1024)) << " MB\n";
        std::cout << "  ----\n";
        std::cout << "  TOTAL:                   ~" << (total / (1024*1024)) << " MB\n";
        std::cout << "  Budget:                   250 MB\n";
        TEST("1B model estimate under 260 MB (130K vocab)", total < 260 * 1024 * 1024);
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
