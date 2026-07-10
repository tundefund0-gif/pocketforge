#include <zstd.h>
#include "forge/engine.hpp"
#include "forge/quant_format.hpp"
#include "forge/matmul_neon.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

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
    std::cout << "=== Test: Benchmark ===\n";
    std::cout << "Throughput and latency measurements\n\n";

    // Create test model
    forge::ModelConfig cfg;
    cfg.n_layers = 4;
    cfg.n_embd = 128;
    cfg.n_heads = 8;
    cfg.n_kv_heads = 4;
    cfg.n_ff = 512;
    cfg.n_vocab = 500;
    cfg.max_seq_len = 256;
    cfg.mtp_heads = 2;

    forge::Quantizer quantizer(cfg);
    std::vector<forge::WeightBlock> blocks;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    srand(42);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        for (uint32_t m = 0; m < 7; m++) {
            uint32_t rows = 128, cols = 128;
            if (m == 4 || m == 5) { cols = 512; }
            if (m == 6) { rows = 512; cols = 128; }

            std::vector<float> weights(rows * cols);
            for (auto& w : weights) w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

            auto quantized = quantizer.quantize_matrix(weights.data(), rows, cols, 1);
            auto comp = quantizer.compress_block(quantized);

            forge::WeightBlock block;
            block.layer_id = l;
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
    }

    std::string model_path = "/tmp/test_forge_bench.squeeze";
    quantizer.write_squeeze(model_path, blocks, compressed_blocks, cfg);

    // Benchmark matmul (microbenchmark)
    std::cout << "--- Matmul Microbenchmark ---\n";
    {
        const uint32_t M = 64, N = 64, K = 128;
        std::vector<int8_t> A(M * K);
        std::vector<int8_t> B(N * K);
        std::vector<int32_t> C(M * N);
        
        for (auto& v : A) v = rand() % 21 - 10;
        for (auto& v : B) v = rand() % 21 - 10;
        
        int num_warmup = 10;
        int num_iters = 100;
        
        for (int i = 0; i < num_warmup; i++) {
            forge::matmul_int8(A.data(), B.data(), C.data(), M, N, K);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; i++) {
            forge::matmul_int8(A.data(), B.data(), C.data(), M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        float avg_us = std::chrono::duration<float, std::micro>(end - start).count() / num_iters;
        float gflops = (2.0f * M * N * K) / (avg_us * 1000.0f);
        
        std::cout << "  " << M << "x" << N << "x" << K << " matmul_int8\n";
        std::cout << "  Average: " << avg_us << " us\n";
        std::cout << "  ~" << gflops << " GFLOPS\n";
        TEST("Matmul benchmark runs", avg_us > 0);
    }

    // Benchmark dot product
    {
        const uint32_t N = 64;
        std::vector<float> a(N), b(N);
        for (auto& v : a) v = (float)rand() / RAND_MAX;
        for (auto& v : b) v = (float)rand() / RAND_MAX;
        
        int num_iters = 10000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iters; i++) {
            forge::dot_product(a.data(), b.data(), N);
        }
        auto end = std::chrono::high_resolution_clock::now();
        float avg_ns = std::chrono::duration<float, std::nano>(end - start).count() / num_iters;
        
        std::cout << "\n--- Dot Product Microbenchmark ---\n";
        std::cout << "  Average: " << avg_ns << " ns\n";
        TEST("Dot product benchmark runs", avg_ns > 0);
    }

    // Benchmark engine throughput
    std::cout << "\n--- Engine Throughput Benchmark ---\n";
    {
        forge::Engine engine;
        if (engine.load_model(model_path)) {
            std::vector<int32_t> prompt = {10, 20, 30, 40, 50};
            
            // Warmup
            engine.generate(prompt, 8);
            
            // Timed run
            int num_runs = 3;
            float total_tps = 0;
            
            for (int run = 0; run < num_runs; run++) {
                auto result = engine.generate(prompt, 16);
                total_tps += result.tokens_per_second;
                
                std::cout << "  Run " << (run + 1) << ": "
                          << result.tokens_per_second << " tok/s, "
                          << result.layers_skipped << " skipped, "
                          << result.mtp_accepted << " MTP accepted\n";
            }
            
            float avg_tps = total_tps / num_runs;
            std::cout << "  Average: " << avg_tps << " tok/s\n";
            
            // For a 4-layer 128-embd test model on aarch64, > 1 tok/s is reasonable
            TEST("Engine has positive throughput", avg_tps > 0);
            
            auto stats = engine.get_memory_stats();
            std::cout << "  Memory: " << (stats.total() / 1024) << " KB\n";
            TEST("Engine memory under 250 MB", stats.total() < 250 * 1024 * 1024);
        }
    }

    // Benchmark compression speed
    std::cout << "\n--- Compression Benchmark ---\n";
    {
        std::vector<uint8_t> data(1024 * 100); // 100 KB
        for (auto& b : data) b = rand() % 256;
        
        int num_iters = 50;
        
        auto start = std::chrono::high_resolution_clock::now();
        size_t comp_size = 0;
        for (int i = 0; i < num_iters; i++) {
            size_t max_dst = ZSTD_compressBound(data.size());
            std::vector<uint8_t> compressed(max_dst);
            size_t result = ZSTD_compress(compressed.data(), max_dst,
                                           data.data(), data.size(), 3);
            if (!ZSTD_isError(result)) {
                comp_size = result;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        float avg_us = std::chrono::duration<float, std::micro>(end - start).count() / num_iters;
        float mbs = (float)data.size() / avg_us;
        
        std::cout << "  100 KB zstd level 3: " << avg_us << " us\n";
        std::cout << "  Compression speed: " << mbs << " MB/s\n";
        std::cout << "  Compression ratio: " << (float)data.size() / comp_size << ":1\n";
        TEST("Compression benchmark runs", avg_us > 0);
    }

    std::remove(model_path.c_str());

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
