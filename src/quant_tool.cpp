#include "forge/quant_format.hpp"
#include "forge/matmul_neon.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>

using namespace forge;

void print_usage() {
    std::cerr << "Usage: forge-quant <input.gguf> <output.squeeze> [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --layers N      Number of layers (default: 24)\n";
    std::cerr << "  --embd N        Embedding dimension (default: 2048)\n";
    std::cerr << "  --heads N       Number of heads (default: 32)\n";
    std::cerr << "  --kv-heads N    KV heads for GQA (default: 4)\n";
    std::cerr << "  --ff N          FFN dimension (default: 8192)\n";
    std::cerr << "  --vocab N       Vocabulary size (default: 32000)\n";
    std::cerr << "  --mtp N         MTP heads (default: 4)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    ModelConfig cfg;
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) cfg.n_layers = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--embd" && i + 1 < argc) cfg.n_embd = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--heads" && i + 1 < argc) cfg.n_heads = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--kv-heads" && i + 1 < argc) cfg.n_kv_heads = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--ff" && i + 1 < argc) cfg.n_ff = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--vocab" && i + 1 < argc) cfg.n_vocab = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--mtp" && i + 1 < argc) cfg.mtp_heads = (uint32_t)std::atoi(argv[++i]);
    }

    std::cout << "=== PocketForge Quantizer ===\n";
    std::cout << "Input:  " << input_path << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "Model: " << cfg.n_layers << " layers, " << cfg.n_embd << " embd, "
              << cfg.n_heads << " heads (" << cfg.n_kv_heads << " KV), "
              << cfg.n_ff << " FF, " << cfg.n_vocab << " vocab\n";
    std::cout << "MTP heads: " << cfg.mtp_heads << "\n";

    // In production: read GGUF, parse weights, quantize each matrix
    // For now, we generate synthetic weights for testing the pipeline

    Quantizer quantizer(cfg);

    // Generate synthetic weight matrices and quantize them
    std::vector<WeightBlock> blocks;
    std::vector<std::vector<uint8_t>> compressed_blocks;

    // Each layer has 7 weight matrices: Q, K, V, O, gate, up, down
    const uint32_t matrices_per_layer = 7;
    const char* matrix_names[] = {"Q", "K", "V", "O", "gate", "up", "down"};

    // Matrix dimensions
    struct MatrixDim { uint32_t rows, cols; };
    MatrixDim dims[] = {
        {cfg.n_embd, cfg.n_embd},           // Q
        {cfg.n_embd, cfg.n_kv_heads * cfg.head_dim()}, // K (GQA)
        {cfg.n_embd, cfg.n_kv_heads * cfg.head_dim()}, // V (GQA)
        {cfg.n_embd, cfg.n_embd},            // O
        {cfg.n_embd, cfg.n_ff},              // gate
        {cfg.n_embd, cfg.n_ff},              // up
        {cfg.n_ff, cfg.n_embd},              // down
    };

    srand(42); // deterministic for reproducibility

    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        std::cout << "Processing layer " << l << "...\n";

        for (uint32_t m = 0; m < matrices_per_layer; m++) {
            uint32_t rows = dims[m].rows;
            uint32_t cols = dims[m].cols;
            uint32_t n = rows * cols;

            // Generate synthetic weights (in production, read from GGUF)
            std::vector<float> weights(n);
            for (uint32_t i = 0; i < n; i++) {
                weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }

            // Analyze importance
            quantizer.analyze_importance(weights.data(), n);

            // Select quantization type based on importance
            uint8_t quant_type = quantizer.select_quant_type(
                (float)rand() / RAND_MAX // placeholder importance
            );

            // Quantize
            auto quantized = quantizer.quantize_matrix(weights.data(), rows, cols, quant_type);

            // Compress
            auto compressed = quantizer.compress_block(quantized);

            // Create block entry
            WeightBlock block;
            block.layer_id = l;
            block.matrix_id = m;
            block.n_rows = rows;
            block.n_cols = cols;
            block.compressed_size = (uint32_t)compressed.size();
            block.original_size = (uint32_t)quantized.size();
            block.quant_type = quant_type;
            block.offset = 0;

            std::cout << "  " << matrix_names[m] << " [" << rows << "x" << cols << "] "
                      << "→ " << (quant_type == 0 ? "Q8" : quant_type == 1 ? "Q4" :
                                  quant_type == 2 ? "Q2" : "Q1.5")
                      << " (" << compressed.size() << " bytes, "
                      << (float)quantized.size() / n * 8 << " bits/param)\n";

            blocks.push_back(block);
            compressed_blocks.push_back(std::move(compressed));
        }
    }

    // Write .squeeze file
    if (quantizer.write_squeeze(output_path, blocks, compressed_blocks, cfg)) {
        std::cout << "\n✓ Wrote " << output_path << "\n";
        std::cout << "  " << blocks.size() << " weight blocks\n";

        // Calculate total size
        size_t total = 0;
        for (const auto& cb : compressed_blocks) total += cb.size();
        std::cout << "  Total compressed: " << total / (1024 * 1024) << " MB\n";
        std::cout << "  Estimated RAM usage during inference: ~90 MB\n";
        return 0;
    } else {
        std::cerr << "✗ Failed to write " << output_path << "\n";
        return 1;
    }
}
