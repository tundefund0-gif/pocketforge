#include "forge/types.hpp"
#include "forge/quant_format.hpp"
#include <cassert>
#include <iostream>
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
    std::cout << "=== Test: Types ===\n\n";

    // ModelConfig
    forge::ModelConfig cfg;
    TEST("Default n_layers", cfg.n_layers == 24);
    TEST("Default n_embd", cfg.n_embd == 1536);
    TEST("Default n_heads", cfg.n_heads == 16);
    TEST("Default n_kv_heads", cfg.n_kv_heads == 2);
    TEST("Default n_ff", cfg.n_ff == 4608);
    TEST("Default n_vocab", cfg.n_vocab == 130560);
    TEST("Default max_seq_len", cfg.max_seq_len == 131072);
    TEST("Default mtp_heads", cfg.mtp_heads == 4);
    TEST("Default skip_interval", cfg.skip_interval == 4);
    TEST("Default max_memory", cfg.max_memory == 250 * 1024 * 1024);

    // Derived
    TEST("head_dim()", cfg.head_dim() == 96); // 1536/16 when head_dim_=0
    TEST("kv_head_dim()", cfg.kv_head_dim() == 768); // 1536/2 when head_dim_=0

    // Q8_0 size
    forge::Q8_0 q8;
    TEST("Q8_0 block size", sizeof(q8) == sizeof(float) + 32);
    
    // Q4_0 size
    forge::Q4_0 q4;
    TEST("Q4_0 block size", sizeof(q4) == sizeof(float) + 16);

    // Q2_0 size
    forge::Q2_0 q2;
    TEST("Q2_0 block size", sizeof(q2) == sizeof(float) + 8);

    // Q1_5 size
    forge::Q1_5 q15;
    TEST("Q1.5 block size", sizeof(q15) == 16);

    // SqueezeHeader size
    forge::SqueezeHeader hdr;
    TEST("SqueezeHeader size", sizeof(hdr) == 256);

    // Magic bytes
    TEST("Magic header correct",
         hdr.magic[0] == 'F' && hdr.magic[1] == 'O' &&
         hdr.magic[2] == 'R' && hdr.magic[3] == 'G' &&
         hdr.magic[4] == 'E');

    // WeightBlock
    forge::WeightBlock wb;
    wb.layer_id = 5;
    wb.matrix_id = 3;
    wb.n_rows = 2048;
    wb.n_cols = 2048;
    wb.compressed_size = 100000;
    wb.original_size = 400000;
    wb.quant_type = 1;
    wb.offset = 12345;

    TEST("WeightBlock fields", 
         wb.layer_id == 5 && wb.matrix_id == 3 &&
         wb.n_rows == 2048 && wb.n_cols == 2048 &&
         wb.compressed_size == 100000 &&
         wb.original_size == 400000 &&
         wb.quant_type == 1 && wb.offset == 12345);

    // GenerationResult
    forge::GenerationResult gen;
    gen.tokens = {1, 2, 3, 4, 5};
    gen.tokens_per_second = 12.5f;
    gen.peak_memory_bytes = 90 * 1024 * 1024;
    gen.mtp_accepted = 3;
    gen.layers_skipped = 42;

    TEST("GenerationResult fields",
         gen.tokens.size() == 5 &&
         gen.tokens_per_second > 12.0f &&
         gen.peak_memory_bytes == 90 * 1024 * 1024 &&
         gen.mtp_accepted == 3 &&
         gen.layers_skipped == 42);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
