#include "forge/gguf_reader.hpp"
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
    std::cout << "=== Test: GGUF Reader ===\n\n";

    {
        forge::GGUFFile gguf;
        TEST("Default not open", !gguf.is_open());

        // Test with a non-existent file
        TEST("Open non-existent fails", !gguf.open("/tmp/nonexistent.gguf"));
    }

    // Test tensor name patterns with fake data
    // We'll test the helper functions indirectly via quant_tool patterns
    
    // Test GGUFValueType enum values
    TEST("GGUF_TYPE_UINT8 == 0", forge::GGUF_TYPE_UINT8 == 0);
    TEST("GGUF_TYPE_INT8 == 1", forge::GGUF_TYPE_INT8 == 1);
    TEST("GGUF_TYPE_FLOAT32 == 6", forge::GGUF_TYPE_FLOAT32 == 6);
    TEST("GGUF_TYPE_STRING == 8", forge::GGUF_TYPE_STRING == 8);
    TEST("GGUF_TYPE_ARRAY == 9", forge::GGUF_TYPE_ARRAY == 9);

    // Test GGMLType enum values
    TEST("GGML_TYPE_F32 == 0", forge::GGML_TYPE_F32 == 0);
    TEST("GGML_TYPE_F16 == 1", forge::GGML_TYPE_F16 == 1);
    TEST("GGML_TYPE_Q8_0 == 8", forge::GGML_TYPE_Q8_0 == 8);
    TEST("GGML_TYPE_Q4_0 == 2", forge::GGML_TYPE_Q4_0 == 2);

    // Test GGUFTensorInfo helpers
    {
        forge::GGUFTensorInfo info;
        info.type = forge::GGML_TYPE_F32;
        info.n_elems = 100;
        TEST("F32 element_size == 4", info.element_size() == 4);
        TEST("F32 is not quantized", !info.is_quantized());

        info.type = forge::GGML_TYPE_F16;
        TEST("F16 element_size == 2", info.element_size() == 2);
        TEST("F16 is not quantized", !info.is_quantized());

        info.type = forge::GGML_TYPE_Q8_0;
        TEST("Q8_0 element_size == 1", info.element_size() == 1);
        TEST("Q8_0 is quantized", info.is_quantized());
        TEST("Q8_0 block_size == 32", info.block_size() == 32);
        TEST("Q8_0 block_bytes == 34", info.block_bytes() == 34);

        info.type = forge::GGML_TYPE_Q4_0;
        TEST("Q4_0 is quantized", info.is_quantized());
        TEST("Q4_0 block_size == 32", info.block_size() == 32);
        TEST("Q4_0 block_bytes == 18", info.block_bytes() == 18);

        info.type = forge::GGML_TYPE_Q2_K;
        TEST("Q2_K block_size == 256", info.block_size() == 256);
        TEST("Q2_K block_bytes == 28", info.block_bytes() == 28);
    }

    // Test GGUF magic constant
    TEST("GGUF_MAGIC correct", forge::GGUF_MAGIC == 0x46554747u);

    // Test ModelConfig read_config (without actual file - default values)
    {
        forge::ModelConfig cfg;
        TEST("Default layers == 24", cfg.n_layers == 24);
        TEST("Default n_embd == 1536", cfg.n_embd == 1536);
        TEST("Default n_heads == 16", cfg.n_heads == 16);
        TEST("Default n_kv_heads == 2", cfg.n_kv_heads == 2);
        TEST("Default n_ff == 4608", cfg.n_ff == 4608);
        TEST("Default n_vocab == 130560", cfg.n_vocab == 130560);
        TEST("Default max_seq_len == 131072", cfg.max_seq_len == 131072);
    }

    // Test quantized tensor size calculation
    {
        forge::GGUFTensorInfo info;
        info.type = forge::GGML_TYPE_Q8_0;
        info.n_elems = 4096;
        info.n_dims = 2;
        info.dims = {64, 64};
        
        uint64_t n_blocks = (info.n_elems + info.block_size() - 1) / info.block_size();
        TEST("Q8_0 n_blocks correct", n_blocks == 128);
        
        uint64_t tensor_size = n_blocks * info.block_bytes();
        TEST("Q8_0 tensor_size correct", tensor_size == 128 * 34);
    }

    // Test list_tensors on empty file
    {
        forge::GGUFFile gguf;
        auto names = gguf.list_tensors();
        TEST("Empty tensor list", names.empty());
    }

    // Test read_config on closed file (uses defaults)
    {
        forge::GGUFFile gguf;
        auto cfg = gguf.read_config();
        TEST("read_config returns defaults for unopened", cfg.n_layers == 24);
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, " 
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
