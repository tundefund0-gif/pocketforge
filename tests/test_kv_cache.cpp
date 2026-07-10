#include "forge/kv_cache.hpp"
#include <iostream>
#include <cmath>
#include <cstring>

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
    std::cout << "=== Test: KV Cache (streaming int8, 16K+ context) ===\n\n";

    forge::KVCacheConfig cfg;
    cfg.n_layers = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.max_positions = 256;
    cfg.n_sinks = 2;

    forge::KVCache cache(cfg);

    TEST("Initial size is 0", cache.size() == 0);
    TEST("Capacity set", cache.capacity() == 256);
    TEST("Memory usage > 0 for storage", cache.memory_usage() > 0);

    // Store some KV entries
    float key[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float val[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    cache.store(0, 0, 0, key, val);
    TEST("Size after 1 store", cache.size() >= 1);

    // Store second position
    float key2[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    float val2[4] = {13.0f, 14.0f, 15.0f, 16.0f};
    cache.store(0, 0, 1, key2, val2);
    TEST("Size after 2 stores", cache.size() >= 2);

    // Test attention_scores
    float query[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float scores[256] = {0};
    uint32_t count = 0;

    cache.attention_scores(0, 0, query, scores, &count);
    TEST("Attention scores count", count == 2);
    TEST("Scores are finite", std::isfinite(scores[0]) && std::isfinite(scores[1]));

    // Test weighted_sum
    float output[256] = {0};
    cache.attention_weighted_sum(0, 0, scores, output, count);
    TEST("Weighted sum output finite",
         std::isfinite(output[0]) && std::isfinite(output[1]) &&
         std::isfinite(output[2]) && std::isfinite(output[3]));
    TEST("Weighted sum non-zero",
         std::abs(output[0]) > 0.001f || std::abs(output[1]) > 0.001f);

    // Different layer/head
    cache.store(1, 1, 0, key, val);
    cache.attention_scores(1, 1, query, scores, &count);
    TEST("Different layer/head scores count", count == 1);
    TEST("Different layer/head scores finite", std::isfinite(scores[0]));

    // Multiple layers
    uint32_t total_positions = 0;
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        for (uint32_t h = 0; h < cfg.n_kv_heads; h++) {
            float k[4] = {(float)(l * 10 + h), (float)(l * 10 + h + 1), 0, 0};
            float v[4] = {(float)(l * 10 + h + 100), (float)(l * 10 + h + 101), 0, 0};
            cache.store(l, h, 0, k, v);
            total_positions++;
        }
    }
    cache.attention_scores(3, 1, query, scores, &count);
    TEST("Multi-layer store, layer 3 head 1 works", count >= 1);

    // Stress: many positions
    for (uint32_t i = 0; i < 100; i++) {
        float k[4] = {(float)i, (float)i, (float)i, (float)i};
        float v[4] = {(float)i, (float)i, (float)i, (float)i};
        cache.store(0, 0, i + 2, k, v);
    }
    TEST("Many positions stored", cache.size() > 50);

    cache.attention_scores(0, 0, query, scores, &count);
    TEST("Many positions scores count", count > 50);
    TEST("Many positions scores are finite",
         std::isfinite(scores[0]) && std::isfinite(scores[count-1]));

    cache.attention_weighted_sum(0, 0, scores, output, count);
    TEST("Many positions weighted sum", std::isfinite(output[0]));

    // Test clear
    cache.clear();
    TEST("Size after clear", cache.size() == 0);

    // Test max positions
    forge::KVCacheConfig big_cfg;
    big_cfg.n_layers = 24;
    big_cfg.n_kv_heads = 4;
    big_cfg.head_dim = 64;
    big_cfg.max_positions = 16384;
    forge::KVCache big_cache(big_cfg);
    TEST("16K context memory under 250 MB", big_cache.memory_usage() < 250 * 1024 * 1024);

    size_t mb = big_cache.memory_usage() / (1024 * 1024);
    std::cout << "  16K context KV cache: ~" << mb << " MB\n";

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
