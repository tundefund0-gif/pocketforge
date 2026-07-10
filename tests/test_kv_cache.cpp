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
    std::cout << "=== Test: 3-Tier KV Cache (131K context) ===\n\n";

    forge::KVCacheConfig cfg;
    cfg.n_layers = 2;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.sink_size = 2;
    cfg.window_size = 16;
    cfg.pool_block = 4;
    cfg.max_context = 128;

    forge::KVCache cache(cfg);

    TEST("Initial size 0", cache.size() == 0);
    TEST("Memory usage > 0", cache.memory_usage() > 0);

    // Store sink tokens
    float k[4] = {1,2,3,4}, v[4] = {5,6,7,8};
    cache.store(0, 0, 0, k, v); // sink
    cache.store(0, 0, 1, k, v); // sink
    TEST("Sink stored", cache.size() >= 1);

    // Store window tokens (fill window of 16)
    for (uint32_t i = 2; i < 18; i++) {
        float kv[4] = {(float)i, (float)i, (float)i, (float)i};
        cache.store(0, 0, i, kv, kv);
    }
    TEST("Window filled", cache.size() >= 17);

    // Store more tokens (should trigger eviction + pooling)
    for (uint32_t i = 18; i < 30; i++) {
        float kv[4] = {(float)i, (float)i, (float)i, (float)i};
        cache.store(0, 0, i, kv, kv);
    }
    TEST("Pooling triggered", cache.size() >= 29);

    // Test attention scores
    float query[4] = {1,1,1,1};
    float scores[256] = {0};
    uint32_t count = cache.attention_scores(0, 0, query, scores);
    TEST("Scores returned", count > 0);
    TEST("Scores finite", std::isfinite(scores[0]));

    // Check that scores include sink + pooled + window
    // sink=2, pool=(window_evicted/4), window=16
    TEST("Scores include sink+pool+window", count >= 2 + 16); 

    // Test weighted sum
    float output[4] = {0};
    cache.attention_weighted_sum(0, 0, scores, output, count);
    TEST("Weighted sum finite",
         std::isfinite(output[0]) && std::isfinite(output[1]));

    // Test with different layer/head
    cache.store(1, 1, 0, k, v);
    count = cache.attention_scores(1, 1, query, scores);
    TEST("Diff layer scores", count >= 1);

    // Test clear
    cache.clear();
    TEST("Size after clear", cache.size() == 0);

    // Memory test for 131K config
    forge::KVCacheConfig big;
    big.n_layers = 24;
    big.n_kv_heads = 4;
    big.head_dim = 64;
    big.window_size = 8192;
    big.max_context = 131072;
    big.pool_block = 64;
    forge::KVCache big_cache(big);

    size_t big_mem = big_cache.memory_usage();
    std::cout << "  131K context KV cache: ~" << (big_mem / 1024 / 1024) << " MB\n";
    TEST("131K context under 150 MB", big_mem < 150 * 1024 * 1024);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
