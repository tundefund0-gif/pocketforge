#include "forge/layer_skip.hpp"
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
    std::cout << "=== Test: Layer Skip ===\n\n";

    forge::ModelConfig cfg;
    cfg.n_layers = 12;
    cfg.n_embd = 8;
    cfg.skip_threshold = 0.5f;
    cfg.skip_interval = 4;

    forge::LayerSkipController skipper(cfg);

    TEST("Initial skipped count 0", skipper.total_skipped() == 0);
    TEST("Initial not anchor (t=0, interval=4)", !skipper.is_anchor_token());

    // Test new_token tracking
    skipper.new_token();
    TEST("Token 1 not anchor", !skipper.is_anchor_token());

    skipper.new_token();
    TEST("Token 2 not anchor", !skipper.is_anchor_token());

    skipper.new_token();
    TEST("Token 3 not anchor", !skipper.is_anchor_token());

    skipper.new_token();
    TEST("Token 4 is anchor (interval=4)", skipper.is_anchor_token());

    // Test skip decision
    // First layer with identical hidden states → should skip
    std::vector<float> hidden1(cfg.n_embd, 1.0f);
    std::vector<float> hidden2(cfg.n_embd, 1.0f); // identical
    
    skipper.reset();
    
    // Layer 0: should NOT skip (first 2 layers are protected)
    bool skip = skipper.should_skip_ffn(0, hidden1.data(), cfg.n_embd);
    TEST("Layer 0 not skipped (protected)", !skip);

    // Layer 1: should NOT skip (protected)
    skip = skipper.should_skip_ffn(1, hidden2.data(), cfg.n_embd);
    TEST("Layer 1 not skipped (protected)", !skip);

    // Layer 5 (middle): prime cache, then check skip
    skipper.should_skip_ffn(5, hidden1.data(), cfg.n_embd); // prime
    skip = skipper.should_skip_ffn(5, hidden1.data(), cfg.n_embd);
    TEST("Layer 5 identical after priming → skip", skip == true);

    // Layer 10 (near end): should NOT skip (last 2 layers protected)
    skip = skipper.should_skip_ffn(10, hidden1.data(), cfg.n_embd);
    TEST("Layer 10 not skipped (protected)", !skip);

    // Test with significantly different hidden states
    std::vector<float> hidden_diff(cfg.n_embd, 100.0f); // very different
    skip = skipper.should_skip_ffn(5, hidden_diff.data(), cfg.n_embd);
    TEST("Layer 5 very different → not skip", !skip);

    // Test skip count tracking
    skipper.record_layer_output(5, hidden1.data(), cfg.n_embd);
    TEST("Skipped count starts at 0", skipper.total_skipped() == 0);

    // Record a non-skipped layer
    skipper.record_layer_output(0, hidden1.data(), cfg.n_embd);

    // Test reset
    skipper.reset();
    TEST("After reset, total_skipped is 0", skipper.total_skipped() == 0);

    // Test threshold adjustment
    skipper.set_threshold(0.01f);
    TEST("Threshold set", skipper.threshold() < 0.1f);

    // Test that threshold affects decisions
    skip = skipper.should_skip_ffn(5, hidden1.data(), cfg.n_embd);
    TEST("Lower threshold = fewer skips (layer5 may not skip)", skip == false);

    skipper.set_threshold(100.0f); // very high → skip everything
    skip = skipper.should_skip_ffn(5, hidden1.data(), cfg.n_embd);
    TEST("Very high threshold → skip (middle layer)", skip == true);

    // Test skip mask
    const auto& mask = skipper.skip_mask();
    TEST("Skip mask has correct size", mask.size() == cfg.n_layers);

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
