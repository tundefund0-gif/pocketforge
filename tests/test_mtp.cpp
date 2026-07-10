#include "forge/mtp.hpp"
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
    std::cout << "=== Test: MTP Heads (Low-Memory) ===\n\n";

    forge::MTPConfig cfg;
    cfg.n_mtp_heads = 4;
    cfg.n_embd = 64;
    cfg.n_vocab = 128;
    cfg.mtp_ff_dim = 64;

    forge::MTPHeads mtp(cfg);

    TEST("MTP heads initialized", mtp.memory_usage() > 0);

    // Hidden state
    std::vector<float> hidden(cfg.n_embd);
    for (uint32_t i = 0; i < cfg.n_embd; i++) {
        hidden[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Forward without embeddings (placeholder mode)
    auto logits = mtp.forward(hidden.data(), {});

    TEST("MTP forward returns correct head count",
         logits.size() == cfg.n_mtp_heads);
    TEST("MTP head 0 logits non-empty", !logits[0].empty());
    TEST("MTP head 0 logits size matches vocab",
         logits[0].size() == cfg.n_vocab);

    for (uint32_t h = 0; h < cfg.n_mtp_heads; h++) {
        bool all_finite = true;
        for (float l : logits[h]) {
            if (!std::isfinite(l)) { all_finite = false; break; }
        }
        TEST("MTP head " << h << " logits finite", all_finite);
    }

    // Test argmax
    {
        float tl[5] = {-1.0f, 3.0f, 0.5f, -0.2f, 2.0f};
        TEST("argmax at index 1", forge::MTPHeads::argmax(tl, 5) == 1);
    }

    // Test speculative_accept
    {
        std::vector<std::vector<float>> tl(4, std::vector<float>(cfg.n_vocab, -1.0f));
        tl[0][5] = 10.0f;
        tl[1][5] = 10.0f;
        tl[2][7] = 10.0f;
        tl[3][5] = 10.0f;
        int accepted = mtp.speculative_accept(5, tl);
        TEST("Speculative accept stops at mismatch", accepted == 2);
    }

    // Test with tied embeddings (flat vector)
    {
        std::vector<float> tied(cfg.n_vocab * cfg.n_embd, 0.0f);
        for (uint32_t v = 0; v < cfg.n_vocab; v++) {
            for (uint32_t i = 0; i < cfg.n_embd; i++) {
                tied[v * cfg.n_embd + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            }
        }
        auto lt = mtp.forward(hidden.data(), tied);
        TEST("MTP with tied embeddings works", !lt[0].empty());
    }

    // With tied embeddings, different hidden gives different logits
    {
        std::vector<float> tied(cfg.n_vocab * cfg.n_embd, 0.0f);
        for (uint32_t v = 0; v < cfg.n_vocab; v++) {
            for (uint32_t i = 0; i < cfg.n_embd; i++) {
                tied[v * cfg.n_embd + i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            }
        }
        std::vector<float> hidden2(cfg.n_embd, 0.0f);
        auto l1 = mtp.forward(hidden.data(), tied);
        auto l2 = mtp.forward(hidden2.data(), tied);
        bool different = false;
        for (uint32_t v = 0; v < cfg.n_vocab && !different; v++) {
            if (std::abs(l1[0][v] - l2[0][v]) > 0.01f) different = true;
        }
        TEST("Different hidden => different logits (tied emb)", different);
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
