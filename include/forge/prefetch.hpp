#ifndef FORGE_PREFETCH_HPP
#define FORGE_PREFETCH_HPP

#include "types.hpp"
#include "quant_format.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <functional>
#include <vector>

namespace forge {

// ============================================================
//  Async prefetch pipeline for weight decompression
// ============================================================
//
// While the main thread computes layer L, a background thread
// decompresses layer L+1 weight blocks into a double buffer.
//
// Pipeline stages:
//   [Main] Compute(L, M) → signal prefetcher
//   [Prefetcher] Decompress(L+1, M+1..) → signal main
//

struct DecompressedBlock {
    uint32_t layer_id;
    uint32_t matrix_id;
    std::vector<int8_t> data;
    float scale;
    bool ready = false;
};

class PrefetchPipeline {
public:
    PrefetchPipeline(WeightLoader* loader, uint32_t n_layers, uint32_t n_matrices_per_layer);
    ~PrefetchPipeline();

    // Start prefetch thread
    void start();

    // Stop prefetch thread
    void stop();

    // Request next layer to be prefetched
    void prefetch_layer(uint32_t layer_id);

    // Wait for a specific block to be ready and get it
    // Blocks calling thread until data is available
    DecompressedBlock wait_for_block(uint32_t layer_id, uint32_t matrix_id);

    // Get a block if ready (non-blocking)
    bool try_get_block(uint32_t layer_id, uint32_t matrix_id, DecompressedBlock* out);

    // Signal that we've finished computing with a block (buffer can be reused)
    void release_block(uint32_t layer_id, uint32_t matrix_id);

    // Memory usage
    size_t memory_usage() const;

private:
    void prefetch_worker();

    WeightLoader* loader_;
    uint32_t n_layers_;
    uint32_t n_matrices_;

    std::thread worker_;
    std::atomic<bool> running_{false};

    // Double-buffered: we can have up to 2 layers in-flight
    static constexpr uint32_t kMaxBufferedLayers = 2;
    static constexpr uint32_t kMaxBufferedMatrices = 14; // 2 layers × 7 matrices

    struct BufferSlot {
        uint32_t layer_id;
        uint32_t matrix_id;
        std::vector<int8_t> data;
        float scale;
        bool occupied = false;
        bool ready = false;
    };

    std::vector<BufferSlot> buffer_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable producer_cv_;

    std::queue<std::pair<uint32_t, uint32_t>> request_queue_;
};

} // namespace forge
#endif // FORGE_PREFETCH_HPP
