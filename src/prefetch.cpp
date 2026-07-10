#include "forge/prefetch.hpp"
#include <cassert>

namespace forge {

PrefetchPipeline::PrefetchPipeline(WeightLoader* loader, uint32_t n_layers, uint32_t n_matrices_per_layer)
    : loader_(loader)
    , n_layers_(n_layers)
    , n_matrices_(n_matrices_per_layer)
{
    buffer_.resize(kMaxBufferedMatrices);
}

PrefetchPipeline::~PrefetchPipeline() {
    stop();
}

void PrefetchPipeline::start() {
    if (running_) return;
    running_ = true;
    worker_ = std::thread(&PrefetchPipeline::prefetch_worker, this);
}

void PrefetchPipeline::stop() {
    if (!running_) return;
    running_ = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_cv_.notify_one();
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void PrefetchPipeline::prefetch_layer(uint32_t layer_id) {
    if (layer_id >= n_layers_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    // Queue all matrices for this layer
    for (uint32_t m = 0; m < n_matrices_; m++) {
        request_queue_.push({layer_id, m});
    }
    producer_cv_.notify_one();
}

DecompressedBlock PrefetchPipeline::wait_for_block(uint32_t layer_id, uint32_t matrix_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
        for (const auto& slot : buffer_) {
            if (slot.occupied && slot.ready &&
                slot.layer_id == layer_id && slot.matrix_id == matrix_id) {
                return true;
            }
        }
        return false;
    });

    DecompressedBlock result;
    for (auto& slot : buffer_) {
        if (slot.occupied && slot.ready &&
            slot.layer_id == layer_id && slot.matrix_id == matrix_id) {
            result.layer_id = slot.layer_id;
            result.matrix_id = slot.matrix_id;
            result.data = std::move(slot.data);
            result.scale = slot.scale;
            result.ready = true;
            slot.occupied = false;
            slot.ready = false;
            break;
        }
    }
    return result;
}

bool PrefetchPipeline::try_get_block(uint32_t layer_id, uint32_t matrix_id,
                                      DecompressedBlock* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& slot : buffer_) {
        if (slot.occupied && slot.ready &&
            slot.layer_id == layer_id && slot.matrix_id == matrix_id) {
            out->layer_id = slot.layer_id;
            out->matrix_id = slot.matrix_id;
            out->data = std::move(slot.data);
            out->scale = slot.scale;
            out->ready = true;
            slot.occupied = false;
            slot.ready = false;
            return true;
        }
    }
    return false;
}

void PrefetchPipeline::release_block(uint32_t layer_id, uint32_t matrix_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& slot : buffer_) {
        if (slot.layer_id == layer_id && slot.matrix_id == matrix_id) {
            slot.occupied = false;
            slot.ready = false;
            slot.data.clear();
            slot.data.shrink_to_fit();
            break;
        }
    }
    producer_cv_.notify_one();
}

size_t PrefetchPipeline::memory_usage() const {
    size_t total = 0;
    for (const auto& slot : buffer_) {
        total += slot.data.capacity();
    }
    return total;
}

void PrefetchPipeline::prefetch_worker() {
    while (running_) {
        std::pair<uint32_t, uint32_t> request;
        bool has_work = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            producer_cv_.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                return !request_queue_.empty() || !running_;
            });

            if (!running_) return;
            if (request_queue_.empty()) continue;

            request = request_queue_.front();
            request_queue_.pop();
            has_work = true;
        }

        if (!has_work) continue;

        // Find a free buffer slot
        BufferSlot* slot = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& s : buffer_) {
                if (!s.occupied) {
                    slot = &s;
                    break;
                }
            }
        }

        if (!slot) continue; // no free slot, skip this request

        // Load and decompress
        float scale = 1.0f;
        auto data = loader_->load_block(request.first, request.second, &scale);

        if (!data.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            slot->layer_id = request.first;
            slot->matrix_id = request.second;
            slot->data = std::move(data);
            slot->scale = scale;
            slot->occupied = true;
            slot->ready = true;
        }

        cv_.notify_all();
    }
}

} // namespace forge
