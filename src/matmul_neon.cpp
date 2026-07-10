#include "forge/matmul_neon.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace forge {

// ============================================================
//  Quantize activations to int8 (per row)
// ============================================================
void quantize_activation_row(
    const float* src, int8_t* dst, float* scale_out, uint32_t n
) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float absv = std::abs(src[i]);
        if (absv > amax) amax = absv;
    }
    float scale = (amax == 0.0f) ? 1.0f : 127.0f / amax;
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = (int8_t)std::round(src[i] * scale);
    }
    *scale_out = (amax == 0.0f) ? 1.0f : amax / 127.0f;
}

// ============================================================
//  Dequantize Q4 to int8
// ============================================================
void dequant_q4_to_int8(const uint8_t* q4_data, int8_t* i8_data, float scale, uint32_t n) {
    // Q4: each byte has 2 values: low nibble = val0, high nibble = val1
    // Values are in [-8, 7]
    (void)scale;
    for (uint32_t i = 0; i < n; i += 2) {
        uint8_t packed = q4_data[i / 2];
        int8_t v0 = (int8_t)(packed & 0x0F) - 8;
        int8_t v1 = (int8_t)((packed >> 4) & 0x0F) - 8;
        i8_data[i] = v0;
        if (i + 1 < n) i8_data[i + 1] = v1;
    }
}

// ============================================================
//  Dequantize Q2 to int8
// ============================================================
void dequant_q2_to_int8(const uint8_t* q2_data, int8_t* i8_data, float scale, uint32_t n) {
    // Q2: each byte has 4 values
    // Values are in [-2, 1]
    (void)scale;
    for (uint32_t i = 0; i < n; i += 4) {
        uint8_t packed = q2_data[i / 4];
        i8_data[i]     = (int8_t)((packed & 0x03)     ) - 2;
        if (i + 1 < n) i8_data[i + 1] = (int8_t)(((packed >> 2) & 0x03)) - 2;
        if (i + 2 < n) i8_data[i + 2] = (int8_t)(((packed >> 4) & 0x03)) - 2;
        if (i + 3 < n) i8_data[i + 3] = (int8_t)(((packed >> 6) & 0x03)) - 2;
    }
}

// ============================================================
//  Dequantize ternary (Q1.5) to int8
// ============================================================
void dequant_ternary_to_int8(const uint8_t* t_data, int8_t* i8_data, uint32_t n) {
    // Ternary: 2 values per byte
    // Encoded: 00 = -1, 01 = 0, 10 = +1, 11 = unused
    static const int8_t ternary_decode[4] = {-1, 0, 1, 0};
    for (uint32_t i = 0; i < n; i += 2) {
        uint8_t packed = t_data[i / 2];
        i8_data[i]     = ternary_decode[packed & 0x03];
        if (i + 1 < n) i8_data[i + 1] = ternary_decode[(packed >> 2) & 0x03];
    }
}

// ============================================================
//  int8 matmul with NEON
// ============================================================
void matmul_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    uint32_t M, uint32_t N, uint32_t K
) {
#ifdef __ARM_NEON
    #ifdef __aarch64__
    // NEON path using SDOT on aarch64
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            const int8_t* row_a = A + i * K;
            const int8_t* row_b = B + j * K;

            int32x4_t acc = vdupq_n_s32(0);
            uint32_t k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t va = vld1q_s8(row_a + k);
                int8x16_t vb = vld1q_s8(row_b + k);
                acc = vdotq_s32(acc, va, vb);
            }
            int32_t sum = vaddvq_s32(acc);
            for (; k < K; k++) {
                sum += (int32_t)row_a[k] * (int32_t)row_b[k];
            }
            C[i * N + j] = sum;
        }
    }
    #else
    // ARM32 NEON path: pairwise add
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            const int8_t* row_a = A + i * K;
            const int8_t* row_b = B + j * K;

            int32x2_t acc = vdup_n_s32(0);
            uint32_t k = 0;
            for (; k + 8 <= K; k += 8) {
                int8x8_t va = vld1_s8(row_a + k);
                int8x8_t vb = vld1_s8(row_b + k);
                int16x8_t prod = vmull_s8(va, vb);
                acc = vpadal_s16(acc, vget_low_s16(prod));
                acc = vpadal_s16(acc, vget_high_s16(prod));
            }
            int32_t sum = vget_lane_s32(acc, 0) + vget_lane_s32(acc, 1);
            for (; k < K; k++) {
                sum += (int32_t)row_a[k] * (int32_t)row_b[k];
            }
            C[i * N + j] = sum;
        }
    }
    #endif
#else
    // Scalar fallback
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            const int8_t* row_a = A + i * K;
            const int8_t* row_b = B + j * K;
            for (uint32_t k = 0; k < K; k++) {
                sum += (int32_t)row_a[k] * (int32_t)row_b[k];
            }
            C[i * N + j] = sum;
        }
    }
#endif
}

// ============================================================
//  Fused quantized matmul
// ============================================================
void matmul_quantized(
    const float* A, const int8_t* B, float* C,
    uint32_t M, uint32_t N, uint32_t K,
    const float* weight_scales, const float* act_scales
) {
    // Quantize activations per-row
    std::vector<int8_t> A_q(M * K);
    std::vector<float> a_scales(M);
    
    if (act_scales) {
        for (uint32_t i = 0; i < M; i++) {
            a_scales[i] = act_scales[i];
            for (uint32_t k = 0; k < K; k++) {
                A_q[i * K + k] = (int8_t)std::round(A[i * K + k] / act_scales[i]);
            }
        }
    } else {
        for (uint32_t i = 0; i < M; i++) {
            quantize_activation_row(A + i * K, A_q.data() + i * K, &a_scales[i], K);
        }
    }

    // int8 matmul
    std::vector<int32_t> C_int(M * N);
    matmul_int8(A_q.data(), B, C_int.data(), M, N, K);

    // Apply scales
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            C[i * N + j] = (float)C_int[i * N + j] * a_scales[i] * weight_scales[j];
        }
    }
}

// ============================================================
//  Dot product (scalar)
// ============================================================
float dot_product(const float* a, const float* b, uint32_t n) {
#ifdef __ARM_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

// ============================================================
//  Softmax
// ============================================================
void softmax(float* vals, uint32_t n) {
    float max_val = vals[0];
    for (uint32_t i = 1; i < n; i++) {
        if (vals[i] > max_val) max_val = vals[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        vals[i] = std::exp(vals[i] - max_val);
        sum += vals[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (uint32_t i = 0; i < n; i++) {
        vals[i] *= inv_sum;
    }
}

// ============================================================
//  RMSNorm
// ============================================================
void rmsnorm(float* out, const float* x, const float* weight, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = std::sqrt(ss / n + 1e-6f);
    float inv_rms = 1.0f / rms;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

} // namespace forge
