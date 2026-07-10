#ifndef FORGE_MATMUL_NEON_HPP
#define FORGE_MATMUL_NEON_HPP

#include "types.hpp"
#include <cstdint>
#include <vector>

namespace forge {

// ============================================================
//  NEON-optimized integer matrix operations
// ============================================================
//
// All operations use pure integer arithmetic where possible.
// For 32-bit ARM, we target armv7a + NEON.
// For 64-bit ARM, we use advanced SIMD (same intrinsics).
//

// Quantize activations to int8 (per-row)
void quantize_activation_row(
    const float* src,
    int8_t* dst,
    float* scale_out,
    uint32_t n
);

// Dequantize weights from Q4 packed to int8
void dequant_q4_to_int8(
    const uint8_t* q4_data,
    int8_t* i8_data,
    float scale,
    uint32_t n
);

// Dequantize weights from Q2 packed to int8
void dequant_q2_to_int8(
    const uint8_t* q2_data,
    int8_t* i8_data,
    float scale,
    uint32_t n
);

// Dequantize ternary (Q1.5) packed to int8
void dequant_ternary_to_int8(
    const uint8_t* t_data,
    int8_t* i8_data,
    uint32_t n
);

// ============================================================
//  Matrix multiply: C = A * B^T
//  A: int8 [M x K], B: int8 [N x K], C: int32 [M x N]
//  Uses NEON dot product (SDOT) on aarch64 or smlad on arm32
// ============================================================
void matmul_int8(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    uint32_t M,
    uint32_t N,
    uint32_t K
);

// ============================================================
//  Fused: quantize A → matmul → apply scales → output float
//  A: float [M x K], B: int8 [N x K], C: float [M x N]
// ============================================================
void matmul_quantized(
    const float* A,
    const int8_t* B,
    float* C,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const float* weight_scales,  // [N] per-row weight scales
    const float* act_scales      // [M] per-row activation scales (optional, computed if null)
);

// ============================================================
//  Scalar dot product (for attention scores)
// ============================================================
float dot_product(const float* a, const float* b, uint32_t n);

// ============================================================
//  Softmax (in-place, on float vector)
// ============================================================
void softmax(float* vals, uint32_t n);

// ============================================================
//  RMSNorm (int32 accumulator, float output)
// ============================================================
void rmsnorm(float* out, const float* x, const float* weight, uint32_t n);

} // namespace forge
#endif // FORGE_MATMUL_NEON_HPP
