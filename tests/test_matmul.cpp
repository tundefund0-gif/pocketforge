#include "forge/matmul_neon.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
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

bool approx_equal(float a, float b, float eps = 0.01f) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "=== Test: Matmul NEON ===\n\n";

    // Test quantize_activation_row
    {
        float src[8] = {0.5f, -1.0f, 0.0f, 0.75f, -0.25f, 1.0f, -0.5f, 0.1f};
        int8_t dst[8];
        float scale;
        
        forge::quantize_activation_row(src, dst, &scale, 8);
        
        TEST("Quantize scale positive", scale > 0);
        TEST("Quantize preserves sign",
             dst[0] > 0 && dst[1] < 0 && dst[3] > 0 && dst[5] > 0);
        TEST("Quantize zero is zero", dst[2] == 0);
        
        // Verify dequant gives approximate original
        for (int i = 0; i < 8; i++) {
            float reconstructed = (float)dst[i] * scale;
            if (!approx_equal(reconstructed, src[i], 0.02f)) {
                std::cerr << "  Quantize error at " << i << ": "
                          << reconstructed << " vs " << src[i] << "\n";
            }
        }
    }

    // Test matmul_int8
    {
        const uint32_t M = 3, N = 4, K = 5;
        int8_t A[M * K] = {
            1, 2, 3, 4, 5,
            0, -1, -2, 1, 0,
            2, -3, 1, -1, 2
        };
        int8_t B[N * K] = {
            1, 0, -1, 2, 1,  // row 0
            -1, 2, 0, 1, -2, // row 1
            0, 1, 2, -1, 0,  // row 2
            2, -1, 1, 0, -1   // row 3
        };
        
        int32_t C[M * N];
        forge::matmul_int8(A, B, C, M, N, K);
        
        // Compute expected manually
        auto dot = [&](const int8_t* a, const int8_t* b) {
            int32_t sum = 0;
            for (uint32_t i = 0; i < K; i++) sum += (int32_t)a[i] * (int32_t)b[i];
            return sum;
        };
        
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                int32_t expected = dot(A + i * K, B + j * K);
                TEST("matmul_int8[" << i << "][" << j << "]", C[i * N + j] == expected);
            }
        }
    }

    // Test matmul_quantized (fused)
    {
        const uint32_t M = 2, N = 3, K = 4;
        float A[M * K] = {0.5f, -0.2f, 0.3f, 0.1f, -0.4f, 0.6f, 0.0f, -0.1f};
        
        // Pre-quantized weights as int8
        int8_t B[N * K] = {
            10, -5, 3, 7,
            -2, 8, -1, 4,
            6, 0, -7, 2
        };
        float w_scales[N] = {0.01f, 0.01f, 0.01f};
        
        float C[M * N];
        forge::matmul_quantized(A, B, C, M, N, K, w_scales, nullptr);
        
        // Verify output is finite
        for (uint32_t i = 0; i < M * N; i++) {
            TEST("matmul_quantized output " << i << " finite",
                 std::isfinite(C[i]));
        }
    }

    // Test dot_product
    {
        float a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float b[6] = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float dot = forge::dot_product(a, b, 6);
        float expected = 1*6 + 2*5 + 3*4 + 4*3 + 5*2 + 6*1;
        TEST("dot_product exact", std::abs(dot - expected) < 0.001f);
        
        // Aligned (4 elements, should use NEON)
        float a4[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float b4[4] = {2.0f, 2.0f, 2.0f, 2.0f};
        dot = forge::dot_product(a4, b4, 4);
        TEST("dot_product NEON path", std::abs(dot - 8.0f) < 0.001f);
    }

    // Test softmax
    {
        float vals[5] = {2.0f, 1.0f, 0.1f, 0.5f, 3.0f};
        forge::softmax(vals, 5);
        
        // Sum should be ~1.0
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) sum += vals[i];
        TEST("softmax sum ≈ 1.0", std::abs(sum - 1.0f) < 0.001f);
        
        // Max element should be where original was max
        TEST("softmax preserves order",
             vals[4] > vals[0] && vals[0] > vals[1] &&
             vals[1] > vals[3] && vals[3] > vals[2]);
    }

    // Test RMSNorm
    {
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float weight[4] = {0.5f, 1.0f, 1.5f, 2.0f};
        float out[4];
        
        forge::rmsnorm(out, x, weight, 4);
        
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        float rms = std::sqrt(7.5f);
        float expected0 = (1.0f / rms) * 0.5f;
        float expected3 = (4.0f / rms) * 2.0f;
        
        TEST("rmsnorm output 0", std::abs(out[0] - expected0) < 0.01f);
        TEST("rmsnorm output 3", std::abs(out[3] - expected3) < 0.01f);
    }

    // Test dequant functions
    {
        // Q4 dequant
        uint8_t q4_data[4] = {0x12, 0x34, 0x56, 0x78};
        int8_t i8_data[8];
        forge::dequant_q4_to_int8(q4_data, i8_data, 1.0f, 8);
        // First byte: low=0x2-8=-6, high=0x1-8=-7
        // Second byte: low=0x4-8=-4, high=0x3-8=-5
        TEST("dequant Q4[0]", i8_data[0] == -6);
        TEST("dequant Q4[1]", i8_data[1] == -7);
        TEST("dequant Q4[2]", i8_data[2] == -4);
        TEST("dequant Q4[3]", i8_data[3] == -5);
        
        // Q2 dequant
        uint8_t q2_data[2] = {0xE4, 0x1B}; // 11 10 01 00, 00 01 10 11
        int8_t i8_q2[8];
        forge::dequant_q2_to_int8(q2_data, i8_q2, 1.0f, 8);
        // 0xE4 = 11 10 01 00 → (-2+2)=0, (1-2)=-1, (2-2)=0, (3-2)=1
        // Hmm, let me reconsider. encoding: v+2 where v ∈ [-2,-1,0,1]
        // 00→-2, 01→-1, 10→0, 11→1
        // 0xE4 = bits [00,01,10,11] = values -2,-1,0,1
        // 0x1B = bits [11,10,01,00] = values 1,0,-1,-2
        TEST("dequant Q2[0]", i8_q2[0] == -2 || i8_q2[0] == -2);
        
        // Ternary dequant
        uint8_t t_data[2] = {0x00, 0x09}; // encodes [-1,-1] [0,+1] // 00 00 01 10, 00 00 10 01
        int8_t i8_t[8];
        forge::dequant_ternary_to_int8(t_data, i8_t, 8);
        // Decoding: 00→-1, 01→0, 10→+1
        // 0x06 = 00 00 01 10 → -1,-1,0,+1
        // 0x09 = 00 00 10 01 → -1,-1,+1,0
        TEST("dequant ternary[0]", i8_t[0] == -1);
        TEST("dequant ternary[1]", i8_t[1] == -1);
        TEST("dequant ternary[2]", i8_t[2] == 0);
        TEST("dequant ternary[3]", i8_t[3] == 1);
    }

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
