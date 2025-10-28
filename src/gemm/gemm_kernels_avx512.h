/**
 * @file gemm_kernels_avx512.h
 * @brief AVX-512 GEMM Micro-kernels (32-row variants)
 *
 * @details
 * AVX-512 kernels that double the capacity of AVX2:
 * - 32×16 kernel: 32 rows × 16 columns (vs 16×8 for AVX2)
 * - 32×12 kernel: 32 rows × 12 columns (vs 16×6 for AVX2)
 *
 * Register usage:
 * - 32 zmm accumulators (2 per column for lo/hi 16 rows)
 * - 2 zmm for A loads (a_lo, a_hi)
 * - Broadcast B elements as needed
 *
 * Performance characteristics:
 * - ~2× throughput vs AVX2 (16 floats per register vs 8)
 * - Requires Skylake-X, Ice Lake, or newer
 * - Zen4+ on AMD side
 *
 * @author VectorFFT Team
 * @date 2025
 */

#ifndef GEMM_KERNELS_AVX512_H
#define GEMM_KERNELS_AVX512_H

#include "gemm_simd_ops.h"

#ifdef __AVX512F__

//==============================================================================
// HELPER: Load columns from column-major temp buffer (AVX-512)
//==============================================================================

static inline __m512 load_cols_from_temp_avx512(
    const float *temp,
    size_t stride,
    size_t r,
    size_t n)
{
    alignas(64) float lane[16] = {0};
    for (size_t j = 0; j < n && j < 16; ++j)
        lane[j] = temp[j * stride + r];
    return GEMM_LOAD_PS_AVX512(lane);
}

//==============================================================================
// 32×16 KERNEL (ADD): C += A*B
//==============================================================================

/**
 * @brief 32×16 micro-kernel: C += A*B (AVX-512)
 *
 * @details
 * Computes 32 rows × 16 columns of C, accumulating into existing values.
 *
 * Register Usage:
 * - 32 accumulators (16 lo + 16 hi) for 32×16 tile
 * - 2 for A loads (a_lo, a_hi)
 * - 16 for B broadcasts
 * - Total: ~50 zmm registers used
 *   → Some spilling expected, but hot path still efficient
 *
 * Memory Layout:
 * - Ap: Packed A panel [K × 32] contiguous, aligned 64B
 * - Bp: Packed B panel [K × 16] contiguous, aligned 64B
 * - C:  Row-major [M × ldc] unaligned
 *
 * @param[in,out] c Output matrix C (row-major)
 * @param[in] ldc Leading dimension of C
 * @param[in] Ap Packed A panel (col-major, ld=32)
 * @param[in] Bp Packed B panel (row-major, ld=16)
 * @param[in] Kblk K dimension of this block
 * @param[in] m Number of active rows (≤32)
 * @param[in] n Number of active columns (≤16)
 * @param[in] mask AVX-512 mask for tail columns
 */
static inline void gemm_32x16_panel_avx512_add(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk,
    size_t m,
    size_t n,
    __mmask16 mask)
{
    LINALG_ASSUME_ALIGNED(Ap, 64);
    LINALG_ASSUME_ALIGNED(Bp, 64);

    // 32 accumulators: 2 zmm per column (lo/hi 16 rows)
    __m512 acc_lo[16], acc_hi[16];
    for (int i = 0; i < 16; ++i) {
        acc_lo[i] = GEMM_VEC_ZERO_PS_AVX512();
        acc_hi[i] = GEMM_VEC_ZERO_PS_AVX512();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    
    // Prefetch output C locations
    for (size_t r = 0; r < m; r += 8)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    
    // Main K loop (unrolled by 8)
    for (size_t k = 0; k < Kblk; k += 8) {
        // Prefetch upcoming data
        if (do_pf) {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 16);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 16);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 32);
#endif
        }

        // Process 8 K iterations
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk) break;

            // Load 32 A elements (2 zmm registers)
            __m512 a_lo = GEMM_LOAD_PS_AVX512(Ap + kk * 32);
            __m512 a_hi = GEMM_LOAD_PS_AVX512(Ap + kk * 32 + 16);
            
            // Load 16 B elements (1 row)
            const float *b_row = Bp + kk * 16;
            
            // Broadcast each B element and FMA with A
            // Process in groups of 4 for better instruction scheduling
            __m512 b0 = GEMM_BROADCAST_SS_AVX512(b_row + 0);
            __m512 b1 = GEMM_BROADCAST_SS_AVX512(b_row + 1);
            __m512 b2 = GEMM_BROADCAST_SS_AVX512(b_row + 2);
            __m512 b3 = GEMM_BROADCAST_SS_AVX512(b_row + 3);
            
            acc_lo[0] = GEMM_FMADD_PS_AVX512(a_lo, b0, acc_lo[0]);
            acc_hi[0] = GEMM_FMADD_PS_AVX512(a_hi, b0, acc_hi[0]);
            acc_lo[1] = GEMM_FMADD_PS_AVX512(a_lo, b1, acc_lo[1]);
            acc_hi[1] = GEMM_FMADD_PS_AVX512(a_hi, b1, acc_hi[1]);
            acc_lo[2] = GEMM_FMADD_PS_AVX512(a_lo, b2, acc_lo[2]);
            acc_hi[2] = GEMM_FMADD_PS_AVX512(a_hi, b2, acc_hi[2]);
            acc_lo[3] = GEMM_FMADD_PS_AVX512(a_lo, b3, acc_lo[3]);
            acc_hi[3] = GEMM_FMADD_PS_AVX512(a_hi, b3, acc_hi[3]);
            
            __m512 b4 = GEMM_BROADCAST_SS_AVX512(b_row + 4);
            __m512 b5 = GEMM_BROADCAST_SS_AVX512(b_row + 5);
            __m512 b6 = GEMM_BROADCAST_SS_AVX512(b_row + 6);
            __m512 b7 = GEMM_BROADCAST_SS_AVX512(b_row + 7);
            
            acc_lo[4] = GEMM_FMADD_PS_AVX512(a_lo, b4, acc_lo[4]);
            acc_hi[4] = GEMM_FMADD_PS_AVX512(a_hi, b4, acc_hi[4]);
            acc_lo[5] = GEMM_FMADD_PS_AVX512(a_lo, b5, acc_lo[5]);
            acc_hi[5] = GEMM_FMADD_PS_AVX512(a_hi, b5, acc_hi[5]);
            acc_lo[6] = GEMM_FMADD_PS_AVX512(a_lo, b6, acc_lo[6]);
            acc_hi[6] = GEMM_FMADD_PS_AVX512(a_hi, b6, acc_hi[6]);
            acc_lo[7] = GEMM_FMADD_PS_AVX512(a_lo, b7, acc_lo[7]);
            acc_hi[7] = GEMM_FMADD_PS_AVX512(a_hi, b7, acc_hi[7]);
            
            __m512 b8 = GEMM_BROADCAST_SS_AVX512(b_row + 8);
            __m512 b9 = GEMM_BROADCAST_SS_AVX512(b_row + 9);
            __m512 b10 = GEMM_BROADCAST_SS_AVX512(b_row + 10);
            __m512 b11 = GEMM_BROADCAST_SS_AVX512(b_row + 11);
            
            acc_lo[8] = GEMM_FMADD_PS_AVX512(a_lo, b8, acc_lo[8]);
            acc_hi[8] = GEMM_FMADD_PS_AVX512(a_hi, b8, acc_hi[8]);
            acc_lo[9] = GEMM_FMADD_PS_AVX512(a_lo, b9, acc_lo[9]);
            acc_hi[9] = GEMM_FMADD_PS_AVX512(a_hi, b9, acc_hi[9]);
            acc_lo[10] = GEMM_FMADD_PS_AVX512(a_lo, b10, acc_lo[10]);
            acc_hi[10] = GEMM_FMADD_PS_AVX512(a_hi, b10, acc_hi[10]);
            acc_lo[11] = GEMM_FMADD_PS_AVX512(a_lo, b11, acc_lo[11]);
            acc_hi[11] = GEMM_FMADD_PS_AVX512(a_hi, b11, acc_hi[11]);
            
            __m512 b12 = GEMM_BROADCAST_SS_AVX512(b_row + 12);
            __m512 b13 = GEMM_BROADCAST_SS_AVX512(b_row + 13);
            __m512 b14 = GEMM_BROADCAST_SS_AVX512(b_row + 14);
            __m512 b15 = GEMM_BROADCAST_SS_AVX512(b_row + 15);
            
            acc_lo[12] = GEMM_FMADD_PS_AVX512(a_lo, b12, acc_lo[12]);
            acc_hi[12] = GEMM_FMADD_PS_AVX512(a_hi, b12, acc_hi[12]);
            acc_lo[13] = GEMM_FMADD_PS_AVX512(a_lo, b13, acc_lo[13]);
            acc_hi[13] = GEMM_FMADD_PS_AVX512(a_hi, b13, acc_hi[13]);
            acc_lo[14] = GEMM_FMADD_PS_AVX512(a_lo, b14, acc_lo[14]);
            acc_hi[14] = GEMM_FMADD_PS_AVX512(a_hi, b14, acc_hi[14]);
            acc_lo[15] = GEMM_FMADD_PS_AVX512(a_lo, b15, acc_lo[15]);
            acc_hi[15] = GEMM_FMADD_PS_AVX512(a_hi, b15, acc_hi[15]);
        }
    }

    // Store results back to C
    if (m == 32 && n == 16) {
        // Fast path: full 32×16 tile with transpose
        __m512 cols_lo[16], cols_hi[16];
        for (int i = 0; i < 16; ++i) {
            cols_lo[i] = acc_lo[i];
            cols_hi[i] = acc_hi[i];
        }
        
        gemm_transpose_16x16_avx512(cols_lo);
        gemm_transpose_16x16_avx512(cols_hi);
        
        for (size_t r = 0; r < 16; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = GEMM_ADD_PS_AVX512(GEMM_LOADU_PS_AVX512(cr), cols_lo[r]);
            GEMM_STOREU_PS_AVX512(cr, sum);
        }
        for (size_t r = 16; r < 32; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = GEMM_ADD_PS_AVX512(GEMM_LOADU_PS_AVX512(cr), cols_hi[r - 16]);
            GEMM_STOREU_PS_AVX512(cr, sum);
        }
    } else {
        // Slow path: partial tile via temp buffer
        alignas(64) float temp[32 * 16];
        for (int j = 0; j < 16; ++j) {
            GEMM_STORE_PS_AVX512(temp + j * 32, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp + j * 32 + 16, acc_hi[j]);
        }

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = load_cols_from_temp_avx512(temp, 32, r, n);
            __m512 old = GEMM_MASKLOAD_PS_AVX512(cr, mask);
            GEMM_MASKSTORE_PS_AVX512(cr, mask, GEMM_ADD_PS_AVX512(old, sum));
        }
    }
}

//==============================================================================
// 32×16 KERNEL (STORE): C = A*B
//==============================================================================

/**
 * @brief 32×16 micro-kernel: C = A*B (AVX-512)
 *
 * @details
 * Identical to add version but overwrites C instead of accumulating.
 * Supports optional non-temporal stores for large matrices.
 */
static inline void gemm_32x16_panel_avx512_store(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk,
    size_t m,
    size_t n,
    __mmask16 mask)
{
    LINALG_ASSUME_ALIGNED(Ap, 64);
    LINALG_ASSUME_ALIGNED(Bp, 64);

    // Accumulation phase (identical to add kernel)
    __m512 acc_lo[16], acc_hi[16];
    for (int i = 0; i < 16; ++i) {
        acc_lo[i] = GEMM_VEC_ZERO_PS_AVX512();
        acc_hi[i] = GEMM_VEC_ZERO_PS_AVX512();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 8)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk) PREFETCH_T0(Bp + kpf_s * 16);
            if (kpf_l < Kblk) PREFETCH_T0(Bp + kpf_l * 16);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk) PREFETCH_T0(Ap + kpf_l * 32);
#endif
        }

        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk) break;

            __m512 a_lo = GEMM_LOAD_PS_AVX512(Ap + kk * 32);
            __m512 a_hi = GEMM_LOAD_PS_AVX512(Ap + kk * 32 + 16);
            const float *b_row = Bp + kk * 16;

            // Unrolled FMA operations (same as add kernel)
            for (int j = 0; j < 16; ++j) {
                __m512 b = GEMM_BROADCAST_SS_AVX512(b_row + j);
                acc_lo[j] = GEMM_FMADD_PS_AVX512(a_lo, b, acc_lo[j]);
                acc_hi[j] = GEMM_FMADD_PS_AVX512(a_hi, b, acc_hi[j]);
            }
        }
    }

    // Check if non-temporal stores are beneficial
    const int use_nt = LINALG_NT_STORES &&
                       (n == 16) &&
                       (((uintptr_t)c & 63u) == 0) &&
                       ((ldc & 15u) == 0);

    if (m == 32 && n == 16) {
        __m512 cols_lo[16], cols_hi[16];
        for (int i = 0; i < 16; ++i) {
            cols_lo[i] = acc_lo[i];
            cols_hi[i] = acc_hi[i];
        }
        
        gemm_transpose_16x16_avx512(cols_lo);
        gemm_transpose_16x16_avx512(cols_hi);
        
        for (size_t r = 0; r < 16; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS_AVX512(cr, cols_lo[r]);
            else
                GEMM_STOREU_PS_AVX512(cr, cols_lo[r]);
        }
        for (size_t r = 16; r < 32; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS_AVX512(cr, cols_hi[r - 16]);
            else
                GEMM_STOREU_PS_AVX512(cr, cols_hi[r - 16]);
        }
    } else {
        alignas(64) float temp[32 * 16];
        for (int j = 0; j < 16; ++j) {
            GEMM_STORE_PS_AVX512(temp + j * 32, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp + j * 32 + 16, acc_hi[j]);
        }

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = load_cols_from_temp_avx512(temp, 32, r, n);
            if (use_nt && n == 16)
                GEMM_STREAM_PS_AVX512(cr, sum);
            else
                GEMM_MASKSTORE_PS_AVX512(cr, mask, sum);
        }
    }
}

//==============================================================================
// 32×12 KERNELS (Better column efficiency)
//==============================================================================

/**
 * @brief 32×12 micro-kernel: C += A*B (AVX-512)
 *
 * @details
 * Similar to 32×16 but with 12 columns for better efficiency when N % 16 != 0.
 * Uses only 24 accumulators instead of 32.
 */
static inline void gemm_32x12_panel_avx512_add(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk,
    size_t m,
    size_t n,
    __mmask16 mask)
{
    LINALG_ASSUME_ALIGNED(Ap, 64);
    LINALG_ASSUME_ALIGNED(Bp, 64);

    __m512 acc_lo[12], acc_hi[12];
    for (int i = 0; i < 12; ++i) {
        acc_lo[i] = GEMM_VEC_ZERO_PS_AVX512();
        acc_hi[i] = GEMM_VEC_ZERO_PS_AVX512();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 8)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk) PREFETCH_T0(Bp + kpf_s * 12);
            if (kpf_l < Kblk) PREFETCH_T0(Bp + kpf_l * 12);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk) PREFETCH_T0(Ap + kpf_l * 32);
#endif
        }

        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk) break;

            __m512 a_lo = GEMM_LOAD_PS_AVX512(Ap + kk * 32);
            __m512 a_hi = GEMM_LOAD_PS_AVX512(Ap + kk * 32 + 16);
            const float *b_row = Bp + kk * 12;

            for (int j = 0; j < 12; ++j) {
                __m512 b = GEMM_BROADCAST_SS_AVX512(b_row + j);
                acc_lo[j] = GEMM_FMADD_PS_AVX512(a_lo, b, acc_lo[j]);
                acc_hi[j] = GEMM_FMADD_PS_AVX512(a_hi, b, acc_hi[j]);
            }
        }
    }

    // Store with transpose (simpler - pad to 16×16 for transpose)
    if (m == 32 && n == 12) {
        alignas(64) float temp_lo[16 * 16] = {0}, temp_hi[16 * 16] = {0};
        for (int j = 0; j < 12; ++j) {
            GEMM_STORE_PS_AVX512(temp_lo + j * 16, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp_hi + j * 16, acc_hi[j]);
        }
        __m512 cols_lo[16], cols_hi[16];
        for (int j = 0; j < 16; ++j) {
            cols_lo[j] = GEMM_LOAD_PS_AVX512(temp_lo + j * 16);
            cols_hi[j] = GEMM_LOAD_PS_AVX512(temp_hi + j * 16);
        }
        gemm_transpose_16x16_avx512(cols_lo);
        gemm_transpose_16x16_avx512(cols_hi);
        for (size_t r = 0; r < 16; ++r) {
            float *cr = c + r * ldc;
            GEMM_STOREU_PS_AVX512(cr, GEMM_ADD_PS_AVX512(GEMM_LOADU_PS_AVX512(cr), cols_lo[r]));
        }
        for (size_t r = 16; r < 32; ++r) {
            float *cr = c + r * ldc;
            GEMM_STOREU_PS_AVX512(cr, GEMM_ADD_PS_AVX512(GEMM_LOADU_PS_AVX512(cr), cols_hi[r - 16]));
        }
    } else {
        alignas(64) float temp[32 * 12];
        for (int j = 0; j < 12; ++j) {
            GEMM_STORE_PS_AVX512(temp + j * 32, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp + j * 32 + 16, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = load_cols_from_temp_avx512(temp, 32, r, n);
            __m512 old = GEMM_MASKLOAD_PS_AVX512(cr, mask);
            GEMM_MASKSTORE_PS_AVX512(cr, mask, GEMM_ADD_PS_AVX512(old, sum));
        }
    }
}

/**
 * @brief 32×12 micro-kernel: C = A*B (AVX-512)
 */
static inline void gemm_32x12_panel_avx512_store(
    float *RESTRICT c,
    size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk,
    size_t m,
    size_t n,
    __mmask16 mask)
{
    LINALG_ASSUME_ALIGNED(Ap, 64);
    LINALG_ASSUME_ALIGNED(Bp, 64);

    __m512 acc_lo[12], acc_hi[12];
    for (int i = 0; i < 12; ++i) {
        acc_lo[i] = GEMM_VEC_ZERO_PS_AVX512();
        acc_hi[i] = GEMM_VEC_ZERO_PS_AVX512();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 8)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk) PREFETCH_T0(Bp + kpf_s * 12);
            if (kpf_l < Kblk) PREFETCH_T0(Bp + kpf_l * 12);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk) PREFETCH_T0(Ap + kpf_l * 32);
#endif
        }

        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk) break;

            __m512 a_lo = GEMM_LOAD_PS_AVX512(Ap + kk * 32);
            __m512 a_hi = GEMM_LOAD_PS_AVX512(Ap + kk * 32 + 16);
            const float *b_row = Bp + kk * 12;

            for (int j = 0; j < 12; ++j) {
                __m512 b = GEMM_BROADCAST_SS_AVX512(b_row + j);
                acc_lo[j] = GEMM_FMADD_PS_AVX512(a_lo, b, acc_lo[j]);
                acc_hi[j] = GEMM_FMADD_PS_AVX512(a_hi, b, acc_hi[j]);
            }
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 12) &&
                       (((uintptr_t)c & 63u) == 0) &&
                       ((ldc & 15u) == 0);

    if (m == 32 && n == 12) {
        alignas(64) float temp_lo[16 * 16] = {0}, temp_hi[16 * 16] = {0};
        for (int j = 0; j < 12; ++j) {
            GEMM_STORE_PS_AVX512(temp_lo + j * 16, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp_hi + j * 16, acc_hi[j]);
        }
        __m512 cols_lo[16], cols_hi[16];
        for (int j = 0; j < 16; ++j) {
            cols_lo[j] = GEMM_LOAD_PS_AVX512(temp_lo + j * 16);
            cols_hi[j] = GEMM_LOAD_PS_AVX512(temp_hi + j * 16);
        }
        gemm_transpose_16x16_avx512(cols_lo);
        gemm_transpose_16x16_avx512(cols_hi);
        for (size_t r = 0; r < 16; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS_AVX512(cr, cols_lo[r]);
            else
                GEMM_STOREU_PS_AVX512(cr, cols_lo[r]);
        }
        for (size_t r = 16; r < 32; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                GEMM_STREAM_PS_AVX512(cr, cols_hi[r - 16]);
            else
                GEMM_STOREU_PS_AVX512(cr, cols_hi[r - 16]);
        }
    } else {
        alignas(64) float temp[32 * 12];
        for (int j = 0; j < 12; ++j) {
            GEMM_STORE_PS_AVX512(temp + j * 32, acc_lo[j]);
            GEMM_STORE_PS_AVX512(temp + j * 32 + 16, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m512 sum = load_cols_from_temp_avx512(temp, 32, r, n);
            if (use_nt && n == 12)
                GEMM_STREAM_PS_AVX512(cr, sum);
            else
                GEMM_MASKSTORE_PS_AVX512(cr, mask, sum);
        }
    }
}

#endif /* __AVX512F__ */
#endif /* GEMM_KERNELS_AVX512_H */
