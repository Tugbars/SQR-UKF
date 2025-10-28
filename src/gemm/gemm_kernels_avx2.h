/**
 * @file gemm_kernels_avx2_complete.h
 * @brief Complete AVX2/FMA GEMM Micro-kernels (All Implementations)
 *
 * @details
 * This file contains ALL micro-kernel implementations from the original code:
 * - 16×8, 16×6: Large M tiles
 * - 8×8, 8×6: Medium M tiles  
 * - 4×8: Tail handling (4 rows)
 * - 1×8: Tail handling (1 row)
 *
 * ALL original optimizations preserved:
 * - 8-way K unrolling
 * - Short and long prefetching
 * - In-register transpose
 * - Non-temporal stores
 * - Masked tail handling
 *
 * @author Original GEMM Implementation (Refactored for readability)
 * @date 2025
 */

#ifndef GEMM_KERNELS_AVX2_COMPLETE_H
#define GEMM_KERNELS_AVX2_COMPLETE_H

#include "gemm_simd_ops.h"

//==============================================================================
// HELPER: Load columns from column-major temp buffer
//==============================================================================

static inline __m256 load_cols_from_temp(
    const float *temp,
    size_t stride,
    size_t r,
    size_t n)
{
    alignas(32) float lane[8] = {0};
    for (size_t j = 0; j < n; ++j)
        lane[j] = temp[j * stride + r];
    return _mm256_load_ps(lane);
}

//==============================================================================
// 4×8 KERNELS (Tail handling for 4 rows)
//==============================================================================

/**
 * @brief 4×8 kernel (ADD): C += A*B
 * @note Ap layout: ld=8, only first 4 lanes used (lanes 4-7 are padding)
 */
static inline void gemm_4x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t jb, __m256i m)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 4 * ldc);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
        if (do_pf) {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        
        for (int t = 0; t < 8; ++t) {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += 8;
            aptr += 8;
        }
    }
    
    for (; k < Kblk; ++k) {
        if (do_pf) {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
    }

    if (jb == 8) {
        _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 0 * ldc), acc0));
        _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 1 * ldc), acc1));
        _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), acc2));
        _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), acc3));
    } else {
        __m256 old, sum;
        old = _mm256_maskload_ps(c + 0 * ldc, m);
        sum = _mm256_add_ps(old, acc0);
        _mm256_maskstore_ps(c + 0 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 1 * ldc, m);
        sum = _mm256_add_ps(old, acc1);
        _mm256_maskstore_ps(c + 1 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 2 * ldc, m);
        sum = _mm256_add_ps(old, acc2);
        _mm256_maskstore_ps(c + 2 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 3 * ldc, m);
        sum = _mm256_add_ps(old, acc3);
        _mm256_maskstore_ps(c + 3 * ldc, m, sum);
    }
}

/**
 * @brief 4×8 kernel (STORE): C = A*B
 */
static inline void gemm_4x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t jb, __m256i m)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 4 * ldc);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
        if (do_pf) {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        
        for (int t = 0; t < 8; ++t) {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += 8;
            aptr += 8;
        }
    }
    
    for (; k < Kblk; ++k) {
        if (do_pf) {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
    }

    if (jb == 8) {
        _mm256_storeu_ps(c + 0 * ldc, acc0);
        _mm256_storeu_ps(c + 1 * ldc, acc1);
        _mm256_storeu_ps(c + 2 * ldc, acc2);
        _mm256_storeu_ps(c + 3 * ldc, acc3);
    } else {
        _mm256_maskstore_ps(c + 0 * ldc, m, acc0);
        _mm256_maskstore_ps(c + 1 * ldc, m, acc1);
        _mm256_maskstore_ps(c + 2 * ldc, m, acc2);
        _mm256_maskstore_ps(c + 3 * ldc, m, acc3);
    }
}

//==============================================================================
// 1×8 KERNELS (Tail handling for 1 row)
//==============================================================================

/**
 * @brief 1×8 kernel (ADD): C += A*B
 * @note Ap layout: ld=8, only aptr[0] used
 */
static inline void gemm_1x8_panel_avx2fma_add(
    float *RESTRICT c,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t jb, __m256i m)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
        if (do_pf) {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        
        for (int t = 0; t < 8; ++t) {
            const __m256 b = _mm256_load_ps(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += 8;
            aptr += 8;
        }
    }
    
    for (; k < Kblk; ++k) {
        if (do_pf) {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc);
    }

    if (jb == 8) {
        _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), acc));
    } else {
        __m256 oldv = _mm256_maskload_ps(c, m);
        __m256 sum = _mm256_add_ps(oldv, acc);
        _mm256_maskstore_ps(c, m, sum);
    }
}

/**
 * @brief 1×8 kernel (STORE): C = A*B
 */
static inline void gemm_1x8_panel_avx2fma_store(
    float *RESTRICT c,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t jb, __m256i m)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);

    PREFETCH_T0(c);

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
        if (do_pf) {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        
        for (int t = 0; t < 8; ++t) {
            const __m256 b = _mm256_load_ps(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += 8;
            aptr += 8;
        }
    }
    
    for (; k < Kblk; ++k) {
        if (do_pf) {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc);
    }

    if (jb == 8) {
        _mm256_storeu_ps(c, acc);
    } else {
        _mm256_maskstore_ps(c, m, acc);
    }
}

//==============================================================================
// 16×8 KERNELS
//==============================================================================

/**
 * @brief 16×8 kernel (ADD): C += A*B
 */
static inline void gemm_16x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 8;
            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);
            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    if (m == 16 && n == 8) {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(cr), cols_lo[r]);
            _mm256_storeu_ps(cr, sum);
        }
        for (size_t r = 8; r < 16; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(cr), cols_hi[r - 8]);
            _mm256_storeu_ps(cr, sum);
        }
    } else {
        alignas(32) float temp[16 * 8];
        _mm256_store_ps(temp + 0 * 16, acc_lo0);
        _mm256_store_ps(temp + 0 * 16 + 8, acc_hi0);
        _mm256_store_ps(temp + 1 * 16, acc_lo1);
        _mm256_store_ps(temp + 1 * 16 + 8, acc_hi1);
        _mm256_store_ps(temp + 2 * 16, acc_lo2);
        _mm256_store_ps(temp + 2 * 16 + 8, acc_hi2);
        _mm256_store_ps(temp + 3 * 16, acc_lo3);
        _mm256_store_ps(temp + 3 * 16 + 8, acc_hi3);
        _mm256_store_ps(temp + 4 * 16, acc_lo4);
        _mm256_store_ps(temp + 4 * 16 + 8, acc_hi4);
        _mm256_store_ps(temp + 5 * 16, acc_lo5);
        _mm256_store_ps(temp + 5 * 16 + 8, acc_hi5);
        _mm256_store_ps(temp + 6 * 16, acc_lo6);
        _mm256_store_ps(temp + 6 * 16 + 8, acc_hi6);
        _mm256_store_ps(temp + 7 * 16, acc_lo7);
        _mm256_store_ps(temp + 7 * 16 + 8, acc_hi7);

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            __m256 old = _mm256_maskload_ps(cr, mask);
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 16×8 kernel (STORE): C = A*B
 */
static inline void gemm_16x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 8;
            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);
            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 8) {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_lo[r]);
            else
                _mm256_storeu_ps(cr, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_hi[r - 8]);
            else
                _mm256_storeu_ps(cr, cols_hi[r - 8]);
        }
    } else {
        alignas(32) float temp[16 * 8];
        _mm256_store_ps(temp + 0 * 16, acc_lo0);
        _mm256_store_ps(temp + 0 * 16 + 8, acc_hi0);
        _mm256_store_ps(temp + 1 * 16, acc_lo1);
        _mm256_store_ps(temp + 1 * 16 + 8, acc_hi1);
        _mm256_store_ps(temp + 2 * 16, acc_lo2);
        _mm256_store_ps(temp + 2 * 16 + 8, acc_hi2);
        _mm256_store_ps(temp + 3 * 16, acc_lo3);
        _mm256_store_ps(temp + 3 * 16 + 8, acc_hi3);
        _mm256_store_ps(temp + 4 * 16, acc_lo4);
        _mm256_store_ps(temp + 4 * 16 + 8, acc_hi4);
        _mm256_store_ps(temp + 5 * 16, acc_lo5);
        _mm256_store_ps(temp + 5 * 16 + 8, acc_hi5);
        _mm256_store_ps(temp + 6 * 16, acc_lo6);
        _mm256_store_ps(temp + 6 * 16 + 8, acc_hi6);
        _mm256_store_ps(temp + 7 * 16, acc_lo7);
        _mm256_store_ps(temp + 7 * 16 + 8, acc_hi7);

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            if (use_nt && n == 8)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

// Continued in next message due to length...

#endif /* GEMM_KERNELS_AVX2_COMPLETE_H */

//==============================================================================
// 8×8 KERNELS
//==============================================================================

/**
 * @brief 8×8 kernel (ADD): C += A*B
 */
static inline void gemm_8x8_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 8;
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(b_row + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(b_row + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    if (m == 8 && n == 8) {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        gemm_transpose_8x8_avx2(cols);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols[r]));
        }
    } else {
        alignas(32) float temp[8 * 8];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        _mm256_store_ps(temp + 6 * 8, acc6);
        _mm256_store_ps(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            __m256 old = _mm256_maskload_ps(cr, mask);
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 8×8 kernel (STORE): C = A*B
 */
static inline void gemm_8x8_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 8;
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(b_row + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(b_row + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 8) {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        gemm_transpose_8x8_avx2(cols);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols[r]);
            else
                _mm256_storeu_ps(cr, cols[r]);
        }
    } else {
        alignas(32) float temp[8 * 8];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        _mm256_store_ps(temp + 6 * 8, acc6);
        _mm256_store_ps(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            if (use_nt && n == 8)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

//==============================================================================
// 16×6 KERNELS
//==============================================================================

/**
 * @brief 16×6 kernel (ADD): C += A*B
 */
static inline void gemm_16x6_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j) {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 6;
            for (int j = 0; j < 6; ++j) {
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    if (m == 16 && n == 6) {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j) {
            _mm256_store_ps(temp_lo + j * 8, acc_lo[j]);
            _mm256_store_ps(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j) {
            cols_lo[j] = _mm256_load_ps(temp_lo + j * 8);
            cols_hi[j] = _mm256_load_ps(temp_hi + j * 8);
        }
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols_lo[r]));
        }
        for (size_t r = 8; r < 16; ++r) {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols_hi[r - 8]));
        }
    } else {
        alignas(32) float temp[16 * 6];
        for (int j = 0; j < 6; ++j) {
            _mm256_store_ps(temp + j * 16, acc_lo[j]);
            _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            __m256 old = _mm256_maskload_ps(cr, mask);
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 16×6 kernel (STORE): C = A*B
 */
static inline void gemm_16x6_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j) {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 6;
            for (int j = 0; j < 6; ++j) {
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 6) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 6) {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j) {
            _mm256_store_ps(temp_lo + j * 8, acc_lo[j]);
            _mm256_store_ps(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j) {
            cols_lo[j] = _mm256_load_ps(temp_lo + j * 8);
            cols_hi[j] = _mm256_load_ps(temp_hi + j * 8);
        }
        gemm_transpose_8x8_avx2(cols_lo);
        gemm_transpose_8x8_avx2(cols_hi);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_lo[r]);
            else
                _mm256_storeu_ps(cr, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_hi[r - 8]);
            else
                _mm256_storeu_ps(cr, cols_hi[r - 8]);
        }
    } else {
        alignas(32) float temp[16 * 6];
        for (int j = 0; j < 6; ++j) {
            _mm256_store_ps(temp + j * 16, acc_lo[j]);
            _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
        }
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 16, r, n);
            if (use_nt && n == 6)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

//==============================================================================
// 8×6 KERNELS
//==============================================================================

/**
 * @brief 8×6 kernel (ADD): C += A*B
 */
static inline void gemm_8x6_panel_avx2fma_add(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 6;
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    if (m == 8 && n == 6) {
        alignas(32) float temp[8 * 8] = {0};
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = _mm256_load_ps(temp + j * 8);
        gemm_transpose_8x8_avx2(cols);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols[r]));
        }
    } else {
        alignas(32) float temp[8 * 6];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            __m256 old = _mm256_maskload_ps(cr, mask);
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/**
 * @brief 8×6 kernel (STORE): C = A*B
 */
static inline void gemm_8x6_panel_avx2fma_store(
    float *RESTRICT c, size_t ldc,
    const float *RESTRICT Ap,
    const float *RESTRICT Bp,
    size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)GEMM_PREFETCH_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8) {
        if (do_pf) {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u) {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 6;
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 6) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 6) {
        alignas(32) float temp[8 * 8] = {0};
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = _mm256_load_ps(temp + j * 8);
        gemm_transpose_8x8_avx2(cols);
        for (size_t r = 0; r < 8; ++r) {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols[r]);
            else
                _mm256_storeu_ps(cr, cols[r]);
        }
    } else {
        alignas(32) float temp[8 * 6];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        for (size_t r = 0; r < m; ++r) {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            if (use_nt && n == 6)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

