// SPDX-License-Identifier: MIT
/**
 * @file inv_blas3.c
 * @brief Blocked BLAS-3 matrix inverse via LU (single-precision), AVX2/FMA accelerated.
 *
 * @details
 * Computes A^{-1} by:
 *   1) LU with partial pivoting: P A = L U  (external lup()).
 *   2) Solve A X = I in large RHS tiles (BLAS-3 GETRS style):
 *        - Apply pivots once to the RHS tile (RHS ← P·RHS).
 *        - Forward solve:  L · Y = RHS  (unit lower).
 *        - Backward solve: U · X = Y    (upper, non-unit).
 * The approach keeps the work in Level-3 (matrix–matrix) kernels, which
 * maximizes cache reuse and SIMD utilization, unlike per-column solves.
 *
 * AVX2/FMA path processes RHS micro-panels of 8 columns with dual-accumulator
 * fused multiply-add loops (software pipelined). Scalar fallback provided.
 *
 * Requirements:
 *  - Row-major storage.
 *  - External helpers:
 *      int  lup(const float* A, float* LU, uint8_t* P, uint16_t n);
 *      int  linalg_has_avx2(void);
 *      void* linalg_aligned_alloc(size_t align, size_t bytes);
 *      void  linalg_aligned_free(void* ptr);
 *
 * Notes:
 *  - No threading here (single-threaded). You can wrap outer tiles with threads.
 *  - Robust relative singularity checks on U’s diagonal protect against near-singular rows.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <errno.h>
#include <float.h>
#include <math.h>

#include "linalg_simd.h" // lup(), linalg_has_avx2(), linalg_aligned_alloc/free, LINALG_SMALL_N_THRESH

#ifndef INV_NRHS_TILE
#define INV_NRHS_TILE 64 // RHS tile width (try 64–192; must be >= 8 and multiple of 8 for full-vector tiles)
#endif

#ifndef INV_JC_BLOCK
#define INV_JC_BLOCK 256 // Inner accumulation blocking for L/U dot products (tune 192–384)
#endif

/* ========================= Scalar helpers ========================= */

/* Apply permutation P (as in GETRF) to RHS (RHS ← P·RHS). P encodes the swap row i <-> P[i]. */
static void apply_pivots_to_rhs(float *RESTRICT RHS, uint16_t n, uint16_t jb,
                                const uint8_t *RESTRICT P)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        uint16_t pi = P[i];
        if (pi != i)
        {
            float *ri = RHS + (size_t)i * jb;
            float *rpi = RHS + (size_t)pi * jb;
            for (uint16_t c = 0; c < jb; ++c)
            {
                float t = ri[c];
                ri[c] = rpi[c];
                rpi[c] = t;
            }
        }
    }
}

/* Forward solve: L (unit lower) * Y = B, scalar, blocked over j in [0..i) for cache. */
static int forward_solve_L_scalar(const float *RESTRICT LU, float *RESTRICT B,
                                  uint16_t n, uint16_t jb)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;

    for (uint16_t i = 0; i < n; ++i)
    {
        const float *Li = LU + (size_t)i * n;

        /* B[i,:] -= sum_{j<i} L(i,j) * B[j,:]  (unit diag, so no divide) */
        for (uint16_t j0 = 0; j0 < i; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < i) ? (j0 + Jc) : i);
            for (uint16_t j = j0; j < j1; ++j)
            {
                float lij = Li[j];
                if (lij != 0.0f)
                {
                    const float *Bj = B + (size_t)j * jb;
                    float *Bi = B + (size_t)i * jb;
                    for (uint16_t c = 0; c < jb; ++c)
                        Bi[c] -= lij * Bj[c];
                }
            }
        }
    }
    return 0;
}

/* Backward solve: U (upper, non-unit) * X = Y, scalar, with relative pivot safety. */
static int backward_solve_U_scalar(const float *RESTRICT LU, float *RESTRICT B,
                                   uint16_t n, uint16_t jb)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;

    for (int ii = (int)n - 1; ii >= 0; --ii)
    {
        uint16_t i = (uint16_t)ii;
        const float *Ui = LU + (size_t)i * n;

        /* Accumulate s = sum_{j>i} U(i,j) * X[j,:] */
        for (uint16_t j0 = (uint16_t)i + 1; j0 < n; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < n) ? (j0 + Jc) : n);
            for (uint16_t j = j0; j < j1; ++j)
            {
                float uij = Ui[j];
                if (uij != 0.0f)
                {
                    const float *Xj = B + (size_t)j * jb;
                    float *Xi = B + (size_t)i * jb;
                    for (uint16_t c = 0; c < jb; ++c)
                        Xi[c] -= uij * Xj[c];
                }
            }
        }

        /* Divide by diag with relative tolerance */
        float scale = 0.0f;
        for (uint16_t k = i; k < n; ++k)
        {
            float ak = fabsf(Ui[k]);
            if (ak > scale)
                scale = ak;
        }
        float di = Ui[i];
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(di) <= tol)
            return -ENOTSUP;

        float invd = 1.0f / di;
        float *Xi = B + (size_t)i * jb;
        for (uint16_t c = 0; c < jb; ++c)
            Xi[c] *= invd;
    }
    return 0;
}

/* ========================= AVX2 micro-panels (8 RHS) ========================= */

#if LINALG_SIMD_ENABLE
/* Process exactly 8 RHS columns (micro-panel), in-place in B8 (n x 8, row-major by rows). */
static int forward_solve_L_8rhs_avx(const float *RESTRICT LU, float *RESTRICT B8,
                                    uint16_t n)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;

    for (uint16_t i = 0; i < n; ++i)
    {
        const float *Li = LU + (size_t)i * n;
        __m256 Bi = _mm256_load_ps(B8 + (size_t)i * 8);

        /* Bi -= sum_{j<i} L(i,j) * B8[j,:] */
        for (uint16_t j0 = 0; j0 < i; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < i) ? (j0 + Jc) : i);
            uint16_t j = j0;
            for (; j + 7 < j1; j += 8)
            {
                __m256 lpack = _mm256_loadu_ps(Li + j);
                __m256 b0 = _mm256_load_ps(B8 + (size_t)(j + 0) * 8);
                __m256 b1 = _mm256_load_ps(B8 + (size_t)(j + 1) * 8);
                __m256 b2 = _mm256_load_ps(B8 + (size_t)(j + 2) * 8);
                __m256 b3 = _mm256_load_ps(B8 + (size_t)(j + 3) * 8);
                __m256 b4 = _mm256_load_ps(B8 + (size_t)(j + 4) * 8);
                __m256 b5 = _mm256_load_ps(B8 + (size_t)(j + 5) * 8);
                __m256 b6 = _mm256_load_ps(B8 + (size_t)(j + 6) * 8);
                __m256 b7 = _mm256_load_ps(B8 + (size_t)(j + 7) * 8);

                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(0)), b0, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(1)), b1, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(2)), b2, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(3)), b3, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(4)), b4, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(5)), b5, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(6)), b6, Bi);
                Bi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(lpack, _mm256_set1_epi32(7)), b7, Bi);
            }
            for (; j < j1; ++j)
            {
                __m256 lij = _mm256_set1_ps(Li[j]);
                __m256 Bj = _mm256_load_ps(B8 + (size_t)j * 8);
                Bi = _mm256_fnmadd_ps(lij, Bj, Bi);
            }
        }
        _mm256_store_ps(B8 + (size_t)i * 8, Bi);
    }
    return 0;
}

static int backward_solve_U_8rhs_avx(const float *RESTRICT LU, float *RESTRICT B8,
                                     uint16_t n)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;

    for (int ii = (int)n - 1; ii >= 0; --ii)
    {
        uint16_t i = (uint16_t)ii;
        const float *Ui = LU + (size_t)i * n;

        __m256 Xi = _mm256_load_ps(B8 + (size_t)i * 8);

        /* Xi -= sum_{j>i} U(i,j) * X[j,:] */
        for (uint16_t j0 = (uint16_t)i + 1; j0 < n; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < n) ? (j0 + Jc) : n);
            uint16_t j = j0;
            for (; j + 7 < j1; j += 8)
            {
                __m256 upack = _mm256_loadu_ps(Ui + j);
                __m256 x0 = _mm256_load_ps(B8 + (size_t)(j + 0) * 8);
                __m256 x1 = _mm256_load_ps(B8 + (size_t)(j + 1) * 8);
                __m256 x2 = _mm256_load_ps(B8 + (size_t)(j + 2) * 8);
                __m256 x3 = _mm256_load_ps(B8 + (size_t)(j + 3) * 8);
                __m256 x4 = _mm256_load_ps(B8 + (size_t)(j + 4) * 8);
                __m256 x5 = _mm256_load_ps(B8 + (size_t)(j + 5) * 8);
                __m256 x6 = _mm256_load_ps(B8 + (size_t)(j + 6) * 8);
                __m256 x7 = _mm256_load_ps(B8 + (size_t)(j + 7) * 8);

                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(0)), x0, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(1)), x1, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(2)), x2, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(3)), x3, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(4)), x4, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(5)), x5, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(6)), x6, Xi);
                Xi = _mm256_fnmadd_ps(_mm256_permutevar8x32_ps(upack, _mm256_set1_epi32(7)), x7, Xi);
            }
            for (; j < j1; ++j)
            {
                __m256 uij = _mm256_set1_ps(Ui[j]);
                __m256 Xj = _mm256_load_ps(B8 + (size_t)j * 8);
                Xi = _mm256_fnmadd_ps(uij, Xj, Xi);
            }
        }

        /* Relative tolerance on U(i,i) */
        float scale = 0.0f;
        for (uint16_t k = i; k < n; ++k)
        {
            float ak = fabsf(Ui[k]);
            if (ak > scale)
                scale = ak;
        }
        float di = Ui[i];
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(di) <= tol)
            return -ENOTSUP;

        __m256 invd = _mm256_set1_ps(1.0f / di);
        Xi = _mm256_mul_ps(Xi, invd);
        _mm256_store_ps(B8 + (size_t)i * 8, Xi);
    }
    return 0;
}
#endif /* LINALG_SIMD_ENABLE */


// mr = 8, nr = 16, kc arbitrary (multiple of 2 preferred)
// A pack layout: A[0..mr*kc-1], row-major: row r at A + r*kc
// B pack layout: B[0..kc*nr-1], col-major-by-tiles-of-16 for contiguous loads: 
//                at k-step, load two 8-float vectors for the 16 columns.
//
// Contract: C is row-major, leading dimension ldc (in elements).
// op: C += A * B  (for Z = T*Y or Y = V^T*C)  or  C -= A * B  (for C -= V*Z)
static inline void sgemm_8x16_u2_fma_avx2(
    const float *RESTRICT A,      // mr×kc
    const float *RESTRICT B,      // kc×nr
    float       *RESTRICT C,      // mr×nr
    uint16_t kc, uint16_t ldc,
    int op_minus                 // 0: C += AB, 1: C -= AB
){
    // 16 accumulators (8 rows × 2 blocks of 8 cols)
    __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps(); // row0, col[0:7],[8:15]
    __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps(); __m256 c61 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps(); __m256 c71 = _mm256_setzero_ps();

    const float *a0 = A + 0*kc;  const float *a1 = A + 1*kc;
    const float *a2 = A + 2*kc;  const float *a3 = A + 3*kc;
    const float *a4 = A + 4*kc;  const float *a5 = A + 5*kc;
    const float *a6 = A + 6*kc;  const float *a7 = A + 7*kc;

    const float *bk = B;

    // Prefetch streams
    _mm_prefetch((const char*)(A + 64), _MM_HINT_T0);
    _mm_prefetch((const char*)(B + 64), _MM_HINT_T0);

    // Pipeline: load next B vectors while doing FMAs on current ones (unroll k by 2)
    uint16_t k = 0;
    if (kc >= 2) {
        // prime: load first two B 8-lane halves
        __m256 b0_0 = _mm256_load_ps(bk +  0); // cols [0..7] at k
        __m256 b0_1 = _mm256_load_ps(bk +  8); // cols [8..15] at k
        __m256 b1_0 = _mm256_load_ps(bk + 16); // cols [0..7] at k+1
        __m256 b1_1 = _mm256_load_ps(bk + 24); // cols [8..15] at k+1

        // consume two k at a time
        for (; k + 1 < kc; k += 2) {
            // load A scalars for k   (broadcast to 8 lanes)
            __m256 a0k = _mm256_set1_ps(a0[k]), a1k = _mm256_set1_ps(a1[k]);
            __m256 a2k = _mm256_set1_ps(a2[k]), a3k = _mm256_set1_ps(a3[k]);
            __m256 a4k = _mm256_set1_ps(a4[k]), a5k = _mm256_set1_ps(a5[k]);
            __m256 a6k = _mm256_set1_ps(a6[k]), a7k = _mm256_set1_ps(a7[k]);

            // FMA for k using b0_0/b0_1 (already loaded)
            c00 = _mm256_fmadd_ps(a0k, b0_0, c00); c01 = _mm256_fmadd_ps(a0k, b0_1, c01);
            c10 = _mm256_fmadd_ps(a1k, b0_0, c10); c11 = _mm256_fmadd_ps(a1k, b0_1, c11);
            c20 = _mm256_fmadd_ps(a2k, b0_0, c20); c21 = _mm256_fmadd_ps(a2k, b0_1, c21);
            c30 = _mm256_fmadd_ps(a3k, b0_0, c30); c31 = _mm256_fmadd_ps(a3k, b0_1, c31);
            c40 = _mm256_fmadd_ps(a4k, b0_0, c40); c41 = _mm256_fmadd_ps(a4k, b0_1, c41);
            c50 = _mm256_fmadd_ps(a5k, b0_0, c50); c51 = _mm256_fmadd_ps(a5k, b0_1, c51);
            c60 = _mm256_fmadd_ps(a6k, b0_0, c60); c61 = _mm256_fmadd_ps(a6k, b0_1, c61);
            c70 = _mm256_fmadd_ps(a7k, b0_0, c70); c71 = _mm256_fmadd_ps(a7k, b0_1, c71);

            // prefetch next B (two steps ahead) while doing next FMAs
            _mm_prefetch((const char*)(bk + 64), _MM_HINT_T0); // stride choice depends on packing

            // load A scalars for k+1
            __m256 a0k1 = _mm256_set1_ps(a0[k+1]), a1k1 = _mm256_set1_ps(a1[k+1]);
            __m256 a2k1 = _mm256_set1_ps(a2[k+1]), a3k1 = _mm256_set1_ps(a3[k+1]);
            __m256 a4k1 = _mm256_set1_ps(a4[k+1]), a5k1 = _mm256_set1_ps(a5[k+1]);
            __m256 a6k1 = _mm256_set1_ps(a6[k+1]), a7k1 = _mm256_set1_ps(a7[k+1]);

            // FMA for k+1 using b1_0/b1_1 (already loaded)
            c00 = _mm256_fmadd_ps(a0k1, b1_0, c00); c01 = _mm256_fmadd_ps(a0k1, b1_1, c01);
            c10 = _mm256_fmadd_ps(a1k1, b1_0, c10); c11 = _mm256_fmadd_ps(a1k1, b1_1, c11);
            c20 = _mm256_fmadd_ps(a2k1, b1_0, c20); c21 = _mm256_fmadd_ps(a2k1, b1_1, c21);
            c30 = _mm256_fmadd_ps(a3k1, b1_0, c30); c31 = _mm256_fmadd_ps(a3k1, b1_1, c31);
            c40 = _mm256_fmadd_ps(a4k1, b1_0, c40); c41 = _mm256_fmadd_ps(a4k1, b1_1, c41);
            c50 = _mm256_fmadd_ps(a5k1, b1_0, c50); c51 = _mm256_fmadd_ps(a5k1, b1_1, c51);
            c60 = _mm256_fmadd_ps(a6k1, b1_0, c60); c61 = _mm256_fmadd_ps(a6k1, b1_1, c61);
            c70 = _mm256_fmadd_ps(a7k1, b1_0, c70); c71 = _mm256_fmadd_ps(a7k1, b1_1, c71);

            // advance B pointers by 2*nr elements
            bk += 32;

            // load next pair (pipeline)
            if (k + 2 < kc) {
                b0_0 = _mm256_load_ps(bk +  0);
                b0_1 = _mm256_load_ps(bk +  8);
                b1_0 = _mm256_load_ps(bk + 16);
                b1_1 = _mm256_load_ps(bk + 24);
            }
        }
    }

    // tail k if odd
    if (k < kc) {
        __m256 b0 = _mm256_load_ps(bk + 0);
        __m256 b1 = _mm256_load_ps(bk + 8);

        __m256 a0k = _mm256_set1_ps(a0[k]), a1k = _mm256_set1_ps(a1[k]);
        __m256 a2k = _mm256_set1_ps(a2[k]), a3k = _mm256_set1_ps(a3[k]);
        __m256 a4k = _mm256_set1_ps(a4[k]), a5k = _mm256_set1_ps(a5[k]);
        __m256 a6k = _mm256_set1_ps(a6[k]), a7k = _mm256_set1_ps(a7[k]);

        c00 = _mm256_fmadd_ps(a0k, b0, c00); c01 = _mm256_fmadd_ps(a0k, b1, c01);
        c10 = _mm256_fmadd_ps(a1k, b0, c10); c11 = _mm256_fmadd_ps(a1k, b1, c11);
        c20 = _mm256_fmadd_ps(a2k, b0, c20); c21 = _mm256_fmadd_ps(a2k, b1, c21);
        c30 = _mm256_fmadd_ps(a3k, b0, c30); c31 = _mm256_fmadd_ps(a3k, b1, c31);
        c40 = _mm256_fmadd_ps(a4k, b0, c40); c41 = _mm256_fmadd_ps(a4k, b1, c41);
        c50 = _mm256_fmadd_ps(a5k, b0, c50); c51 = _mm256_fmadd_ps(a5k, b1, c51);
        c60 = _mm256_fmadd_ps(a6k, b0, c60); c61 = _mm256_fmadd_ps(a6k, b1, c61);
        c70 = _mm256_fmadd_ps(a7k, b0, c70); c71 = _mm256_fmadd_ps(a7k, b1, c71);
    }

    // Combine to C: add or subtract
    // C rows are contiguous with stride ldc
    float *c0p = C + 0*ldc;
    float *c1p = C + 1*ldc;
    float *c2p = C + 2*ldc;
    float *c3p = C + 3*ldc;
    float *c4p = C + 4*ldc;
    float *c5p = C + 5*ldc;
    float *c6p = C + 6*ldc;
    float *c7p = C + 7*ldc;

    // load/store pairs per row
    #define UPD_ROW(cp, v0, v1) do {                       \
        __m256 dst0 = _mm256_load_ps((cp) + 0);            \
        __m256 dst1 = _mm256_load_ps((cp) + 8);            \
        if (op_minus) {                                    \
            dst0 = _mm256_sub_ps(dst0, (v0));              \
            dst1 = _mm256_sub_ps(dst1, (v1));              \
        } else {                                           \
            dst0 = _mm256_add_ps(dst0, (v0));              \
            dst1 = _mm256_add_ps(dst1, (v1));              \
        }                                                  \
        _mm256_store_ps((cp) + 0, dst0);                   \
        _mm256_store_ps((cp) + 8, dst1);                   \
    } while(0)

    UPD_ROW(c0p, c00, c01);  UPD_ROW(c1p, c10, c11);
    UPD_ROW(c2p, c20, c21);  UPD_ROW(c3p, c30, c31);
    UPD_ROW(c4p, c40, c41);  UPD_ROW(c5p, c50, c51);
    UPD_ROW(c6p, c60, c61);  UPD_ROW(c7p, c70, c71);

    #undef UPD_ROW
}

/* ========================= Public: inv() ========================= */

int inv(float *RESTRICT Ai_out, const float *RESTRICT A, uint16_t n)
{
    if (n == 0)
        return -EINVAL;

    /* Small matrices: scalar path is simpler and often faster. */
    if (n < LINALG_SMALL_N_THRESH)
    {
        float LU[(size_t)n * n];
        uint8_t P[n];
        if (lup(A, LU, P, n) != 0)
            return -ENOTSUP;

        /* Build inverse by GETRS scalar over identity columns (blocked a bit) */
        const uint16_t tile = 16;
        float *RHS = (float *)linalg_aligned_alloc(32, (size_t)n * tile * sizeof(float));
        if (!RHS)
            return -ENOMEM;

        for (uint16_t col0 = 0; col0 < n; col0 += tile)
        {
            uint16_t jb = (uint16_t)((col0 + tile <= n) ? tile : (n - col0));

            /* RHS tile = I(:, col0:col0+jb-1) */
            memset(RHS, 0, (size_t)n * jb * sizeof(float));
            for (uint16_t t = 0; t < jb; ++t)
                RHS[(size_t)(col0 + t) * jb + t] = 1.0f; /* row-major by rows */

            /* Apply pivots: RHS ← P · RHS */
            apply_pivots_to_rhs(RHS, n, jb, P);

            /* Forward: L * Y = RHS */
            int rc = forward_solve_L_scalar(LU, RHS, n, jb);
            if (rc)
            {
                linalg_aligned_free(RHS);
                return rc;
            }

            /* Backward: U * X = Y */
            rc = backward_solve_U_scalar(LU, RHS, n, jb);
            if (rc)
            {
                linalg_aligned_free(RHS);
                return rc;
            }

            /* Scatter into Ai_out */
            for (uint16_t r = 0; r < n; ++r)
            {
                float *dst = Ai_out + (size_t)r * n + col0;
                memcpy(dst, RHS + (size_t)r * jb, (size_t)jb * sizeof(float));
            }
        }

        linalg_aligned_free(RHS);
        return 0;
    }

    /* General case: blocked BLAS-3 style with AVX2 micro-panels for 8 RHS. */
    float *LU = (float *)linalg_aligned_alloc(32, (size_t)n * n * sizeof(float));
    uint8_t *P = (uint8_t *)linalg_aligned_alloc(32, (size_t)n * sizeof(uint8_t));
    if (!LU || !P)
    {
        if (LU)
            linalg_aligned_free(LU);
        if (P)
            linalg_aligned_free(P);
        return -ENOMEM;
    }

    /* LU factorization */
    if (lup(A, LU, P, n) != 0)
    {
        linalg_aligned_free(LU);
        linalg_aligned_free(P);
        return -ENOTSUP;
    }

    /* RHS tile buffer (n x NRHS_TILE) */
    const uint16_t NRHS = (uint16_t)INV_NRHS_TILE;
    float *RHS = (float *)linalg_aligned_alloc(32, (size_t)n * NRHS * sizeof(float));
    if (!RHS)
    {
        linalg_aligned_free(LU);
        linalg_aligned_free(P);
        return -ENOMEM;
    }

    const int use_avx =
#if LINALG_SIMD_ENABLE
        linalg_has_avx2();
#else
        0;
#endif

    for (uint16_t col0 = 0; col0 < n; col0 += NRHS)
    {
        uint16_t jb = (uint16_t)((col0 + NRHS <= n) ? NRHS : (n - col0));

        /* Build RHS tile = I(:, col0:col0+jb-1) */
        memset(RHS, 0, (size_t)n * jb * sizeof(float));
        for (uint16_t t = 0; t < jb; ++t)
            RHS[(size_t)(col0 + t) * jb + t] = 1.0f; /* Kronecker columns placed on corresponding rows */

        /* Apply pivoting once: RHS ← P · RHS (row swaps) */
        apply_pivots_to_rhs(RHS, n, jb, P);

        if (!use_avx)
        {
            /* Scalar BLAS-3 forward/backward */
            int rc = forward_solve_L_scalar(LU, RHS, n, jb);
            if (rc)
            {
                linalg_aligned_free(RHS);
                linalg_aligned_free(LU);
                linalg_aligned_free(P);
                return rc;
            }
            rc = backward_solve_U_scalar(LU, RHS, n, jb);
            if (rc)
            {
                linalg_aligned_free(RHS);
                linalg_aligned_free(LU);
                linalg_aligned_free(P);
                return rc;
            }
        }
        else
        {
#if LINALG_SIMD_ENABLE
            /* Process RHS tile in 8-column micro-panels with AVX2 */
            uint16_t t = 0;
            for (; t + 7 < jb; t += 8)
            {
                /* Work on sub-panel B8 (n x 8) in-place, referencing RHS rows */
                /* For cache friendliness, we copy the micro-panel into a tight buffer. */
                float *B8 = (float *)linalg_aligned_alloc(32, (size_t)n * 8 * sizeof(float));
                if (!B8)
                {
                    linalg_aligned_free(RHS);
                    linalg_aligned_free(LU);
                    linalg_aligned_free(P);
                    return -ENOMEM;
                }
                for (uint16_t r = 0; r < n; ++r)
                    memcpy(B8 + (size_t)r * 8, RHS + (size_t)r * jb + t, 8 * sizeof(float));

                int rc = forward_solve_L_8rhs_avx(LU, B8, n);
                if (rc)
                {
                    linalg_aligned_free(B8);
                    linalg_aligned_free(RHS);
                    linalg_aligned_free(LU);
                    linalg_aligned_free(P);
                    return rc;
                }
                rc = backward_solve_U_8rhs_avx(LU, B8, n);
                if (rc)
                {
                    linalg_aligned_free(B8);
                    linalg_aligned_free(RHS);
                    linalg_aligned_free(LU);
                    linalg_aligned_free(P);
                    return rc;
                }

                /* Scatter back */
                for (uint16_t r = 0; r < n; ++r)
                    memcpy(RHS + (size_t)r * jb + t, B8 + (size_t)r * 8, 8 * sizeof(float));
                linalg_aligned_free(B8);
            }
            /* Tail ( <8 RHS ): fallback to scalar */
            if (t < jb)
            {
                int rc = forward_solve_L_scalar(LU, RHS + t, n, (uint16_t)(jb - t));
                if (rc)
                {
                    linalg_aligned_free(RHS);
                    linalg_aligned_free(LU);
                    linalg_aligned_free(P);
                    return rc;
                }
                rc = backward_solve_U_scalar(LU, RHS + t, n, (uint16_t)(jb - t));
                if (rc)
                {
                    linalg_aligned_free(RHS);
                    linalg_aligned_free(LU);
                    linalg_aligned_free(P);
                    return rc;
                }
            }
#endif
        }

        /* Write this tile into Ai_out */
        for (uint16_t r = 0; r < n; ++r)
        {
            float *dst = Ai_out + (size_t)r * n + col0;
            memcpy(dst, RHS + (size_t)r * jb, (size_t)jb * sizeof(float));
        }
    }

    linalg_aligned_free(RHS);
    linalg_aligned_free(LU);
    linalg_aligned_free(P);
    return 0;
}
