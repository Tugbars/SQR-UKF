// SPDX-License-Identifier: MIT
/**
 * @file inv_blas3.c
 * @brief Blocked BLAS-3 inverse via LU (single-precision), with AVX2/FMA 8×16 GEMM microkernel.
 *
 * @details
 * Computes A^{-1} by:
 *   1) LU with partial pivoting:  P A = L U      (external lup()).
 *   2) Solve A X = I in big RHS tiles (BLAS-3 GETRS style):
 *        - Apply pivots once to the RHS tile (RHS ← P·RHS).
 *        - Forward solve:  L · Y = RHS    (unit-lower, **blocked TRSM**).
 *        - Backward solve: U · X = Y      (upper, non-unit, **blocked TRSM**).
 *      The trailing updates are done with GEMM (C ← C − A·B), using an AVX2/FMA 8×16 kernel.
 *
 *  – Row-major storage everywhere.
 *  – Single-threaded core; wrap tiles in threads if you want.
 *  – Robust relative singularity checks on U’s diagonal.
 *
 * Tunables:
 *  - INV_NRHS_TILE  : RHS tile width (N_col per sweep). Try 64–192 (must be ≥16 for full 8×16 use).
 *  - INV_JC_BLOCK   : Inner accumulation block for dot products in scalar paths (192–384).
 *  - INV_NB_PANEL   : Triangular panel block size (e.g., 128). Unblocked TRSM only on these diag blocks.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <errno.h>
#include <float.h>
#include <math.h>

#include "linalg_simd.h" // lup(), linalg_has_avx2(), linalg_aligned_alloc/free, LINALG_SMALL_N_THRESH, LINALG_DEFAULT_ALIGNMENT

#ifndef INV_NRHS_TILE
#define INV_NRHS_TILE 128 /* width of RHS tile; ≥16 recommended for full 8×16 kernel */
#endif

#ifndef INV_JC_BLOCK
#define INV_JC_BLOCK 256 /* scalar accumulate blocking (dot-product breakup) */
#endif

#ifndef INV_NB_PANEL
#define INV_NB_PANEL 128 /* triangular panel block size (diag subproblem size) */
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX2 loads");

/* =========================================================================================
 * Pivot application: RHS ← P · RHS   (P is swap list from GETRF/LUP)
 * ========================================================================================= */
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

/* =========================================================================================
 * Tiny unblocked TRSM on diagonal blocks (scalar)
 * ========================================================================================= */

/* Forward (unit-lower): solve L_ii (nb×nb, unit diag) * X = B (nb×jb). */
static void trsm_ll_unit_unblocked_scalar(const float *RESTRICT Lii, uint16_t n, /* full ld = n */
                                          float *RESTRICT B, uint16_t nb, uint16_t jb)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;
    for (uint16_t i = 0; i < nb; ++i)
    {
        const float *Li = Lii + (size_t)i * n;
        float *Bi = B + (size_t)i * jb;

        for (uint16_t j0 = 0; j0 < i; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < i) ? (j0 + Jc) : i);
            for (uint16_t j = j0; j < j1; ++j)
            {
                float lij = Li[j];
                if (lij != 0.0f)
                {
                    const float *Bj = B + (size_t)j * jb;
                    for (uint16_t c = 0; c < jb; ++c)
                        Bi[c] -= lij * Bj[c];
                }
            }
        }
        /* unit diagonal → no divide */
    }
}

/* Backward (upper, non-unit): solve U_ii (nb×nb) * X = B (nb×jb). */
static int trsm_uu_nonunit_unblocked_scalar(const float *RESTRICT Uii, uint16_t n,
                                            float *RESTRICT B, uint16_t nb, uint16_t jb)
{
    const uint16_t Jc = (uint16_t)INV_JC_BLOCK;

    for (int ii = (int)nb - 1; ii >= 0; --ii)
    {
        uint16_t i = (uint16_t)ii;
        const float *Ui = Uii + (size_t)i * n;
        float *Bi = B + (size_t)i * jb;

        /* Bi -= sum_{j>i} U(i,j) * B[j,:] */
        for (uint16_t j0 = (uint16_t)i + 1; j0 < nb; j0 += Jc)
        {
            uint16_t j1 = (uint16_t)((j0 + Jc < nb) ? (j0 + Jc) : nb);
            for (uint16_t j = j0; j < j1; ++j)
            {
                float uij = Ui[j];
                if (uij != 0.0f)
                {
                    const float *Bj = B + (size_t)j * jb;
                    for (uint16_t c = 0; c < jb; ++c)
                        Bi[c] -= uij * Bj[c];
                }
            }
        }

        /* divide by U(i,i) with relative tolerance */
        float scale = 0.0f;
        for (uint16_t k = i; k < (uint16_t)nb; ++k)
        {
            float ak = fabsf(Ui[k]);
            if (ak > scale)
                scale = ak;
        }
        float di = Ui[i];
        float tol = (float)nb * FLT_EPSILON * scale;
        if (fabsf(di) <= tol)
            return -ENOTSUP;

        float invd = 1.0f / di;
        for (uint16_t c = 0; c < jb; ++c)
            Bi[c] *= invd;
    }
    return 0;
}

/* =========================================================================================
 * GEMM microkernel (8×16, U2 pipelined) + small packers
 * ========================================================================================= */

#if LINALG_SIMD_ENABLE
/* A pack:  up to 8 rows × kc, row-major per row r:  Ap[r*kc + k] = A[row+i+r, k0+k]  */
static inline void pack_A_mr8_kc(const float *RESTRICT A, uint16_t lda,
                                 uint16_t M, uint16_t K,
                                 uint16_t i0, uint16_t k0,
                                 float *RESTRICT Ap)
{
    const uint16_t mr = 8;
    for (uint16_t r = 0; r < mr; ++r)
    {
        uint16_t i = (uint16_t)(i0 + r);
        const float *src = (i < M) ? (A + (size_t)i * lda + k0) : NULL;
        float *dst = Ap + (size_t)r * K;
        if (src)
        {
            memcpy(dst, src, (size_t)K * sizeof(float));
        }
        else
        {
            memset(dst, 0, (size_t)K * sizeof(float));
        }
    }
}

/* B pack: kc × 16 into layout per k: [8 cols][8 cols]
 * Bp[k*16 + 0..7]   = B[k0+k, j0+0..7]
 * Bp[k*16 + 8..15]  = B[k0+k, j0+8..15]
 */
static inline void pack_B_kc_nr16(const float *RESTRICT B, uint16_t ldb,
                                  uint16_t K, uint16_t N,
                                  uint16_t k0, uint16_t j0,
                                  float *RESTRICT Bp)
{
    for (uint16_t kk = 0; kk < K; ++kk)
    {
        const float *row = B + (size_t)(k0 + kk) * ldb + j0;
        float *dst = Bp + (size_t)kk * 16;
        /* first 8 */
        if (j0 + 8 <= N)
        {
            memcpy(dst + 0, row + 0, 8 * sizeof(float));
        }
        else
        {
            uint16_t rem = (uint16_t)(N - j0 < 8 ? N - j0 : 8);
            for (uint16_t t = 0; t < 8; ++t)
                dst[t] = (t < rem) ? row[t] : 0.0f;
        }
        /* next 8 */
        if (j0 + 16 <= N)
        {
            memcpy(dst + 8, row + 8, 8 * sizeof(float));
        }
        else
        {
            uint16_t base = 8;
            uint16_t rem = (uint16_t)(N - (j0 + base) < 8 ? (N - (j0 + base)) : 8);
            for (uint16_t t = 0; t < 8; ++t)
                dst[8 + t] = (t < rem) ? row[base + t] : 0.0f;
        }
    }
}

/* C update microkernel: 8×16, kc arbitrary; op_minus = 0 → C+=AB, 1 → C-=AB
 * C is row-major with ldc in elements; we assume j0..j0+15 are contiguous (so use two aligned 8-loads/stores).
 */
static inline void sgemm_8x16_u2_fma_avx2(const float *RESTRICT A, /* mr×kc (packed) */
                                          const float *RESTRICT B, /* kc×nr (packed) */
                                          float *RESTRICT C,       /* mr×nr (C[i0..i0+7, j0..j0+15]) */
                                          uint16_t kc, uint16_t ldc,
                                          int op_minus)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps(), c61 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps(), c71 = _mm256_setzero_ps();

    const float *a0 = A + 0 * kc, *a1 = A + 1 * kc, *a2 = A + 2 * kc, *a3 = A + 3 * kc;
    const float *a4 = A + 4 * kc, *a5 = A + 5 * kc, *a6 = A + 6 * kc, *a7 = A + 7 * kc;
    const float *bk = B;

    uint16_t k = 0;
    if (kc >= 2)
    {
        __m256 b0_0 = _mm256_load_ps(bk + 0), b0_1 = _mm256_load_ps(bk + 8);
        __m256 b1_0 = _mm256_load_ps(bk + 16), b1_1 = _mm256_load_ps(bk + 24);

        for (; k + 1 < kc; k += 2)
        {
            __m256 a0k = _mm256_set1_ps(a0[k]), a1k = _mm256_set1_ps(a1[k]);
            __m256 a2k = _mm256_set1_ps(a2[k]), a3k = _mm256_set1_ps(a3[k]);
            __m256 a4k = _mm256_set1_ps(a4[k]), a5k = _mm256_set1_ps(a5[k]);
            __m256 a6k = _mm256_set1_ps(a6[k]), a7k = _mm256_set1_ps(a7[k]);

            c00 = _mm256_fmadd_ps(a0k, b0_0, c00);
            c01 = _mm256_fmadd_ps(a0k, b0_1, c01);
            c10 = _mm256_fmadd_ps(a1k, b0_0, c10);
            c11 = _mm256_fmadd_ps(a1k, b0_1, c11);
            c20 = _mm256_fmadd_ps(a2k, b0_0, c20);
            c21 = _mm256_fmadd_ps(a2k, b0_1, c21);
            c30 = _mm256_fmadd_ps(a3k, b0_0, c30);
            c31 = _mm256_fmadd_ps(a3k, b0_1, c31);
            c40 = _mm256_fmadd_ps(a4k, b0_0, c40);
            c41 = _mm256_fmadd_ps(a4k, b0_1, c41);
            c50 = _mm256_fmadd_ps(a5k, b0_0, c50);
            c51 = _mm256_fmadd_ps(a5k, b0_1, c51);
            c60 = _mm256_fmadd_ps(a6k, b0_0, c60);
            c61 = _mm256_fmadd_ps(a6k, b0_1, c61);
            c70 = _mm256_fmadd_ps(a7k, b0_0, c70);
            c71 = _mm256_fmadd_ps(a7k, b0_1, c71);

            __m256 a0k1 = _mm256_set1_ps(a0[k + 1]), a1k1 = _mm256_set1_ps(a1[k + 1]);
            __m256 a2k1 = _mm256_set1_ps(a2[k + 1]), a3k1 = _mm256_set1_ps(a3[k + 1]);
            __m256 a4k1 = _mm256_set1_ps(a4[k + 1]), a5k1 = _mm256_set1_ps(a5[k + 1]);
            __m256 a6k1 = _mm256_set1_ps(a6[k + 1]), a7k1 = _mm256_set1_ps(a7[k + 1]);

            c00 = _mm256_fmadd_ps(a0k1, b1_0, c00);
            c01 = _mm256_fmadd_ps(a0k1, b1_1, c01);
            c10 = _mm256_fmadd_ps(a1k1, b1_0, c10);
            c11 = _mm256_fmadd_ps(a1k1, b1_1, c11);
            c20 = _mm256_fmadd_ps(a2k1, b1_0, c20);
            c21 = _mm256_fmadd_ps(a2k1, b1_1, c21);
            c30 = _mm256_fmadd_ps(a3k1, b1_0, c30);
            c31 = _mm256_fmadd_ps(a3k1, b1_1, c31);
            c40 = _mm256_fmadd_ps(a4k1, b1_0, c40);
            c41 = _mm256_fmadd_ps(a4k1, b1_1, c41);
            c50 = _mm256_fmadd_ps(a5k1, b1_0, c50);
            c51 = _mm256_fmadd_ps(a5k1, b1_1, c51);
            c60 = _mm256_fmadd_ps(a6k1, b1_0, c60);
            c61 = _mm256_fmadd_ps(a6k1, b1_1, c61);
            c70 = _mm256_fmadd_ps(a7k1, b1_0, c70);
            c71 = _mm256_fmadd_ps(a7k1, b1_1, c71);

            bk += 32;
            if (k + 2 < kc)
            {
                b0_0 = _mm256_load_ps(bk + 0);
                b0_1 = _mm256_load_ps(bk + 8);
                b1_0 = _mm256_load_ps(bk + 16);
                b1_1 = _mm256_load_ps(bk + 24);
            }
        }
    }
    if (k < kc)
    {
        __m256 b0 = _mm256_load_ps(bk + 0), b1 = _mm256_load_ps(bk + 8);
        __m256 a0k = _mm256_set1_ps(a0[k]), a1k = _mm256_set1_ps(a1[k]);
        __m256 a2k = _mm256_set1_ps(a2[k]), a3k = _mm256_set1_ps(a3[k]);
        __m256 a4k = _mm256_set1_ps(a4[k]), a5k = _mm256_set1_ps(a5[k]);
        __m256 a6k = _mm256_set1_ps(a6[k]), a7k = _mm256_set1_ps(a7[k]);

        c00 = _mm256_fmadd_ps(a0k, b0, c00);
        c01 = _mm256_fmadd_ps(a0k, b1, c01);
        c10 = _mm256_fmadd_ps(a1k, b0, c10);
        c11 = _mm256_fmadd_ps(a1k, b1, c11);
        c20 = _mm256_fmadd_ps(a2k, b0, c20);
        c21 = _mm256_fmadd_ps(a2k, b1, c21);
        c30 = _mm256_fmadd_ps(a3k, b0, c30);
        c31 = _mm256_fmadd_ps(a3k, b1, c31);
        c40 = _mm256_fmadd_ps(a4k, b0, c40);
        c41 = _mm256_fmadd_ps(a4k, b1, c41);
        c50 = _mm256_fmadd_ps(a5k, b0, c50);
        c51 = _mm256_fmadd_ps(a5k, b1, c51);
        c60 = _mm256_fmadd_ps(a6k, b0, c60);
        c61 = _mm256_fmadd_ps(a6k, b1, c61);
        c70 = _mm256_fmadd_ps(a7k, b0, c70);
        c71 = _mm256_fmadd_ps(a7k, b1, c71);
    }

    /* writeback: add/sub to C rows */
    float *c0 = C + 0 * ldc, *c1 = C + 1 * ldc, *c2 = C + 2 * ldc, *c3 = C + 3 * ldc;
    float *c4 = C + 4 * ldc, *c5 = C + 5 * ldc, *c6 = C + 6 * ldc, *c7 = C + 7 * ldc;

#define UPDATE_ROW(cp, v0, v1)                \
    do                                        \
    {                                         \
        __m256 d0 = _mm256_load_ps((cp) + 0); \
        __m256 d1 = _mm256_load_ps((cp) + 8); \
        if (op_minus)                         \
        {                                     \
            d0 = _mm256_sub_ps(d0, (v0));     \
            d1 = _mm256_sub_ps(d1, (v1));     \
        }                                     \
        else                                  \
        {                                     \
            d0 = _mm256_add_ps(d0, (v0));     \
            d1 = _mm256_add_ps(d1, (v1));     \
        }                                     \
        _mm256_store_ps((cp) + 0, d0);        \
        _mm256_store_ps((cp) + 8, d1);        \
    } while (0)

    UPDATE_ROW(c0, c00, c01);
    UPDATE_ROW(c1, c10, c11);
    UPDATE_ROW(c2, c20, c21);
    UPDATE_ROW(c3, c30, c31);
    UPDATE_ROW(c4, c40, c41);
    UPDATE_ROW(c5, c50, c51);
    UPDATE_ROW(c6, c60, c61);
    UPDATE_ROW(c7, c70, c71);

#undef UPDATE_ROW
}

/* GEMM driver (A: M×K, B: K×N, C: M×N), AVX2 path for 8×16 tiles; tails → scalar. */
static inline void gemm_mkn_avx8x16(const float *RESTRICT A, uint16_t lda,
                                    const float *RESTRICT B, uint16_t ldb,
                                    float *RESTRICT C, uint16_t ldc,
                                    uint16_t M, uint16_t N, uint16_t K,
                                    int op_minus)
{
    const uint16_t mr = 8, nr = 16;
    if (M == 0 || N == 0 || K == 0)
        return;

    float *Ap = (float *)linalg_aligned_alloc(32, (size_t)mr * K * sizeof(float));
    float *Bp = (float *)linalg_aligned_alloc(32, (size_t)K * nr * sizeof(float));
    if (!Ap || !Bp)
    {
        if (Ap)
            linalg_aligned_free(Ap);
        if (Bp)
            linalg_aligned_free(Bp);
        return;
    }

    uint16_t j = 0;
    for (; j + nr - 1 < N; j += nr)
    {
        pack_B_kc_nr16(B, ldb, K, N, 0, j, Bp);
        for (uint16_t i = 0; i < M; i += mr)
        {
            uint16_t mi = (uint16_t)((i + mr <= M) ? mr : (M - i));
            /* initialize C block by reading existing values (kernel does add/sub in place) */
            /* pack A block */
            pack_A_mr8_kc(A, lda, M, K, i, 0, Ap);
            /* compute */
            sgemm_8x16_u2_fma_avx2(Ap, Bp, C + (size_t)i * ldc + j, K, ldc, op_minus);
            /* tail rows mi<8 are handled by zero padding in pack_A */
        }
    }
    /* N tail (<16): use scalar GEMM (compact and safe) */
    if (j < N)
    {
        for (uint16_t ii = 0; ii < M; ++ii)
        {
            for (uint16_t jj = j; jj < N; ++jj)
            {
                float acc = 0.0f;
                for (uint16_t kk = 0; kk < K; ++kk)
                    acc += A[(size_t)ii * lda + kk] * B[(size_t)kk * ldb + jj];
                if (op_minus)
                    C[(size_t)ii * ldc + jj] -= acc;
                else
                    C[(size_t)ii * ldc + jj] += acc;
            }
        }
    }

    linalg_aligned_free(Ap);
    linalg_aligned_free(Bp);
}
#endif /* LINALG_SIMD_ENABLE */

/* Scalar GEMM fallback (C +=/-= A·B) */
static inline void gemm_mkn_scalar(const float *RESTRICT A, uint16_t lda,
                                   const float *RESTRICT B, uint16_t ldb,
                                   float *RESTRICT C, uint16_t ldc,
                                   uint16_t M, uint16_t N, uint16_t K,
                                   int op_minus)
{
    for (uint16_t i = 0; i < M; ++i)
    {
        for (uint16_t j = 0; j < N; ++j)
        {
            float acc = 0.0f;
            for (uint16_t k = 0; k < K; ++k)
                acc += A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
            if (op_minus)
                C[(size_t)i * ldc + j] -= acc;
            else
                C[(size_t)i * ldc + j] += acc;
        }
    }
}

/* =========================================================================================
 * Blocked TRSM using GEMM updates (BLAS-3)
 * ========================================================================================= */

/* Y = inv(L) * RHS,  L is n×n unit-lower (from LU), RHS is n×jb. */
static void forward_trsm_blocked_L(const float *RESTRICT LU, uint16_t n,
                                   float *RESTRICT RHS, uint16_t jb)
{
    const uint16_t nb = (uint16_t)INV_NB_PANEL;

    for (uint16_t i0 = 0; i0 < n; i0 += nb)
    {
        uint16_t ib = (uint16_t)((i0 + nb <= n) ? nb : (n - i0));

        /* 1) Solve L_ii * B_i = B_i (nb×nb unit-lower vs nb×jb) */
        trsm_ll_unit_unblocked_scalar(LU + (size_t)i0 * n + i0, n,
                                      RHS + (size_t)i0 * jb, ib, jb);

        /* 2) Trailing update: B_{i0+ib:} -= L_{21} * B_i  */
        uint16_t m2 = (uint16_t)(n - (i0 + ib));
        if (m2)
        {
            const float *L21 = LU + (size_t)(i0 + ib) * n + i0; /* m2×ib */
            float *B2 = RHS + (size_t)(i0 + ib) * jb;           /* m2×jb */
#if LINALG_SIMD_ENABLE
            if (linalg_has_avx2())
                gemm_mkn_avx8x16(L21, n, RHS + (size_t)i0 * jb, jb, B2, jb, m2, jb, ib, /*minus*/ 1);
            else
#endif
                gemm_mkn_scalar(L21, n, RHS + (size_t)i0 * jb, jb, B2, jb, m2, jb, ib, 1);
        }
    }
}

/* X = inv(U) * RHS,  U is n×n upper (non-unit), RHS is n×jb. */
static int backward_trsm_blocked_U(const float *RESTRICT LU, uint16_t n,
                                   float *RESTRICT RHS, uint16_t jb)
{
    const uint16_t nb = (uint16_t)INV_NB_PANEL;

    for (int ii0 = (int)n - 1; ii0 >= 0; ii0 -= (int)nb)
    {
        uint16_t i0 = (uint16_t)((ii0 + 1 >= (int)nb) ? (ii0 + 1 - nb) : 0);
        uint16_t ib = (uint16_t)(ii0 - (int)i0 + 1);

        /* 1) Solve U_ii * B_i = B_i  (nb×nb upper) */
        int rc = trsm_uu_nonunit_unblocked_scalar(LU + (size_t)i0 * n + i0, n,
                                                  RHS + (size_t)i0 * jb, ib, jb);
        if (rc)
            return rc;

        /* 2) Update above rows: B_{0:i0} -= U_{01} * B_i */
        if (i0 > 0)
        {
            const float *U01 = LU + (size_t)0 * n + i0; /* i0×ib (upper-right part above diag block) */
            float *B0 = RHS + (size_t)0 * jb;           /* i0×jb */
#if LINALG_SIMD_ENABLE
            if (linalg_has_avx2())
                gemm_mkn_avx8x16(U01, n, RHS + (size_t)i0 * jb, jb, B0, jb, i0, jb, ib, /*minus*/ 1);
            else
#endif
                gemm_mkn_scalar(U01, n, RHS + (size_t)i0 * jb, jb, B0, jb, i0, jb, ib, 1);
        }
    }
    return 0;
}

/* =========================================================================================
 * Public: inv() — blocked BLAS-3 GETRS/GETRI-like inverse using LU
 * ========================================================================================= */
int inv(float *RESTRICT Ai_out, const float *RESTRICT A, uint16_t n)
{
    if (n == 0)
        return -EINVAL;

    /* Tiny: scalar path is often faster. */
    if (n < LINALG_SMALL_N_THRESH)
    {
        float LU[(size_t)n * n];
        uint8_t P[n];
        if (lup(A, LU, P, n) != 0)
            return -ENOTSUP;

        const uint16_t tile = 32;
        float *RHS = (float *)linalg_aligned_alloc(32, (size_t)n * tile * sizeof(float));
        if (!RHS)
            return -ENOMEM;

        for (uint16_t col0 = 0; col0 < n; col0 += tile)
        {
            uint16_t jb = (uint16_t)((col0 + tile <= n) ? tile : (n - col0));

            memset(RHS, 0, (size_t)n * jb * sizeof(float));
            for (uint16_t t = 0; t < jb; ++t)
                RHS[(size_t)(col0 + t) * jb + t] = 1.0f;

            apply_pivots_to_rhs(RHS, n, jb, P);

            forward_trsm_blocked_L(LU, n, RHS, jb);
            int rc = backward_trsm_blocked_U(LU, n, RHS, jb);
            if (rc)
            {
                linalg_aligned_free(RHS);
                return rc;
            }

            for (uint16_t r = 0; r < n; ++r)
                memcpy(Ai_out + (size_t)r * n + col0, RHS + (size_t)r * jb, (size_t)jb * sizeof(float));
        }
        linalg_aligned_free(RHS);
        return 0;
    }

    /* General case: LU + blocked BLAS-3 solves with AVX2 GEMM where available */
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

    if (lup(A, LU, P, n) != 0)
    {
        linalg_aligned_free(LU);
        linalg_aligned_free(P);
        return -ENOTSUP;
    }

    const uint16_t NRHS = (uint16_t)INV_NRHS_TILE;
    float *RHS = (float *)linalg_aligned_alloc(32, (size_t)n * NRHS * sizeof(float));
    if (!RHS)
    {
        linalg_aligned_free(LU);
        linalg_aligned_free(P);
        return -ENOMEM;
    }

    /* Sweep RHS in tiles (columns of identity) */
    for (uint16_t col0 = 0; col0 < n; col0 += NRHS)
    {
        uint16_t jb = (uint16_t)((col0 + NRHS <= n) ? NRHS : (n - col0));

        /* Build RHS = I(:, col0:col0+jb-1) */
        memset(RHS, 0, (size_t)n * jb * sizeof(float));
        for (uint16_t t = 0; t < jb; ++t)
            RHS[(size_t)(col0 + t) * jb + t] = 1.0f;

        /* Apply pivots once to the whole RHS tile */
        apply_pivots_to_rhs(RHS, n, jb, P);

        /* Blocked TRSMs with GEMM updates (AVX if available inside helpers) */
        forward_trsm_blocked_L(LU, n, RHS, jb);
        int rc = backward_trsm_blocked_U(LU, n, RHS, jb);
        if (rc)
        {
            linalg_aligned_free(RHS);
            linalg_aligned_free(LU);
            linalg_aligned_free(P);
            return rc;
        }

        /* Scatter into output inverse */
        for (uint16_t r = 0; r < n; ++r)
        {
            memcpy(Ai_out + (size_t)r * n + col0, RHS + (size_t)r * jb, (size_t)jb * sizeof(float));
        }
    }

    linalg_aligned_free(RHS);
    linalg_aligned_free(LU);
    linalg_aligned_free(P);
    return 0;
}
