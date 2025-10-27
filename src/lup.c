// SPDX-License-Identifier: MIT
/**
 * @file lup_blas3.c
 * @brief Blocked BLAS-3 LU with partial pivoting (single-precision), AVX2/FMA GEMM updates.
 *
 * Factorizes A (row-major, n×n) into P·A = L·U in-place in LU, recording final row
 * permutation in P (P[i] = original row index now at position i). The algorithm is a
 * right-looking, blocked GETRF:
 *   for k = 0..n-1 step NB:
 *     1) Panel factorization: unblocked LU on A[k:n, k:k+ib) with partial pivoting.
 *        Apply row swaps to the entire matrix.
 *     2) Compute U12: A[k:k+ib, k+ib:n] ← L11^{-1} * A[k:k+ib, k+ib:n]  (unit-lower TRSM)
 *     3) Compute L21: A[k+ib:n, k:k+ib] ← A[k+ib:n, k:k+ib] * U11^{-1}  (upper TRSM, right-side)
 *     4) Trailing update: A22 ← A22 − L21·U12  (GEMM; AVX2 8×16 kernel if available)
 *
 * Notes:
 *  - Row-major everywhere; leading dimension is n.
 *  - Small unblocked TRSM is used only on the panel (ib×ib). The big A22 update is BLAS-3.
 *  - P ends as a *final permutation table* (not ipiv steps), compatible with your RHS pivot apply.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <immintrin.h>
#include "linalg_simd.h" // linalg_has_avx2(), linalg_aligned_alloc/free, LINALG_DEFAULT_ALIGNMENT

#ifndef LUP_NB
#define LUP_NB 128 // panel/block size (try 96–160)
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX2 loads");

/* ---------- Utilities ---------- */

static inline void swap_rows(float *RESTRICT A, uint16_t n, uint16_t r1, uint16_t r2)
{
    if (r1 == r2)
        return;
    float *a = A + (size_t)r1 * n;
    float *b = A + (size_t)r2 * n;
    for (uint16_t j = 0; j < n; ++j)
    {
        float t = a[j];
        a[j] = b[j];
        b[j] = t;
    }
}

/* Search pivot row (max |col|) among rows [r0..n-1] in column c. Returns row index. */
static inline uint16_t argmax_abs_col(const float *RESTRICT A, uint16_t n, uint16_t r0, uint16_t c)
{
    uint16_t p = r0;
    float best = fabsf(A[(size_t)r0 * n + c]);
    for (uint16_t r = (uint16_t)(r0 + 1); r < n; ++r)
    {
        float v = fabsf(A[(size_t)r * n + c]);
        if (v > best)
        {
            best = v;
            p = r;
        }
    }
    return p;
}

/* ---------- Small, unblocked panel LU with partial pivoting ----------
   Panel: A[k:n, k:k+ib) — physically swaps rows in A; updates P (final permutation table).
   On exit, panel contains L11 (unit-lower) and U11 (upper).
*/
static int panel_lu_unblocked(float *RESTRICT A, uint16_t n,
                              uint16_t k, uint16_t ib,
                              uint8_t *RESTRICT P)
{
    uint16_t kend = (uint16_t)(k + ib);
    for (uint16_t j = k; j < kend; ++j)
    {
        /* pivot search on column j, rows j..n-1 (global indices) */
        uint16_t piv = argmax_abs_col(A, n, j, j);

        /* swap physical rows j <-> piv across all columns, update final permutation P */
        if (piv != j)
        {
            swap_rows(A, n, j, piv);
            /* P is final permutation table: we must swap the entries that currently map to j/piv.
               The easiest: since we physically swapped rows, also swap P[j] and P[piv]. */
            uint8_t tmp = P[j];
            P[j] = P[piv];
            P[piv] = tmp;
        }

        /* singularity guard on U(j,j) with relative tol */
        float di = A[(size_t)j * n + j];
        float scale = 0.0f;
        for (uint16_t t = j; t < n; ++t)
        {
            float ak = fabsf(A[(size_t)j * n + t]);
            if (ak > scale)
                scale = ak;
        }
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(di) <= tol)
            return -ENOTSUP;

        /* form multipliers in column j (L part) and rank-1 update columns j+1..kend-1 only
           (we’ll finish the rest via BLAS-3 after panel TRSMs) */
        for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
            A[(size_t)r * n + j] /= di;

        /* Update the rest of the panel columns (up to kend) with rank-1 */
        for (uint16_t c = (uint16_t)(j + 1); c < kend; ++c)
        {
            float u = A[(size_t)j * n + c];
            for (uint16_t r = (uint16_t)(j + 1); r < n; ++r)
                A[(size_t)r * n + c] -= A[(size_t)r * n + j] * u;
        }
    }
    return 0;
}

/* ---------- Small TRSM on the panel (scalar) ---------- */

/* U12 := L11^{-1} * U12, where L11 is ib×ib unit-lower; U12 is ib×nc (rows k..k+ib-1, cols c0..). */
static inline void trsm_left_unit_lower_on_U12(float *RESTRICT A, uint16_t n,
                                               uint16_t k, uint16_t ib,
                                               uint16_t c0, uint16_t nc)
{
    for (uint16_t r = 0; r < ib; ++r)
    {
        float *Ur = A + (size_t)(k + r) * n + c0; /* row vector of U12 */
        for (uint16_t t = 0; t < r; ++t)
        {
            float lij = A[(size_t)(k + r) * n + (k + t)];
            if (lij != 0.0f)
            {
                const float *Ut = A + (size_t)(k + t) * n + c0;
                for (uint16_t j = 0; j < nc; ++j)
                    Ur[j] -= lij * Ut[j];
            }
        }
        /* unit diag → no divide */
    }
}

/* L21 := L21 * U11^{-1}, where U11 is ib×ib upper (non-unit); L21 is m2×ib (rows r0.., cols k..k+ib-1).
   Right-side TRSM: process columns c=ib-1..0. */
static inline int trsm_right_upper_on_L21(float *RESTRICT A, uint16_t n,
                                          uint16_t r0, uint16_t m2,
                                          uint16_t k, uint16_t ib)
{
    for (int cc = (int)ib - 1; cc >= 0; --cc)
    {
        uint16_t c = (uint16_t)cc;
        float ucc = A[(size_t)(k + c) * n + (k + c)];
        /* relative guard */
        float scale = 0.0f;
        const float *Urow = A + (size_t)(k + c) * n;
        for (uint16_t t = c; t < ib; ++t)
        {
            float v = fabsf(Urow[k + t]);
            if (v > scale)
                scale = v;
        }
        float tol = (float)ib * FLT_EPSILON * scale;
        if (fabsf(ucc) <= tol)
            return -ENOTSUP;

        float inv = 1.0f / ucc;

        /* divide column c of L21 by U11(c,c) */
        for (uint16_t r = 0; r < m2; ++r)
            A[(size_t)(r0 + r) * n + (k + c)] *= inv;

        /* update: for t = 0..c-1:  L21[:,t] -= L21[:,c] * U11(t,c) */
        for (int tt = 0; tt < cc; ++tt)
        {
            uint16_t t = (uint16_t)tt;
            float u_tc = A[(size_t)(k + t) * n + (k + c)];
            if (u_tc == 0.0f)
                continue;
            for (uint16_t r = 0; r < m2; ++r)
            {
                float *L_r_t = &A[(size_t)(r0 + r) * n + (k + t)];
                *L_r_t -= A[(size_t)(r0 + r) * n + (k + c)] * u_tc;
            }
        }
    }
    return 0;
}

/* ---------- GEMM: AVX2 8×16 kernel + scalar fallback ---------- */
/* Reuse the same 8×16 kernel + packers from your inverse if you already have them.
   For completeness, minimal inline versions are included here. */

#if LINALG_SIMD_ENABLE
/* pack A: mr(=8)×K (rows r0..r0+7, cols k0..k0+K-1) */
static inline void pack_A_mr8_kc(const float *RESTRICT A, uint16_t lda,
                                 uint16_t M, uint16_t K,
                                 uint16_t r0, uint16_t k0,
                                 float *RESTRICT Ap)
{
    const uint16_t mr = 8;
    for (uint16_t r = 0; r < mr; ++r)
    {
        uint16_t i = (uint16_t)(r0 + r);
        const float *src = (i < M) ? (A + (size_t)i * lda + k0) : NULL;
        float *dst = Ap + (size_t)r * K;
        if (src)
            memcpy(dst, src, (size_t)K * sizeof(float));
        else
            memset(dst, 0, (size_t)K * sizeof(float));
    }
}

/* pack B: K×nr(=16) at col-block j0 */
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
            memcpy(dst + 0, row + 0, 8 * sizeof(float));
        else
        {
            uint16_t rem = (uint16_t)(N - j0 < 8 ? N - j0 : 8);
            for (uint16_t t = 0; t < 8; ++t)
                dst[t] = (t < rem) ? row[t] : 0.0f;
        }
        /* next 8 */
        if (j0 + 16 <= N)
            memcpy(dst + 8, row + 8, 8 * sizeof(float));
        else
        {
            uint16_t base = 8;
            uint16_t rem = (uint16_t)(N - (j0 + base) < 8 ? (N - (j0 + base)) : 8);
            for (uint16_t t = 0; t < 8; ++t)
                dst[8 + t] = (t < rem) ? row[base + t] : 0.0f;
        }
    }
}

/* 8×16 kernel from earlier (U2 pipelined). op_minus: 0 → C+=AB, 1 → C-=AB */
static inline void sgemm_8x16_u2_fma_avx2(const float *RESTRICT A, const float *RESTRICT B,
                                          float *RESTRICT C, uint16_t kc, uint16_t ldc, int op_minus)
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
        __m256 a0k = _mm256_set1_ps((A + 0 * kc)[k]), a1k = _mm256_set1_ps((A + 1 * kc)[k]);
        __m256 a2k = _mm256_set1_ps((A + 2 * kc)[k]), a3k = _mm256_set1_ps((A + 3 * kc)[k]);
        __m256 a4k = _mm256_set1_ps((A + 4 * kc)[k]), a5k = _mm256_set1_ps((A + 5 * kc)[k]);
        __m256 a6k = _mm256_set1_ps((A + 6 * kc)[k]), a7k = _mm256_set1_ps((A + 7 * kc)[k]);

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

#define UPD_ROW(cp, v0, v1)                                                  \
    do                                                                       \
    {                                                                        \
        __m256 d0 = _mm256_load_ps((cp) + 0), d1 = _mm256_load_ps((cp) + 8); \
        if (op_minus)                                                        \
        {                                                                    \
            d0 = _mm256_sub_ps(d0, (v0));                                    \
            d1 = _mm256_sub_ps(d1, (v1));                                    \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            d0 = _mm256_add_ps(d0, (v0));                                    \
            d1 = _mm256_add_ps(d1, (v1));                                    \
        }                                                                    \
        _mm256_store_ps((cp) + 0, d0);                                       \
        _mm256_store_ps((cp) + 8, d1);                                       \
    } while (0)

    float *c0 = C + 0 * ldc, *c1 = C + 1 * ldc, *c2 = C + 2 * ldc, *c3 = C + 3 * ldc;
    float *c4 = C + 4 * ldc, *c5 = C + 5 * ldc, *c6 = C + 6 * ldc, *c7 = C + 7 * ldc;
    UPD_ROW(c0, c00, c01);
    UPD_ROW(c1, c10, c11);
    UPD_ROW(c2, c20, c21);
    UPD_ROW(c3, c30, c31);
    UPD_ROW(c4, c40, c41);
    UPD_ROW(c5, c50, c51);
    UPD_ROW(c6, c60, c61);
    UPD_ROW(c7, c70, c71);
#undef UPD_ROW
}

/* GEMM driver MxN with K (C +=/-= A·B), AVX2 for 8×16 tiles, scalar tails. */
static inline void gemm_mkn_avx8x16(const float *RESTRICT A, uint16_t lda,
                                    const float *RESTRICT B, uint16_t ldb,
                                    float *RESTRICT C, uint16_t ldc,
                                    uint16_t M, uint16_t N, uint16_t K, int op_minus)
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
            pack_A_mr8_kc(A, lda, M, K, i, 0, Ap);
            sgemm_8x16_u2_fma_avx2(Ap, Bp, C + (size_t)i * ldc + j, K, ldc, op_minus);
        }
    }
    /* N tail (<16) scalar */
    for (uint16_t i = 0; i < M; ++i)
    {
        for (uint16_t jj = j; jj < N; ++jj)
        {
            float acc = 0.0f;
            for (uint16_t k = 0; k < K; ++k)
                acc += A[(size_t)i * lda + k] * B[(size_t)k * ldb + jj];
            if (op_minus)
                C[(size_t)i * ldc + jj] -= acc;
            else
                C[(size_t)i * ldc + jj] += acc;
        }
    }

    linalg_aligned_free(Ap);
    linalg_aligned_free(Bp);
}
#endif /* LINALG_SIMD_ENABLE */

static inline void gemm_mkn_scalar(const float *RESTRICT A, uint16_t lda,
                                   const float *RESTRICT B, uint16_t ldb,
                                   float *RESTRICT C, uint16_t ldc,
                                   uint16_t M, uint16_t N, uint16_t K, int op_minus)
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

/* ---------- Public API: blocked BLAS-3 LUP ---------- */
int lup(const float *RESTRICT A, float *RESTRICT LU, uint8_t *P, uint16_t n)
{
    if (n == 0)
        return -EINVAL;
    if (A != LU)
        memcpy(LU, A, (size_t)n * n * sizeof(float));
    for (uint16_t i = 0; i < n; ++i)
        P[i] = (uint8_t)i;

    const uint16_t NB = (uint16_t)LUP_NB;

    for (uint16_t k = 0; k < n; k = (uint16_t)(k + NB))
    {
        uint16_t ib = (uint16_t)((k + NB <= n) ? NB : (n - k));
        uint16_t nc = (uint16_t)(n - (k + ib)); /* columns to the right of panel */
        uint16_t m2 = (uint16_t)(n - (k + ib)); /* rows below panel (same value here since square) */

        /* 1) Panel unblocked LU with partial pivoting (updates P and LU in-place) */
        int rc = panel_lu_unblocked(LU, n, k, ib, P);
        if (rc)
            return rc;

        /* 2) U12: solve L11^{-1} * U12 (unit-lower TRSM on the left) */
        if (nc)
        {
            trsm_left_unit_lower_on_U12(LU, n, k, ib, (uint16_t)(k + ib), nc);
        }

        /* 3) L21: solve L21 * U11^{-1} (upper TRSM on the right) */
        if (m2)
        {
            rc = trsm_right_upper_on_L21(LU, n, (uint16_t)(k + ib), m2, k, ib);
            if (rc)
                return rc;
        }

        /* 4) Trailing update: A22 ← A22 − L21·U12  (m2×nc ← m2×ib · ib×nc) */
        if (m2 && nc)
        {
            const float *L21 = LU + (size_t)(k + ib) * n + k;
            const float *U12 = LU + (size_t)k * n + (k + ib);
            float *A22 = LU + (size_t)(k + ib) * n + (k + ib);

#if LINALG_SIMD_ENABLE
            if (linalg_has_avx2())
                gemm_mkn_avx8x16(L21, n, U12, n, A22, n, m2, nc, ib, /*minus*/ 1);
            else
#endif
                gemm_mkn_scalar(L21, n, U12, n, A22, n, m2, nc, ib, 1);
        }
    }

    return 0;
}
