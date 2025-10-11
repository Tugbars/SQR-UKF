// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <immintrin.h>
#include <string.h>

#include "linalg_simd.h" // linalg_has_avx2(), LINALG_BLOCK_KC, LINALG_BLOCK_JC, RESTRICT

#ifndef LINALG_BLOCK_MC
#define LINALG_BLOCK_MC 256
#endif

#ifndef LINALG_SMALL_N_THRESH
#define LINALG_SMALL_N_THRESH 64
#endif

/* ======================= Packing ======================= */

/* Pack B tile (Kblk x jb) from row-major B into contiguous 8-col panels.
 * Layout: for each 8-wide panel, row k is at panel_base + k*8.
 * Zero-pad the last panel’s short columns.
 */
static inline void
pack_B_8col_tile(float *RESTRICT Bp,
                 const float *RESTRICT B,
                 size_t K, size_t N,
                 size_t kk, size_t Kblk,
                 size_t j0, size_t jb)
{
    const size_t n_panels = (jb + 7) / 8;
    size_t off = 0;
    for (size_t p = 0, j = j0; p < n_panels; ++p, j += 8)
    {
        const size_t w = (j + 8 <= j0 + jb) ? 8 : (j0 + jb - j);
        for (size_t k = 0; k < Kblk; ++k)
        {
            const float *src = B + (kk + k) * N + j;
            float *dst = Bp + off + k * 8;

            size_t t = 0;
            for (; t < w; ++t)
                dst[t] = src[t];
            for (; t < 8; ++t)
                dst[t] = 0.0f; // pad short panel cols
        }
        off += Kblk * 8;
    }
}

/* Pack A tile (ib x Kblk) from row-major A into 8-row-interleaved form.
 * For each k in Kblk, we store up to 8 row scalars contiguously:
 *    Ap[k*8 + r] = A[(i0+r), (kk+k)], r=0..ib-1; zero-pad to 8 rows.
 */
static inline void
pack_A_8row_tile(float *RESTRICT Ap,
                 const float *RESTRICT A,
                 size_t M, size_t K,
                 size_t i0, size_t ib,
                 size_t kk, size_t Kblk)
{
    (void)M; // bounds are guaranteed by caller
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 8;
        size_t r = 0;
        for (; r < ib; ++r)
        {
            const size_t i = i0 + r;
            dst[r] = A[i * K + (kk + k)];
        }
        for (; r < 8; ++r)
            dst[r] = 0.0f; // pad short row block
    }
}

/* ======================= Micro-kernels (AVX2/FMA) ======================= */

#if LINALG_SIMD_ENABLE
/* 8x8: accumulates into C (+=). Bp is Kblk×8 panel; Ap is Kblk×8 row-interleaved. */
static inline void
gemm_8x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
#define STEP(KOFF)                                                                       \
    do                                                                                   \
    {                                                                                    \
        const __m256 b = _mm256_load_ps(Bp + (k + (KOFF)) * 8); /* Bp is 32B-aligned */  \
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 0), b, acc0); \
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 1), b, acc1); \
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 2), b, acc2); \
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 3), b, acc3); \
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 4), b, acc4); \
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 5), b, acc5); \
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 6), b, acc6); \
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (KOFF)) * 8 + 7), b, acc7); \
    } while (0)
        STEP(0);
        STEP(1);
        STEP(2);
        STEP(3);
        STEP(4);
        STEP(5);
        STEP(6);
        STEP(7);
#undef STEP
    }
    for (; k < Kblk; ++k)
    {
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 4), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 5), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 6), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 7), b, acc7);
    }

    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 0 * ldc), acc0));
        _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 1 * ldc), acc1));
        _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), acc2));
        _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), acc3));
        _mm256_storeu_ps(c + 4 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 4 * ldc), acc4));
        _mm256_storeu_ps(c + 5 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 5 * ldc), acc5));
        _mm256_storeu_ps(c + 6 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 6 * ldc), acc6));
        _mm256_storeu_ps(c + 7 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 7 * ldc), acc7));
    }
    else
    {
        alignas(32) float buf[8];
#define ADD_TAIL(ROW, ACCV)             \
    do                                  \
    {                                   \
        _mm256_store_ps(buf, (ACCV));   \
        float *cd = c + (ROW) * ldc;    \
        for (size_t t = 0; t < jb; ++t) \
            cd[t] += buf[t];            \
    } while (0)
        ADD_TAIL(0, acc0);
        ADD_TAIL(1, acc1);
        ADD_TAIL(2, acc2);
        ADD_TAIL(3, acc3);
        ADD_TAIL(4, acc4);
        ADD_TAIL(5, acc5);
        ADD_TAIL(6, acc6);
        ADD_TAIL(7, acc7);
#undef ADD_TAIL
    }
}

/* 4x8 kernel for <=4 leftover rows. */
static inline void
gemm_4x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
#define STEP(T)                                                                       \
    do                                                                                \
    {                                                                                 \
        const __m256 b = _mm256_load_ps(Bp + (k + (T)) * 8);                          \
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (T)) * 8 + 0), b, acc0); \
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (T)) * 8 + 1), b, acc1); \
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (T)) * 8 + 2), b, acc2); \
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (T)) * 8 + 3), b, acc3); \
    } while (0)
        STEP(0);
        STEP(1);
        STEP(2);
        STEP(3);
        STEP(4);
        STEP(5);
        STEP(6);
        STEP(7);
#undef STEP
    }
    for (; k < Kblk; ++k)
    {
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
    }

    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 0 * ldc), acc0));
        _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 1 * ldc), acc1));
        _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), acc2));
        _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), acc3));
    }
    else
    {
        alignas(32) float buf[8];
#define ADD_TAIL4(ROW, ACCV)            \
    do                                  \
    {                                   \
        _mm256_store_ps(buf, (ACCV));   \
        float *cd = c + (ROW) * ldc;    \
        for (size_t t = 0; t < jb; ++t) \
            cd[t] += buf[t];            \
    } while (0)
        ADD_TAIL4(0, acc0);
        ADD_TAIL4(1, acc1);
        ADD_TAIL4(2, acc2);
        ADD_TAIL4(3, acc3);
#undef ADD_TAIL4
    }
}

/* 1x8: single row leftover. */
static inline void
gemm_1x8_panel_avx2fma_add(float *RESTRICT c,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb /*<=8*/)
{
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
#define STEP1(T)                                                                    \
    do                                                                              \
    {                                                                               \
        const __m256 b = _mm256_load_ps(Bp + (k + (T)) * 8);                        \
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + (k + (T)) * 8 + 0), b, acc); \
    } while (0)
        STEP1(0);
        STEP1(1);
        STEP1(2);
        STEP1(3);
        STEP1(4);
        STEP1(5);
        STEP1(6);
        STEP1(7);
#undef STEP1
    }
    for (; k < Kblk; ++k)
    {
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc);
    }

    if (jb == 8)
    {
        _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), acc));
    }
    else
    {
        alignas(32) float buf[8];
        _mm256_store_ps(buf, acc);
        for (size_t t = 0; t < jb; ++t)
            c[t] += buf[t];
    }
}
#endif /* LINALG_SIMD_ENABLE */

/* ======================= Top-level GEMM ======================= */

int mul(float *RESTRICT C,
        const float *RESTRICT A,
        const float *RESTRICT B,
        uint16_t row_a, uint16_t column_a,
        uint16_t row_b, uint16_t column_b)
{
    if (column_a != row_b)
        return -EINVAL;

    const size_t M = row_a, K = column_a, N = column_b;

    /* Scalar fallback or small sizes */
    if (!linalg_has_avx2() || M == 0 || N == 0 || K == 0 ||
        (M < LINALG_SMALL_N_THRESH && N < LINALG_SMALL_N_THRESH))
    {
        for (size_t i = 0; i < M; ++i)
        {
            const float *ai = A + i * K;
            for (size_t j = 0; j < N; ++j)
            {
                const float *bj = B + j;
                float s = 0.f;
                for (size_t k = 0; k < K; ++k)
                    s += ai[k] * bj[k * N];
                C[i * N + j] = s;
            }
        }
        return 0;
    }

#if LINALG_SIMD_ENABLE
    const size_t Kc = (size_t)LINALG_BLOCK_KC; /* ~192–256 */
    const size_t Nc = (size_t)LINALG_BLOCK_JC; /* outer-N tile (header) */
    const size_t Mc = (size_t)LINALG_BLOCK_MC; /* outer-M tile */

    /* For each N tile */
    for (size_t j0 = 0; j0 < N; j0 += Nc)
    {
        const size_t jb_tile = (j0 + Nc <= N) ? Nc : (N - j0);
        const size_t n_panels_tile = (jb_tile + 7) / 8;

        /* Bp buffer sized for max Kc slab */
        size_t Bp_elems = Kc * n_panels_tile * 8;
        float *Bp = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, Bp_elems * sizeof(float));
        if (!Bp)
            return -ENOMEM;

        /* Iterate K slabs; repack B each slab for current j-tile */
        for (size_t kk = 0; kk < K; kk += Kc)
        {
            const size_t Kblk = (kk + Kc <= K) ? Kc : (K - kk);

            /* Pack B (kk..kk+Kblk, j0..j0+jb_tile) into 8-col panels */
            pack_B_8col_tile(Bp, B, K, N, kk, Kblk, j0, jb_tile);

            /* A pack buffer for up to 8 rows x Kblk */
            size_t Ap_elems = Kblk * 8;
            float *Ap = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, Ap_elems * sizeof(float));
            if (!Ap)
            {
                linalg_aligned_free(Bp);
                return -ENOMEM;
            }

            /* For each M tile */
            for (size_t i0 = 0; i0 < M; i0 += Mc)
            {
                const size_t ib_tile = (i0 + Mc <= M) ? Mc : (M - i0);

                /* Process rows in chunks of 8, then 4, then 1 */
                size_t i = i0;
                for (; i + 7 < i0 + ib_tile; i += 8)
                {
                    /* Pack A(8 x Kblk) */
                    pack_A_8row_tile(Ap, A, M, K, i, 8, kk, Kblk);

                    /* Walk panels inside this j-tile */
                    size_t panel_off = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 8, panel_off += Kblk * 8)
                    {
                        const size_t jb = (j + 8 <= j0 + jb_tile) ? 8 : (j0 + jb_tile - j);

                        /* On first K-slab, zero the C subtile so we can always += */
                        if (kk == 0)
                        {
                            for (size_t r = 0; r < 8; ++r)
                            {
                                float *cr = C + (i + r) * N + j;
                                for (size_t t = 0; t < jb; ++t)
                                    cr[t] = 0.0f;
                            }
                        }

                        gemm_8x8_panel_avx2fma_add(
                            /* c   */ C + i * N + j, /* ldc = N */
                            /* ldc */ N,
                            /* Ap  */ Ap,
                            /* Bp  */ Bp + panel_off,
                            /* K   */ Kblk,
                            /* jb  */ jb);
                    }
                }

                for (; i + 3 < i0 + ib_tile; i += 4)
                {
                    pack_A_8row_tile(Ap, A, M, K, i, 4, kk, Kblk);

                    size_t panel_off = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 8, panel_off += Kblk * 8)
                    {
                        const size_t jb = (j + 8 <= j0 + jb_tile) ? 8 : (j0 + jb_tile - j);

                        if (kk == 0)
                        {
                            for (size_t r = 0; r < 4; ++r)
                            {
                                float *cr = C + (i + r) * N + j;
                                for (size_t t = 0; t < jb; ++t)
                                    cr[t] = 0.0f;
                            }
                        }

                        gemm_4x8_panel_avx2fma_add(
                            C + i * N + j, N,
                            Ap,
                            Bp + panel_off,
                            Kblk, jb);
                    }
                }

                for (; i < i0 + ib_tile; ++i)
                {
                    pack_A_8row_tile(Ap, A, M, K, i, 1, kk, Kblk);

                    size_t panel_off = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 8, panel_off += Kblk * 8)
                    {
                        const size_t jb = (j + 8 <= j0 + jb_tile) ? 8 : (j0 + jb_tile - j);

                        if (kk == 0)
                        {
                            float *cr = C + i * N + j;
                            for (size_t t = 0; t < jb; ++t)
                                cr[t] = 0.0f;
                        }

                        gemm_1x8_panel_avx2fma_add(
                            C + i * N + j,
                            Ap,
                            Bp + panel_off,
                            Kblk, jb);
                    }
                }
            } /* i0 over Mc */

            linalg_aligned_free(Ap);
        } /* kk over Kc */

        linalg_aligned_free(Bp);
    } /* j0 over Nc */

    return 0;
#else
    (void)C;
    (void)A;
    (void)B;
    (void)row_a;
    (void)column_a;
    (void)row_b;
    (void)column_b;
    return -ENOTSUP;
#endif
}
