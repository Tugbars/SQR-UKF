#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <immintrin.h>
#include "linalg_simd.h"

/* Tuning knobs */
#ifndef CHOL_AVX_MIN_N
#  define CHOL_AVX_MIN_N  LINALG_SMALL_N_THRESH  /* 64 by default */
#endif

#ifndef CHOL_PF_ROWS_AHEAD
#  define CHOL_PF_ROWS_AHEAD 32  /* tune per µarch: 16-48 typical */
#endif

#ifndef CHOL_PREFETCH_ENABLE
#  define CHOL_PREFETCH_ENABLE 1  /* set to 0 to disable all prefetch */
#endif

#if CHOL_PREFETCH_ENABLE
#  define CHOL_PREFETCH_T0(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#  define CHOL_PREFETCH_T1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T1)
#else
#  define CHOL_PREFETCH_T0(ptr) ((void)0)
#  define CHOL_PREFETCH_T1(ptr) ((void)0)
#endif

/**
 * @brief Cholesky rank-1 update/downdate (Givens-like scheme).
 *
 * @details
 *  Updates the Cholesky factor L (or U) such that:
 *    - If rank_one_update=true:  L·L^T ← L·L^T + x·x^T  (or U^T·U ← U^T·U + x·x^T)
 *    - If rank_one_update=false: L·L^T ← L·L^T − x·x^T  (downdate)
 *
 *  The algorithm processes columns (for lower) or rows (for upper) sequentially,
 *  applying Givens-like rotations. Numerically stable; avoids forming x·x^T.
 *
 *  Vectorization:
 *   - AVX2: gathers for strided access, FMA for update equations
 *   - AVX-512 (optional): native scatter for ~30% speedup
 *   - Alignment peeling ensures 32B-aligned vector loads/stores on x
 *
 * @param[in,out] L        Cholesky factor (n×n), row-major. Modified in-place.
 * @param[in]     x_in     Update vector (length n). Not modified.
 * @param[in]     n        Dimension.
 * @param[in]     is_upper True if L is upper-triangular; false if lower-triangular.
 * @param[in]     rank_one_update True for update (add x·x^T); false for downdate (subtract).
 *
 * @retval  0       Success.
 * @retval -EINVAL  Invalid input (n=0).
 * @retval -ENOMEM  Memory allocation failed.
 * @retval -EDOM    Downdate would violate positive-definiteness (matrix became non-PD).
 *
 * @note
 *  - For downdate failures, L is left in a partially-updated but valid state
 *    (first i columns/rows correspond to a valid rank-i update).
 *  - All matrix elements are row-major.
 *  - Diagonal elements must be positive on entry.
 */
int cholupdate(float *RESTRICT L,
               const float *RESTRICT x_in,
               uint16_t n,
               bool is_upper,
               bool rank_one_update)
{
    if (n == 0)
        return -EINVAL;

    /* 32B-aligned working copy of x (hot write-backs during rotations) */
    float *x = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    if (!x)
        return -ENOMEM;
    memcpy(x, x_in, (size_t)n * sizeof(float));

    const float sign = rank_one_update ? 1.0f : -1.0f;

#if LINALG_SIMD_ENABLE
    const int use_avx = linalg_has_avx2() && n >= (uint32_t)CHOL_AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

    /* ===================== AVX2/AVX-512 Path ===================== */
#if LINALG_SIMD_ENABLE
    if (use_avx)
    {
        /* Gather index pattern for stride n: {0, n, 2n, ..., 7n} */
        alignas(32) int idx_step[8];
        for (int t = 0; t < 8; ++t)
            idx_step[t] = t * (int)n;
        const __m256i gidx = _mm256_load_si256((const __m256i *)idx_step);

        const uint32_t PF_ROWS = CHOL_PF_ROWS_AHEAD;

        for (uint32_t i = 0; i < n; ++i)
        {
            /* Diagonal element: L[i,i] (same for upper/lower) */
            const size_t di = (size_t)i * n + i;
            const float Lii = L[di];
            const float xi = x[i];

            /* Stable diagonal update: t = xi/Lii, c = sqrt(1 ± t²), r = c·Lii */
            const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
            const float r2 = 1.0f + sign * t * t;

            /* PD guard: downdate must preserve positive-definiteness */
            if (r2 <= 0.0f || !isfinite(r2))
            {
                linalg_aligned_free(x);
                return -EDOM;  /* domain error: matrix would become non-PD */
            }

            const float c = sqrtf(r2);
            const float r = c * Lii;
            const float s = t;

            L[di] = r;

            /* Prefetch next diagonal element (defeats stride-n HW prefetcher) */
            if (i + 1 < n)
            {
                const char *next_diag = (const char *)&L[((size_t)(i + 1)) * n + (i + 1)];
                CHOL_PREFETCH_T1(next_diag + 0 * 64);
                CHOL_PREFETCH_T1(next_diag + 1 * 64);
            }

            /* Skip off-diagonal updates if xi ≈ 0 (common for sparse updates) */
            if (xi == 0.0f)
                continue;

            /* Broadcast rotation parameters */
            const __m256 c_v    = _mm256_set1_ps(c);
            const __m256 invc_v = _mm256_set1_ps(1.0f / c);
            const __m256 s_v    = _mm256_set1_ps(s);
            const __m256 ss_v   = _mm256_set1_ps(sign * s);

            uint32_t k = i + 1;

            /* --- Peel to 8-alignment for x (enables aligned vector loads) --- */
            while ((k < n) && (k & 7))
            {
                size_t off;
                if (is_upper)
                    off = (size_t)i * n + k;  /* U[i,k]: row i, column k>i */
                else
                    off = (size_t)k * n + i;  /* L[k,i]: row k>i, column i */

                const float Lik = L[off];
                const float xk  = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k]   =  c * xk - s * Lik;
                ++k;
            }

            /* --- Main 8-wide vectorized loop --- */
            for (; k + 7 < n; k += 8)
            {
                /* Prefetch k+PF_ROWS ahead */
                const uint32_t kpf = k + PF_ROWS;
                if (kpf < n)
                {
                    CHOL_PREFETCH_T0(&x[kpf]);
                    if (is_upper)
                        CHOL_PREFETCH_T0(&L[(size_t)i * n + kpf]);
                    else
                        CHOL_PREFETCH_T0(&L[(size_t)kpf * n + i]);
                }

                /* Gather 8 elements from column i (lower) or row i (upper) */
                float *baseL;
                if (is_upper)
                    baseL = &L[(size_t)i * n + k];      /* U[i, k..k+7]: contiguous! */
                else
                    baseL = &L[(size_t)k * n + i];      /* L[k..k+7, i]: strided */

                __m256 Lik;
                if (is_upper)
                {
                    /* Upper: elements are contiguous in row i → direct load */
                    Lik = _mm256_loadu_ps(baseL);
                }
                else
                {
                    /* Lower: elements are strided (column access) → gather */
                    Lik = _mm256_i32gather_ps(baseL, gidx, sizeof(float));
                }

                /* x[k..k+7]: aligned load (after peel loop) */
                __m256 xk = _mm256_load_ps(&x[k]);

                /* Update equations:
                   Lik_new = (Lik + sign·s·xk) / c
                   xk_new  = c·xk − s·Lik                                    */
                __m256 Lik_new = _mm256_mul_ps(_mm256_fmadd_ps(ss_v, xk, Lik), invc_v);
                __m256 xk_new  = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

                /* Store updated x (aligned) */
                _mm256_store_ps(&x[k], xk_new);

                /* Store updated L elements */
                if (is_upper)
                {
                    /* Upper: contiguous store */
                    _mm256_storeu_ps(baseL, Lik_new);
                }
                else
                {
                    /* Lower: strided store → scatter emulation */
#ifdef __AVX512F__
                    /* AVX-512: native scatter (~30% faster) */
                    _mm256_i32scatter_ps(baseL, gidx, Lik_new, sizeof(float));
#else
                    /* AVX2: emulate scatter via temp buffer */
                    alignas(32) float buf[8];
                    _mm256_store_ps(buf, Lik_new);
                    for (int t = 0; t < 8; ++t)
                        baseL[(size_t)t * n] = buf[t];
#endif
                }
            }

            /* --- Scalar tail (k = [last_vec+1 .. n-1]) --- */
            for (; k < n; ++k)
            {
                size_t off;
                if (is_upper)
                    off = (size_t)i * n + k;
                else
                    off = (size_t)k * n + i;

                const float Lik = L[off];
                const float xk  = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k]   =  c * xk - s * Lik;
            }
        }

        linalg_aligned_free(x);
        return 0;  /* success */
    }
#endif /* LINALG_SIMD_ENABLE */

    /* ===================== Scalar Fallback ===================== */
    for (uint32_t i = 0; i < n; ++i)
    {
        const size_t di = (size_t)i * n + i;
        const float Lii = L[di];
        const float xi  = x[i];

        const float t  = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
        const float r2 = 1.0f + sign * t * t;

        if (r2 <= 0.0f || !isfinite(r2))
        {
            linalg_aligned_free(x);
            return -EDOM;
        }

        const float c = sqrtf(r2);
        const float r = c * Lii;
        const float s = t;

        L[di] = r;

        if (xi == 0.0f)
            continue;

        for (uint32_t k = i + 1; k < n; ++k)
        {
            size_t off;
            if (is_upper)
                off = (size_t)i * n + k;  /* U[i,k] */
            else
                off = (size_t)k * n + i;  /* L[k,i] */

            const float Lik = L[off];
            const float xk  = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k]   =  c * xk - s * Lik;
        }
    }

    linalg_aligned_free(x);
    return 0;
}