// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "linalg_simd.h"

/* Use the library's small-n threshold for the AVX crossover. */
#ifndef AVX_MIN_N
#  define AVX_MIN_N  LINALG_SMALL_N_THRESH
#endif

/*
 * cholupdate (no transpose):
 *  - L is row-major lower-triangular (n x n), with positive diagonal.
 *  - xx is length n.
 *  - rank_one_update = true → rank-one update (A + x x^T); false → downdate (A - x x^T).
 * The routine updates L in place so that L corresponds to the Cholesky factor of the
 * updated/downdated matrix.
 *
 * Notes:
 *  - Column-by-column (i = 0..n-1) Givens-style updates.
 *  - AVX2 path: gathers for strided column elements, contiguous loads/stores for x.
 *  - Unrolled k-loop (2×8) + prefetch to better hide gather latency.
 */
void cholupdate(float *RESTRICT L, const float *RESTRICT x_in,
                uint16_t row16, bool rank_one_update)
{
    if (row16 == 0)
        return;
    const uint32_t n = (uint32_t)row16;

    /* 32B-aligned working copy of x (hot write-backs). */
    float *x = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    if (!x)
        return;
    memcpy(x, x_in, (size_t)n * sizeof(float));

    const float sign = rank_one_update ? 1.0f : -1.0f;

#if LINALG_SIMD_ENABLE
    const int use_avx = linalg_has_avx2() && n >= (uint32_t)AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

#if LINALG_SIMD_ENABLE
    if (use_avx)
    {
        /* gather index pattern for stride n: {0, n, 2n, ..., 7n} */
        alignas(32) int idx_step[8];
        for (int t = 0; t < 8; ++t)
            idx_step[t] = t * (int)n;
        const __m256i gidx = _mm256_load_si256((const __m256i *)idx_step);

        const uint32_t PF_ROWS = 32; /* tune per µarch */

        for (uint32_t i = 0; i < n; ++i)
        {
            const size_t di = (size_t)i * n + i;
            const float Lii = L[di];
            const float xi = x[i];

            /* Cheaper diagonal math (1 divide): t=xi/Lii; c=sqrt(1+sign*t^2); r=c*Lii */
            const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
            const float r2 = 1.0f + sign * t * t;

            /* PD guard for downdate (and safety in general) */
            if (r2 <= 0.0f)
            {
                linalg_aligned_free(x);
                return; /* early exit; L is valid up to previous columns */
            }

            const float c = sqrtf(r2);
            const float r = c * Lii;
            const float s = t;

            L[di] = r;

            /* Warm the first touches of column i+1 (row-major stride-n defeats HW prefetch). */
            if (i + 1 < n)
            {
                const char *next_col = (const char *)&L[((size_t)(i + 1)) * n + (i + 1)];
                _mm_prefetch(next_col + 0 * 64, _MM_HINT_T1);
                _mm_prefetch(next_col + 1 * 64, _MM_HINT_T1);
            }

            if (xi == 0.0f)
                continue; /* predictable skip */

            const __m256 c_v = _mm256_set1_ps(c);
            const __m256 invc_v = _mm256_set1_ps(1.0f / c);
            const __m256 s_v = _mm256_set1_ps(s);
            const __m256 ss_v = _mm256_set1_ps(sign * s);

            uint32_t k = i + 1;

            /* --- peel until k is 8-aligned (so &x[k] is 32B aligned) --- */
            while ((k < n) && (k & 7)) {
                const size_t off = (size_t)k * n + i;
                const float  Lik = L[off];
                const float  xk  = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k]   =  c * xk - s * Lik;
                ++k;
            }

            /* Single 8-lane AVX block; scalar tail. */
            for (; k + 7 < n; k += 8)
            {
                const uint32_t kpf = k + PF_ROWS;
                if (kpf < n)
                {
                    _mm_prefetch((const char *)(&x[kpf]), _MM_HINT_T0);
                    const float *pfL = &L[(size_t)kpf * n + i];
                    _mm_prefetch((const char *)(pfL), _MM_HINT_T0);
                }

                float *baseL = &L[(size_t)k * n + i];
                __m256 Lik = _mm256_i32gather_ps(baseL, gidx, sizeof(float));
                 /* x is 32B-aligned and k % 8 == 0 after the peel → aligned load/store */
                __m256 xk = _mm256_load_ps(&x[k]); 

                /* Lik_new = (Lik + sign*s*xk) / c */
                __m256 Lik_new = _mm256_mul_ps(_mm256_fmadd_ps(ss_v, xk, Lik), invc_v);
                /* xk_new  =  c*xk - s*Lik  =>  -(s*Lik) + c*xk */
                __m256 xk_new = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

                _mm256_store_ps(&x[k], xk_new);

                /* Scatter via small scalar loop sourced from a vector store. */
                alignas(32) float buf[8];
                _mm256_store_ps(buf, Lik_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = buf[t];
            }

            for (; k < n; ++k)
            {
                const size_t off = (size_t)k * n + i;
                const float Lik = L[off];
                const float xk = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k] = c * xk - s * Lik;
            }
        }

        linalg_aligned_free(x);
        return;
    }
#endif /* LINALG_SIMD_ENABLE */

    /* -------- scalar fallback -------- */
    for (uint32_t i = 0; i < n; ++i)
    {
        const size_t di = (size_t)i * n + i;
        const float Lii = L[di];
        const float xi = x[i];

        const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
        const float r2 = 1.0f + sign * t * t;
        if (r2 <= 0.0f)
        {
            linalg_aligned_free(x);
            return;
        }
        const float c = sqrtf(r2);
        const float r = c * Lii;
        const float s = t;

        L[di] = r;

        if (xi == 0.0f)
            continue;

        for (uint32_t k = i + 1; k < n; ++k)
        {
            const size_t off = (size_t)k * n + i;
            const float Lik = L[off];
            const float xk = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k] = c * xk - s * Lik;
        }
    }

    linalg_aligned_free(x);
}