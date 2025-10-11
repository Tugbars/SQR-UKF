// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "linalg_simd.h" 

#ifndef RESTRICT
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#    define RESTRICT restrict
#  else
#    define RESTRICT
#  endif
#endif

static inline int has_avx2(void) {
#if defined(__AVX2__) && defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

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
void cholupdate(float *RESTRICT L, const float *RESTRICT xx, uint16_t n, bool rank_one_update)
{
    if (n == 0) return;

    /* working copy of x (modified in-place) */
    float *x = (float*)malloc((size_t)n * sizeof(float));
    if (!x) return;
    memcpy(x, xx, (size_t)n * sizeof(float));

    const float sign = rank_one_update ? 1.0f : -1.0f;

    if (has_avx2() && n >= 8) {
        /* gather index pattern for stride = n: {0, n, 2n, ..., 7n} */
        alignas(32) int idx_step[8];
        for (int t = 0; t < 8; ++t) idx_step[t] = t * (int)n;
        const __m256i gidx = _mm256_load_si256((const __m256i*)idx_step);

        /* prefetch distance (in rows) for the strided column + x-vector */
        const int PF_ROWS = 32;  /* tuneable: 16..64 sensible */

        for (uint16_t i = 0; i < n; ++i) {
            const size_t di = (size_t)i * n + i;
            const float  Lii = L[di];
            const float  xi  = x[i];

            /* Compute rotation (update/downdate) */
            const float r2 = Lii*Lii + sign * xi*xi;
            const float r  = sqrtf(r2);
            const float c  = r / Lii;
            const float s  = (Lii != 0.0f) ? (xi / Lii) : 0.0f;

            /* Update diagonal */
            L[di] = r;

            /* If xi==0, column i below diagonal and x[k] are unchanged except for the next columns;
               we can early-out the heavy work for this column. */
            if (xi == 0.0f) {
                continue;
            }

            /* Broadcast constants */
            const __m256 c_v     = _mm256_set1_ps(c);
            const __m256 invc_v  = _mm256_set1_ps(1.0f / c);
            const __m256 s_v     = _mm256_set1_ps(s);
            const __m256 ss_v    = _mm256_set1_ps(sign * s);

            /* Update column i below the diagonal: k = i+1..n-1
               Unroll by 16 rows = 2×(8-row) AVX blocks to overlap gathers. */
            uint16_t k = (uint16_t)(i + 1);

            for (; (uint16_t)(k + 15) < n; k = (uint16_t)(k + 16)) {
                /* Prefetch a bit ahead on x and column-L memory to hide latency */
                const uint16_t kpf = (uint16_t)(k + PF_ROWS);
                if (kpf < n) {
                    _mm_prefetch((const char*)(&x[kpf]), _MM_HINT_T0);
                    /* prefetch strided column: take one base and hope HW prefetch helps */
                    const float *pfL = &L[(size_t)kpf * n + i];
                    _mm_prefetch((const char*)(pfL), _MM_HINT_T0);
                }

                /* ----- Block 0: rows k..k+7 ----- */
                float *baseL0 = &L[(size_t)k * n + i];
                __m256 Lik0 = _mm256_i32gather_ps(baseL0, gidx, sizeof(float));
                __m256 xk0  = _mm256_loadu_ps(&x[k]);

                __m256 Lik0_new = _mm256_mul_ps(_mm256_add_ps(Lik0, _mm256_mul_ps(ss_v, xk0)), invc_v);
                __m256 xk0_new  = _mm256_fnmadd_ps(s_v, Lik0, _mm256_mul_ps(c_v, xk0));

                /* store x[k..k+7] */
                _mm256_storeu_ps(&x[k], xk0_new);

                /* scatter back Lik0_new via scalar stores */
                alignas(32) float buf0[8];
                _mm256_store_ps(buf0, Lik0_new);
                for (int t = 0; t < 8; ++t)
                    baseL0[(size_t)t * n] = buf0[t];

                /* ----- Block 1: rows k+8..k+15 ----- */
                float *baseL1 = &L[(size_t)(k + 8) * n + i];
                __m256 Lik1 = _mm256_i32gather_ps(baseL1, gidx, sizeof(float));
                __m256 xk1  = _mm256_loadu_ps(&x[k + 8]);

                __m256 Lik1_new = _mm256_mul_ps(_mm256_add_ps(Lik1, _mm256_mul_ps(ss_v, xk1)), invc_v);
                __m256 xk1_new  = _mm256_fnmadd_ps(s_v, Lik1, _mm256_mul_ps(c_v, xk1));

                _mm256_storeu_ps(&x[k + 8], xk1_new);

                alignas(32) float buf1[8];
                _mm256_store_ps(buf1, Lik1_new);
                for (int t = 0; t < 8; ++t)
                    baseL1[(size_t)t * n] = buf1[t];
            }

            /* 8-row AVX remainder */
            for (; (uint16_t)(k + 7) < n; k = (uint16_t)(k + 8)) {
                /* Optional prefetch */
                const uint16_t kpf = (uint16_t)(k + PF_ROWS);
                if (kpf < n) {
                    _mm_prefetch((const char*)(&x[kpf]), _MM_HINT_T0);
                    const float *pfL = &L[(size_t)kpf * n + i];
                    _mm_prefetch((const char*)(pfL), _MM_HINT_T0);
                }

                float *baseL = &L[(size_t)k * n + i];
                __m256 Lik = _mm256_i32gather_ps(baseL, gidx, sizeof(float));
                __m256 xk  = _mm256_loadu_ps(&x[k]);

                __m256 Lik_new = _mm256_mul_ps(_mm256_add_ps(Lik, _mm256_mul_ps(ss_v, xk)), invc_v);
                __m256 xk_new  = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

                _mm256_storeu_ps(&x[k], xk_new);

                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, Lik_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }

            /* scalar tail */
            for (; k < n; ++k) {
                const size_t off = (size_t)k * n + i;
                const float  Lik = L[off];
                const float  xk  = x[k];
                const float  Lik_new = (Lik + sign * s * xk) / c;
                const float  xk_new  =  c * xk - s * Lik;
                L[off] = Lik_new;
                x[k]   = xk_new;
            }
        }
    } else {
        /* scalar fallback */
        for (uint16_t i = 0; i < n; ++i) {
            const size_t di = (size_t)i * n + i;
            const float  Lii = L[di];
            const float  xi  = x[i];

            const float r2 = Lii*Lii + sign * xi*xi;
            const float r  = sqrtf(r2);
            const float c  = r / Lii;
            const float s  = (Lii != 0.0f) ? (xi / Lii) : 0.0f;

            L[di] = r;

            if (xi == 0.0f) continue;

            for (uint16_t k = (uint16_t)(i + 1); k < n; ++k) {
                const size_t off = (size_t)k * n + i;
                const float  Lik = L[off];
                const float  xk  = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k]   =  c * xk - s * Lik;
            }
        }
    }

    free(x);
}

/*
Prefetching:

The single prefetch for L relies on hardware prefetching, which may miss strided accesses. Prefetching could be extended to the scalar path.
*/
