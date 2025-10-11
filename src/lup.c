// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "linalg_simd.h" 

/* ---------- corrected AVX2 rank-1 update ---------- */
static inline void rank1_update_avx2(float *RESTRICT LU, uint16_t n, uint16_t i,
                                     const uint8_t *RESTRICT P)
{
    const uint16_t pi      = P[i];                 /* pivot row index */
    const size_t   pivBase = (size_t)n * pi;       /* base offset */
    const float    piv     = LU[pivBase + i];
    const uint16_t k0      = i + 1;                /* first column to update */

    /* 1. multipliers  L(j,i) = LU[P[j],i] / piv */
    for (uint16_t j = i + 1; j < n; ++j) {
        const size_t rBase = (size_t)n * P[j];
        LU[rBase + i] /= piv;
    }

    /* 2. trailing update  U(j,k) -= L(j,i)*U(i,k)   (vectorised) */
    for (uint16_t j = i + 1; j < n; ++j) {
        const size_t rBase = (size_t)n * P[j];
        const float  lij   = LU[rBase + i];
        const __m256 lVec  = _mm256_set1_ps(lij);

        uint16_t k = k0;
        for (; k + 7 < n; k += 8) {                  /* full 8-wide */
            const __m256 uVec = _mm256_loadu_ps(&LU[pivBase + k]);
            __m256       vVec = _mm256_loadu_ps(&LU[rBase + k]);
            vVec = _mm256_fnmadd_ps(lVec, uVec, vVec);
            _mm256_storeu_ps(&LU[rBase + k], vVec);
        }
        for (; k < n; ++k)                           /* scalar tail */
            LU[rBase + k] -= lij * LU[pivBase + k];
    }
}

/* ---------- scalar fallback (correct indexing) ---------- */
static void rank1_update_scalar(float *RESTRICT LU, uint16_t n, uint16_t i,
                                const uint8_t *RESTRICT P)
{
    const uint16_t pi      = P[i];
    const size_t   pivBase = (size_t)n * pi;
    const float    piv     = LU[pivBase + i];

    for (uint16_t j = i + 1; j < n; ++j) {
        const size_t rBase = (size_t)n * P[j];
        LU[rBase + i] /= piv;
        const float lij = LU[rBase + i];
        for (uint16_t k = i + 1; k < n; ++k)
            LU[rBase + k] -= lij * LU[pivBase + k];
    }
}

/* ---------- public API (unchanged) ---------- */
int lup(const float *RESTRICT A, float *RESTRICT LU, uint8_t *P, uint16_t n)
{
    if (n == 0) return -EINVAL;
    if (A != LU) memcpy(LU, A, (size_t)n * n * sizeof(float));
    for (uint16_t i = 0; i < n; ++i) P[i] = i;

    const int use_avx2 =
#if LINALG_SIMD_ENABLE
        linalg_has_avx2();
#else
        0;
#endif

    for (uint16_t i = 0; i < n - 1; ++i) {
        uint16_t imax = i;
        for (uint16_t j = i + 1; j < n; ++j)
            if (fabsf(LU[(size_t)n * P[j] + i]) > fabsf(LU[(size_t)n * P[imax] + i]))
                imax = j;

        uint8_t tmp = P[i]; P[i] = P[imax]; P[imax] = tmp;

        float d = LU[(size_t)n * P[i] + i];
        float tol = n * FLT_EPSILON * fabsf(d);
        if (fabsf(d) <= tol) return -ENOTSUP;

#if LINALG_SIMD_ENABLE
        if (use_avx2) rank1_update_avx2(LU, n, i, P);
        else
#endif
        rank1_update_scalar(LU, n, i, P);
    }
    return 0;
}
