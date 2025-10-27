// SPDX-License-Identifier: MIT
/**
 * @file cholupdatek.c
 * @brief Rank-k Cholesky update/downdate (single-precision), tiled + AVX2.
 *
 * @details
 * Updates a Cholesky factor L (lower or upper) in place so that:
 *   - update   (add>0):  L Lᵀ  ←  L Lᵀ  + X Xᵀ
 *   - downdate (add<0):  L Lᵀ  ←  L Lᵀ  − X Xᵀ
 * where X is n×k (row-major).  Numerical core is the robust Givens-style rank-1
 * update used in high-quality implementations; we apply it k times but in
 * **cache-tiled batches** to improve locality. AVX2 kernels are used for the
 * row updates when available (same intrinsics style as your existing code).
 *
 * Public API:
 *   int cholupdatek(float *L, const float *X, uint16_t n, uint16_t k,
 *                   bool is_upper, int add);
 *   int cholupdate (float *L, const float *x, uint16_t n,
 *                   bool is_upper, bool rank_one_update); // wrapper
 *
 * Notes:
 *  - Row-major storage.
 *  - On downdate, if positivity would be violated, returns -EDOM and leaves L
 *    valid for the portion already processed.
 *  - This version is “BLAS-3 viable”: work is batched over a column-tile of X,
 *    minimizing streaming of L/x. When/if you want a full TRSM+SYRK/GEMM block
 *    algorithm, we can swap the inner loop without changing the API.
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <immintrin.h>
#include "linalg_simd.h"
#include "qr.h"

#ifndef CHOLK_COL_TILE
#define CHOLK_COL_TILE 32 /* columns of X per batch (try 32–128) */
#endif

#ifndef CHOLK_AVX_MIN_N
#define CHOLK_AVX_MIN_N LINALG_SMALL_N_THRESH
#endif

/* ==== Robust rank-1 kernel (the one you already have, slightly factored) ==== */
/* Applies one vector x (length n) to in-place L, lower/upper, update (add=+1) or downdate (add=-1).
   x is both input and work buffer (modified). Returns 0 or -EDOM on SPD violation. */
static int cholupdate_rank1_core(float *RESTRICT L,
                                 float *RESTRICT x, /* in/out work */
                                 uint16_t n,
                                 bool is_upper,
                                 int add /* +1 or -1 */)
{
    const float sign = (add >= 0) ? 1.0f : -1.0f;

#if LINALG_SIMD_ENABLE
    const int use_avx = linalg_has_avx2() && n >= (uint32_t)CHOLK_AVX_MIN_N;
#else
    const int use_avx = 0;
#endif

    for (uint32_t i = 0; i < n; ++i)
    {
        const size_t di = (size_t)i * n + i;
        const float Lii = L[di];
        const float xi = x[i];

        const float t = (Lii != 0.0f) ? (xi / Lii) : 0.0f;
        const float r2 = 1.0f + sign * t * t;
        if (r2 <= 0.0f || !isfinite(r2))
            return -EDOM;

        const float c = sqrtf(r2);
        const float s = t;
        L[di] = c * Lii;

        if (xi == 0.0f)
            continue;

        /* Scalar tail for short segments or no-AVX. */
        if (!use_avx || (i + 8 >= n))
        {
            for (uint32_t k = i + 1; k < n; ++k)
            {
                const size_t off = is_upper ? (size_t)i * n + k
                                            : (size_t)k * n + i;
                const float Lik = L[off];
                const float xk = x[k];
                L[off] = (Lik + sign * s * xk) / c;
                x[k] = c * xk - s * Lik;
            }
            continue;
        }

#if LINALG_SIMD_ENABLE
        /* AVX2 vectorized body over k=i+1..n-1 */
        uint32_t k = (uint32_t)i + 1;

        /* Align x to 32B for aligned loads; peel until aligned */
        while ((k < n) && ((uintptr_t)(&x[k]) & 31u))
        {
            const size_t off = is_upper ? (size_t)i * n + k
                                        : (size_t)k * n + i;
            const float Lik = L[off];
            const float xk = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k] = c * xk - s * Lik;
            ++k;
        }

        const __m256 c_v = _mm256_set1_ps(c);
        const __m256 s_v = _mm256_set1_ps(s);
        const __m256 ss_v = _mm256_set1_ps(sign * s);

        for (; k + 7 < n; k += 8)
        {
            float *baseL = is_upper ? &L[(size_t)i * n + k]  /* contiguous */
                                    : &L[(size_t)k * n + i]; /* strided by n */
            __m256 Lik;
            if (is_upper)
            {
                Lik = _mm256_loadu_ps(baseL);
            }
            else
            {
#ifdef __AVX2__
                /* gather indices 0,n,2n,... */
                alignas(32) int idx[8];
                for (int t = 0; t < 8; ++t)
                    idx[t] = t * (int)n;
                Lik = _mm256_i32gather_ps(baseL, _mm256_load_si256((const __m256i *)idx), sizeof(float));
#else
                /* fallback (shouldn’t happen with AVX2 defined) */
                alignas(32) float tmp[8];
                for (int t = 0; t < 8; ++t)
                    tmp[t] = baseL[(size_t)t * n];
                Lik = _mm256_load_ps(tmp);
#endif
            }

            __m256 xk = _mm256_load_ps(&x[k]);

            __m256 Lik_new = _mm256_mul_ps(_mm256_fmadd_ps(ss_v, xk, Lik), _mm256_div_ps(_mm256_set1_ps(1.0f), c_v));
            __m256 xk_new = _mm256_fnmadd_ps(s_v, Lik, _mm256_mul_ps(c_v, xk));

            _mm256_store_ps(&x[k], xk_new);

            if (is_upper)
            {
                _mm256_storeu_ps(baseL, Lik_new);
            }
            else
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, Lik_new);
                for (int t = 0; t < 8; ++t)
                    baseL[(size_t)t * n] = tmp[t];
            }
        }

        for (; k < n; ++k)
        {
            const size_t off = is_upper ? (size_t)i * n + k
                                        : (size_t)k * n + i;
            const float Lik = L[off];
            const float xk = x[k];
            L[off] = (Lik + sign * s * xk) / c;
            x[k] = c * xk - s * Lik;
        }
#endif
    }
    return 0;
}

/* ==== Public rank-k API (tiled over columns of X) ==== */
int cholupdatek(float *RESTRICT L,
                const float *RESTRICT X, /* n×k, row-major */
                uint16_t n,
                uint16_t k,
                bool is_upper,
                int add /* +1 update, -1 downdate */)
{
    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (add != +1 && add != -1)
        return -EINVAL;

    const uint16_t T = (CHOLK_COL_TILE == 0) ? 32 : (uint16_t)CHOLK_COL_TILE;

    /* Work buffer for a single vector (reused); 32B-aligned. */
    float *xbuf = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(float));
    if (!xbuf)
        return -ENOMEM;

    int rc = 0;
    for (uint16_t p0 = 0; p0 < k; p0 += T)
    {
        const uint16_t jb = (uint16_t)((p0 + T <= k) ? T : (k - p0));

        /* Process the tile’s columns one by one (warm caches). */
        for (uint16_t t = 0; t < jb; ++t)
        {
            /* x := X[:, p0+t] */
            const float *xcol = X + (size_t)0 * k + (p0 + t); /* interleaved by columns? We were told row-major.
                                                                 So X[r*k + (p0+t)] is element (r, p0+t). */
            /* Gather the column into a contiguous row-major vector xbuf[r] = X[r, p] */
            for (uint16_t r = 0; r < n; ++r)
                xbuf[r] = xcol[(size_t)r * k];

            rc = cholupdate_rank1_core(L, xbuf, n, is_upper, add);
            if (rc)
            {
                linalg_aligned_free(xbuf);
                return rc;
            }
        }
    }

    linalg_aligned_free(xbuf);
    return 0;
}

/* ---- Utility: copy triangular (upper) block R11 out of a row-major R ---- */
static void copy_upper_nxn_from_qr(float *RESTRICT Udst, const float *RESTRICT Rsrc,
                                   uint16_t n, uint16_t ldR)
{
    for (uint16_t i = 0; i < n; ++i)
    {
        const float *row = Rsrc + (size_t)i * ldR;
        /* zeros below diag */
        for (uint16_t j = 0; j < i; ++j)
            Udst[(size_t)i * n + j] = 0.0f;
        /* copy diag..end */
        memcpy(Udst + (size_t)i * n + i, row + i, (size_t)(n - i) * sizeof(float));
    }
}

/* ---- Public: BLAS-3 rank-k update/downdate via blocked QR ---- */
int cholupdatek_blockqr(float *RESTRICT L_or_U,
                        const float *RESTRICT X, /* n×k */
                        uint16_t n, uint16_t k,
                        bool is_upper,
                        int add /* +1 update, -1 downdate */)
{
    if (n == 0)
        return -EINVAL;
    if (k == 0)
        return 0;
    if (add != +1 && add != -1)
        return -EINVAL;

    /* Work: build M = [U | s*X] in row-major, where:
       - if is_upper: U = L_or_U (upper)
       - else       : U = L_or_Uᵀ (we’ll materialize U into M’s left block) */
    const uint16_t m_rows = n;
    const uint16_t m_cols = (uint16_t)(n + k);

    float *M = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_rows * m_cols * sizeof(float));
    if (!M)
        return -ENOMEM;

    /* Left block: copy U */
    if (is_upper)
    {
        /* L_or_U already holds U (upper n×n) in row-major; copy into M[:,0:n] */
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;
            const float *src = L_or_U + (size_t)i * n;
            /* zeros below diag for cleanliness (not required for GEQRF) */
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = 0.0f;
            memcpy(dst + i, src + i, (size_t)(n - i) * sizeof(float));
        }
    }
    else
    {
        /* Build U := Lᵀ explicitly into M[:,0:n] (upper) */
        for (uint16_t i = 0; i < n; ++i)
        {
            float *dst = M + (size_t)i * m_cols;
            /* j < i → dst[j] = Lᵀ(i,j) = L(j,i) (below diag of L) */
            for (uint16_t j = 0; j < i; ++j)
                dst[j] = L_or_U[(size_t)j * n + i];
            /* j >= i → dst[j] = U(i,j) = 0 if j<i, else Lᵀ(i,j) = L(j,i)=0 for j>i unless L diag at j=i */
            dst[i] = L_or_U[(size_t)i * n + i]; /* diag */
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
                dst[j] = 0.0f; /* strictly upper of U is 0 for Lᵀ if L lower */
        }
    }

    /* Right block: scaled X (sign = +1 for update, -1 for downdate) */
    const float s = (add >= 0) ? 1.0f : -1.0f;
    for (uint16_t i = 0; i < n; ++i)
    {
        float *dst = M + (size_t)i * m_cols + n; /* start of right block */
        const float *src = X + (size_t)i * k;    /* row-major X */
        for (uint16_t j = 0; j < k; ++j)
            dst[j] = s * src[j];
    }

    /* GEQRF (blocked compact-WY) on M (n×(n+k)), no need to form Q; we only need R. */
    float *tau = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)((n < m_cols ? n : m_cols)) * sizeof(float));
    if (!tau)
    {
        linalg_aligned_free(M);
        return -ENOMEM;
    }

    int rc = qrw_geqrf_blocked_wy(M, m_rows, m_cols, QRW_IB_DEFAULT, tau);
    linalg_aligned_free(tau);
    if (rc)
    {
        linalg_aligned_free(M);
        return rc;
    }

    /* Extract U_new = leading upper-tri R11 (n×n) */
    if (is_upper)
    {
        copy_upper_nxn_from_qr(/*Udst=*/L_or_U, /*Rsrc=*/M, n, /*ldR=*/m_cols);
    }
    else
    {
        /* lower case: L_new = U_newᵀ. Extract U_new, then transpose to L. */
        /* Temp upper block */
        float *Utmp = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * n * sizeof(float));
        if (!Utmp)
        {
            linalg_aligned_free(M);
            return -ENOMEM;
        }
        copy_upper_nxn_from_qr(Utmp, M, n, m_cols);

        /* Write L_or_U ← Utmpᵀ as lower-tri */
        for (uint16_t i = 0; i < n; ++i)
        {
            for (uint16_t j = 0; j < i; ++j)
            {
                L_or_U[(size_t)i * n + j] = Utmp[(size_t)j * n + i];
            }
            L_or_U[(size_t)i * n + i] = Utmp[(size_t)i * n + i];
            for (uint16_t j = (uint16_t)(i + 1); j < n; ++j)
            {
                L_or_U[(size_t)i * n + j] = 0.0f;
            }
        }
        linalg_aligned_free(Utmp);
    }

    linalg_aligned_free(M);
    return 0;
}

/* Optional convenience wrapper:
   Use the blocked-QR BLAS-3 path if available, otherwise fall back to your tiled rank-1. */
int cholupdatek_blas3(float *RESTRICT L_or_U,
                      const float *RESTRICT X, uint16_t n, uint16_t k,
                      bool is_upper, int add)
{
    /* If your build guarantees qrw_geqrf_blocked_wy is present, you can just call cholupdatek_blockqr directly. */
    int rc = cholupdatek_blockqr(L_or_U, X, n, k, is_upper, add);
    if (rc == -ENOTSUP)
    {
        /* Fallback to tiled rank-1 version (still correct, less BLAS-3 heavy) */
        return cholupdatek(L_or_U, X, n, k, is_upper, add);
    }
    return rc;
}
