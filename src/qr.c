// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <immintrin.h>

#include "linalg_simd.h" // RESTRICT, linalg_has_avx2(), LINALG_* knobs
// also exports mul(), inv() that qr_scalar uses

/**
 * @brief Scalar (reference) QR decomposition via Householder reflections.
 *
 * @details
 *  Computes the QR factorization of A (m×n) using classical Householder
 *  reflectors in single precision. Produces R in upper-triangular form and,
 *  unless @p only_R is true, forms Q explicitly by accumulating reflectors
 *  and then inverting the product (Q = H₀H₁…H_{l-1}). This path is intended
 *  for small matrices or as a portable fallback when SIMD is unavailable.
 *
 * @param[in]  A       Input matrix (m×n), row-major.
 * @param[out] Q       Output orthogonal matrix (m×m). Ignored if @p only_R=true.
 * @param[out] R       Output upper-triangular factor (m×n).
 * @param[in]  m       Number of rows of A.
 * @param[in]  n       Number of columns of A.
 * @param[in]  only_R  If true, compute only R (skip forming Q).
 *
 * @retval 0          Success.
 * @retval -ENOMEM    Allocation failure for temporaries.
 *
 * @warning Q and R must not alias A. All buffers must be valid and sized.
 * @note Uses mul() and inv() from linalg_simd.h for dense ops in the scalar path.
 */
static int qr_scalar(const float *RESTRICT A, float *RESTRICT Q,
                     float *RESTRICT R, uint16_t m, uint16_t n, bool only_R)
{
    const uint16_t l = (m - 1 < n) ? (m - 1) : n;

    memcpy(R, A, (size_t)m * n * sizeof(float));

    float *H = (float *)malloc((size_t)m * m * sizeof(float));
    float *W = (float *)malloc((size_t)m * sizeof(float));
    float *WW = (float *)malloc((size_t)m * m * sizeof(float));
    float *Hi = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiH = (float *)malloc((size_t)m * m * sizeof(float));
    float *HiR = (float *)malloc((size_t)m * n * sizeof(float));
    if (!H || !W || !WW || !Hi || !HiH || !HiR)
    {
        free(H);
        free(W);
        free(WW);
        free(Hi);
        free(HiH);
        free(HiR);
        return -ENOMEM;
    }

    memset(H, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i)
        H[(size_t)i * m + i] = 1.0f;

    for (uint16_t k = 0; k < l; ++k)
    {
        float s = 0.0f;
        for (uint16_t i = k; i < m; ++i)
        {
            float x = R[(size_t)i * n + k];
            s += x * x;
        }
        s = sqrtf(s);
        float Rk = R[(size_t)k * n + k];
        if (Rk < 0.0f)
            s = -s;
        float r = sqrtf(2.0f * s * (Rk + s));

        memset(W, 0, (size_t)m * sizeof(float));
        W[k] = (Rk + s) / r;
        for (uint16_t i = k + 1; i < m; ++i)
            W[i] = R[(size_t)i * n + k] / r;

        mul(WW, W, W, m, 1, 1, m); // WW = W * Wᵀ
        for (size_t i = 0; i < (size_t)m * m; ++i)
            Hi[i] = -2.0f * WW[i];
        for (uint16_t i = 0; i < m; ++i)
            Hi[(size_t)i * m + i] += 1.0f;

        if (!only_R)
        {
            mul(HiH, Hi, H, m, m, m, m);
            memcpy(H, HiH, (size_t)m * m * sizeof(float));
        }
        mul(HiR, Hi, R, m, m, m, n);
        memcpy(R, HiR, (size_t)m * n * sizeof(float));
    }

    if (!only_R)
    {
        // Avoid aliasing restrict parameters by using a temporary copy
        float *Hin = (float *)malloc((size_t)m * m * sizeof(float));
        if (!Hin)
        {
            free(H);
            free(W);
            free(WW);
            free(Hi);
            free(HiH);
            free(HiR);
            return -ENOMEM;
        }
        memcpy(Hin, H, (size_t)m * m * sizeof(float));

        int rc = inv(H, Hin, m); // separate input/output
        free(Hin);
        if (rc != 0)
        {
            free(H);
            free(W);
            free(WW);
            free(Hi);
            free(HiH);
            free(HiR);
            return -ENOTSUP;
        }

        memcpy(Q, H, (size_t)m * m * sizeof(float));
    }

    free(H);
    free(W);
    free(WW);
    free(Hi);
    free(HiH);
    free(HiR);
    return 0;
}

#if LINALG_SIMD_ENABLE
/* ---------------- AVX helpers (compiled only with AVX2/FMA) ---------------- */

/**
 * @brief Horizontal sum of 8 packed single-precision floats.
 *
 * @param[in] v   __m256 containing 8 lanes to sum.
 * @return Sum of all 8 lanes as a scalar float.
 *
 * @note Helper for AVX2 accumulation; not exported.
 */
static inline float hsum8_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/**
 * @brief Broadcast the selected 32-bit lane of a packed 8-float vector to all lanes.
 *
 * @param[in] pack  Source packed vector.
 * @param[in] lane  Lane index (0..7) to broadcast.
 * @return __m256 with all lanes equal to pack[lane].
 *
 * @note Helper for AVX2 lane manipulation; not exported.
 */
static inline __m256 bcast_lane(__m256 pack, int lane)
{
    const __m256i idx = _mm256_set1_epi32(lane);
    return _mm256_permutevar8x32_ps(pack, idx);
}

/**
 * @brief Robust Householder vector construction (AVX2-assisted).
 *
 * @details
 *  Builds a numerically stable Householder reflector for the vector x of length
 *  @p len using scaling by max |x| to avoid overflow/underflow. On return:
 *   - v[0] = 1, v[1:] = x[1:]/β
 *   - x[0] = -β (β has the sign chosen to reduce cancellation)
 *   - Returns τ = 2 / (1 + σ/β²) (the standard Householder scalar)
 *
 * @param[out] v     Output Householder vector (length @p len).
 * @param[in,out] x  On input: target vector segment; on output: x[0] = -β.
 * @param[in]  len   Length of the vector segment.
 *
 * @return τ (tau) scaling for the reflector; returns 0 when x is (near) zero.
 *
 * @note Uses AVX2 to compute max norm and squared norm; has scalar tails.
 * @warning v and x must reference valid buffers of length at least @p len.
 */
static float householder_vec_avx(float *RESTRICT v, float *RESTRICT x, uint16_t len)
{
    if (len == 0)
        return 0.0f;

    /* Find amax = max_i |x[i]| (vectorized head + scalar tail) */
    __m256 amax8 = _mm256_setzero_ps();
    uint16_t i = 0;
    for (; i + 7 < len; i += 8)
    {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), xv); /* fabsf */
        amax8 = _mm256_max_ps(amax8, ax);
    }
    float amax = 0.0f;
    {
        __m128 lo = _mm256_castps256_ps128(amax8);
        __m128 hi = _mm256_extractf128_ps(amax8, 1);
        __m128 m1 = _mm_max_ps(lo, hi);
        m1 = _mm_max_ps(m1, _mm_movehl_ps(m1, m1));
        m1 = _mm_max_ps(m1, _mm_shuffle_ps(m1, m1, 0x55));
        amax = _mm_cvtss_f32(m1);
    }
    for (; i < len; ++i)
    {
        float ax = fabsf(x[i]);
        if (ax > amax)
            amax = ax;
    }

    if (amax == 0.0f)
    {
        /* x is zero ⇒ no reflection */
        v[0] = 1.0f;
        for (uint16_t t = 1; t < len; ++t)
            v[t] = 0.0f;
        x[0] = 0.0f;
        return 0.0f;
    }

    /* Scale: y = x / amax */
    float alpha = x[0] / amax;

    /* ||y||^2 (vectorized) */
    __m256 acc = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < len; i += 8)
    {
        __m256 xv = _mm256_loadu_ps(x + i);
        xv = _mm256_div_ps(xv, _mm256_set1_ps(amax));
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    float normy2 = 0.0f;
    {
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        normy2 = _mm_cvtss_f32(s);
    }
    for (; i < len; ++i)
    {
        const float yi = x[i] / amax;
        normy2 += yi * yi;
    }

    float sigma = normy2 - alpha * alpha; /* sum(y^2) - y0^2 */
    if (sigma <= 0.0f)
    {
        v[0] = 1.0f;
        for (uint16_t t = 1; t < len; ++t)
            v[t] = 0.0f;
        x[0] = -x[0]; /* keep contract "x[0] = -beta" (beta=|x0|) */
        return 0.0f;
    }

    /* normx = amax * ||y|| */
    float normy = sqrtf(alpha * alpha + sigma);
    float beta_scaled = (alpha <= 0.0f) ? (alpha - normy)
                                        : (-sigma / (alpha + normy));
    float beta = beta_scaled * amax; /* rescale */

    /* tau = 2 / (1 + (sigma / beta_scaled^2))  (same as 2*β^2/(σ+β^2)) */
    float tau = (2.0f * beta_scaled * beta_scaled) / (sigma + beta_scaled * beta_scaled);

    /* v */
    v[0] = 1.0f;
    for (uint16_t t = 1; t < len; ++t)
        v[t] = x[t] / beta;
    x[0] = -beta;
    return tau;
}

/**
 * @brief Apply a Householder reflector from the left to a trailing block (AVX2, blocked).
 *
 * @details
 *  Computes Rk ← Rk − v (τ vᵀ Rk), where Rk points to the top-left of the
 *  trailing block (R[k:,k:]) with leading dimension @p ld and width @p nc.
 *  The update is carried out in row/column tiles to improve cache locality.
 *
 * @param[in,out] Rk   Pointer to R[k,k] (row-major), updated in place.
 * @param[in]     ld   Leading dimension of R (number of columns n).
 * @param[in]     v    Householder vector of length @p len (v[0]=1).
 * @param[in]     len  Number of rows in the trailing block (m-k).
 * @param[in]     tau  Householder scalar τ.
 * @param[in]     nc   Number of columns in the trailing block (n-k).
 *
 * @note Temporarily allocates a workspace t[0..nc-1] (aligned).
 * @warning Returns silently if workspace allocation fails. Caller should ensure
 *          Rk has room for (len×nc) elements from the given pointer/ld.
 */
static void apply_reflector_left_blocked_avx(float *RESTRICT Rk, uint16_t ld,
                                             const float *RESTRICT v, uint16_t len,
                                             float tau, uint16_t nc)
{
    if (tau == 0.0f)
        return;

    const uint16_t Jc = (uint16_t)LINALG_BLOCK_JC;
    const uint16_t Kc = (uint16_t)LINALG_BLOCK_KC;

    /* temp: t[0..nc-1] */
    float *t = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)nc * sizeof(float));
    if (!t)
        return;

    /* t = tau * (v^T * Rk) — compute in column tiles */
    for (uint16_t c0 = 0; c0 < nc; c0 += Kc)
    {
        const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
        float *tc = t + c0;

        /* fast zero tc */
        uint16_t zc = 0;
        const __m256 z = _mm256_setzero_ps();
        for (; zc + 7 < kc; zc += 8)
            _mm256_storeu_ps(tc + zc, z);
        for (; zc < kc; ++zc)
            tc[zc] = 0.0f;

        /* accumulate v^T * Rk in row tiles */
        for (uint16_t r0 = 0; r0 < len; r0 += Jc)
        {
            const uint16_t jr = (uint16_t)((r0 + Jc <= len) ? Jc : (len - r0));
            uint16_t r = r0;

            /* 8-row chunks */
            for (; r + 7 < r0 + jr; r += 8)
            {
                const float *row0 = Rk + (size_t)(r + 0) * ld + c0;
                const float *row1 = Rk + (size_t)(r + 1) * ld + c0;
                const float *row2 = Rk + (size_t)(r + 2) * ld + c0;
                const float *row3 = Rk + (size_t)(r + 3) * ld + c0;
                const float *row4 = Rk + (size_t)(r + 4) * ld + c0;
                const float *row5 = Rk + (size_t)(r + 5) * ld + c0;
                const float *row6 = Rk + (size_t)(r + 6) * ld + c0;
                const float *row7 = Rk + (size_t)(r + 7) * ld + c0;

                const __m256 v0 = _mm256_broadcast_ss(v + (r + 0));
                const __m256 v1 = _mm256_broadcast_ss(v + (r + 1));
                const __m256 v2 = _mm256_broadcast_ss(v + (r + 2));
                const __m256 v3 = _mm256_broadcast_ss(v + (r + 3));
                const __m256 v4 = _mm256_broadcast_ss(v + (r + 4));
                const __m256 v5 = _mm256_broadcast_ss(v + (r + 5));
                const __m256 v6 = _mm256_broadcast_ss(v + (r + 6));
                const __m256 v7 = _mm256_broadcast_ss(v + (r + 7));

                uint16_t cc = 0;
                for (; cc + 7 < kc; cc += 8)
                {
                    __m256 acc = _mm256_loadu_ps(tc + cc);

                    _mm_prefetch((const char *)(row0 + cc + 64), _MM_HINT_T0);
                    acc = _mm256_fmadd_ps(v0, _mm256_loadu_ps(row0 + cc), acc);
                    acc = _mm256_fmadd_ps(v1, _mm256_loadu_ps(row1 + cc), acc);
                    acc = _mm256_fmadd_ps(v2, _mm256_loadu_ps(row2 + cc), acc);
                    acc = _mm256_fmadd_ps(v3, _mm256_loadu_ps(row3 + cc), acc);
                    acc = _mm256_fmadd_ps(v4, _mm256_loadu_ps(row4 + cc), acc);
                    acc = _mm256_fmadd_ps(v5, _mm256_loadu_ps(row5 + cc), acc);
                    acc = _mm256_fmadd_ps(v6, _mm256_loadu_ps(row6 + cc), acc);
                    acc = _mm256_fmadd_ps(v7, _mm256_loadu_ps(row7 + cc), acc);

                    _mm256_storeu_ps(tc + cc, acc);
                }
                for (; cc < kc; ++cc)
                {
                    tc[cc] += v[r + 0] * row0[cc] + v[r + 1] * row1[cc] + v[r + 2] * row2[cc] + v[r + 3] * row3[cc] + v[r + 4] * row4[cc] + v[r + 5] * row5[cc] + v[r + 6] * row6[cc] + v[r + 7] * row7[cc];
                }
            }

            /* leftover rows in this row tile */
            for (; r < r0 + jr; ++r)
            {
                const float vr = v[r];
                const float *row = Rk + (size_t)r * ld + c0;
                uint16_t cc2 = 0;
                const __m256 vrv = _mm256_set1_ps(vr);
                for (; cc2 + 7 < kc; cc2 += 8)
                {
                    __m256 acc = _mm256_loadu_ps(tc + cc2);
                    acc = _mm256_fmadd_ps(vrv, _mm256_loadu_ps(row + cc2), acc);
                    _mm256_storeu_ps(tc + cc2, acc);
                }
                for (; cc2 < kc; ++cc2)
                    tc[cc2] += vr * row[cc2];
            }
        }

        /* scale t by tau */
        const __m256 tv = _mm256_set1_ps(tau);
        uint16_t sc = 0;
        for (; sc + 7 < kc; sc += 8)
        {
            __m256 vv = _mm256_loadu_ps(tc + sc);
            _mm256_storeu_ps(tc + sc, _mm256_mul_ps(vv, tv));
        }
        for (; sc < kc; ++sc)
            tc[sc] *= tau;
    }

    /* Apply: Rk -= v * t   (rank-1 update) */
    for (uint16_t r0 = 0; r0 < len; r0 += Jc)
    {
        const uint16_t jr = (uint16_t)((r0 + Jc <= len) ? Jc : (len - r0));
        uint16_t r = r0;

        /* 8-row chunks */
        for (; r + 7 < r0 + jr; r += 8)
        {
            float *row0 = Rk + (size_t)(r + 0) * ld;
            float *row1 = Rk + (size_t)(r + 1) * ld;
            float *row2 = Rk + (size_t)(r + 2) * ld;
            float *row3 = Rk + (size_t)(r + 3) * ld;
            float *row4 = Rk + (size_t)(r + 4) * ld;
            float *row5 = Rk + (size_t)(r + 5) * ld;
            float *row6 = Rk + (size_t)(r + 6) * ld;
            float *row7 = Rk + (size_t)(r + 7) * ld;

            const __m256 v0 = _mm256_broadcast_ss(v + (r + 0));
            const __m256 v1 = _mm256_broadcast_ss(v + (r + 1));
            const __m256 v2 = _mm256_broadcast_ss(v + (r + 2));
            const __m256 v3 = _mm256_broadcast_ss(v + (r + 3));
            const __m256 v4 = _mm256_broadcast_ss(v + (r + 4));
            const __m256 v5 = _mm256_broadcast_ss(v + (r + 5));
            const __m256 v6 = _mm256_broadcast_ss(v + (r + 6));
            const __m256 v7 = _mm256_broadcast_ss(v + (r + 7));

            for (uint16_t c0 = 0; c0 < nc; c0 += Kc)
            {
                const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
                float *tc = t + c0;

                uint16_t cc = 0;
                for (; cc + 7 < kc; cc += 8)
                {
                    const __m256 t8 = _mm256_loadu_ps(tc + cc);

                    __m256 r0v = _mm256_loadu_ps(row0 + c0 + cc);
                    __m256 r1v = _mm256_loadu_ps(row1 + c0 + cc);
                    __m256 r2v = _mm256_loadu_ps(row2 + c0 + cc);
                    __m256 r3v = _mm256_loadu_ps(row3 + c0 + cc);
                    __m256 r4v = _mm256_loadu_ps(row4 + c0 + cc);
                    __m256 r5v = _mm256_loadu_ps(row5 + c0 + cc);
                    __m256 r6v = _mm256_loadu_ps(row6 + c0 + cc);
                    __m256 r7v = _mm256_loadu_ps(row7 + c0 + cc);

                    r0v = _mm256_fnmadd_ps(v0, t8, r0v);
                    r1v = _mm256_fnmadd_ps(v1, t8, r1v);
                    r2v = _mm256_fnmadd_ps(v2, t8, r2v);
                    r3v = _mm256_fnmadd_ps(v3, t8, r3v);
                    r4v = _mm256_fnmadd_ps(v4, t8, r4v);
                    r5v = _mm256_fnmadd_ps(v5, t8, r5v);
                    r6v = _mm256_fnmadd_ps(v6, t8, r6v);
                    r7v = _mm256_fnmadd_ps(v7, t8, r7v);

                    _mm256_storeu_ps(row0 + c0 + cc, r0v);
                    _mm256_storeu_ps(row1 + c0 + cc, r1v);
                    _mm256_storeu_ps(row2 + c0 + cc, r2v);
                    _mm256_storeu_ps(row3 + c0 + cc, r3v);
                    _mm256_storeu_ps(row4 + c0 + cc, r4v);
                    _mm256_storeu_ps(row5 + c0 + cc, r5v);
                    _mm256_storeu_ps(row6 + c0 + cc, r6v);
                    _mm256_storeu_ps(row7 + c0 + cc, r7v);
                }
                for (; cc < kc; ++cc)
                {
                    const float tc1 = tc[cc];
                    row0[c0 + cc] -= v[r + 0] * tc1;
                    row1[c0 + cc] -= v[r + 1] * tc1;
                    row2[c0 + cc] -= v[r + 2] * tc1;
                    row3[c0 + cc] -= v[r + 3] * tc1;
                    row4[c0 + cc] -= v[r + 4] * tc1;
                    row5[c0 + cc] -= v[r + 5] * tc1;
                    row6[c0 + cc] -= v[r + 6] * tc1;
                    row7[c0 + cc] -= v[r + 7] * tc1;
                }
            }
        }

        /* leftover rows in this tile */
        for (; r < r0 + jr; ++r)
        {
            const float vr = v[r];
            float *row = Rk + (size_t)r * ld;
            const __m256 vrv = _mm256_set1_ps(vr);

            for (uint16_t c0 = 0; c0 < nc; c0 += Kc)
            {
                const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
                float *tc = t + c0;

                uint16_t cc = 0;
                for (; cc + 7 < kc; cc += 8)
                {
                    __m256 rv = _mm256_loadu_ps(row + c0 + cc);
                    rv = _mm256_fnmadd_ps(vrv, _mm256_loadu_ps(tc + cc), rv);
                    _mm256_storeu_ps(row + c0 + cc, rv);
                }
                for (; cc < kc; ++cc)
                    row[c0 + cc] -= vr * tc[cc];
            }
        }
    }

    linalg_aligned_free(t);
}

/**
 * @brief Apply a Householder reflector to Q from the right (AVX2).
 *
 * @details
 *  Updates Q[:,k:] ← Q[:,k:] − (τ Q[:,k:] v) vᵀ, where v has length @p len
 *  and @p k is the starting column. Uses AVX2 for both the row-wise dot product
 *  (Q[:,k:] * v) and the subsequent rank-1 update.
 *
 * @param[in,out] Q    Orthogonal accumulator (m×m), updated in place.
 * @param[in]     m    Order of Q (rows == cols).
 * @param[in]     k    Starting column index for the trailing block.
 * @param[in]     v    Householder vector (length @p len, v[0]=1).
 * @param[in]     len  Number of columns in the trailing block (m-k).
 * @param[in]     tau  Householder scalar τ.
 *
 * @note Allocates a temporary y[0..m-1] (aligned) for the intermediate product.
 * @warning No bounds checks; caller guarantees valid ranges (k+len ≤ m).
 */
static void apply_reflector_right_Q_avx(float *RESTRICT Q, uint16_t m,
                                        uint16_t k,
                                        const float *RESTRICT v, uint16_t len,
                                        float tau)
{
    if (tau == 0.0f)
        return;

    float *y = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(float));
    if (!y)
        return;

    const uint16_t Jc = (uint16_t)LINALG_BLOCK_JC; /* reuse as row-blocking for Q */

    /* y = tau * (Q[:,k:] * v) */
    for (uint16_t r0 = 0; r0 < m; r0 += Jc)
    {
        const uint16_t jr = (uint16_t)((r0 + Jc <= m) ? Jc : (m - r0));
        for (uint16_t r = 0; r < jr; ++r)
        {
            const float *q = Q + (size_t)(r0 + r) * m + k;
            uint16_t j = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; j + 7 < len; j += 8)
            {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + j), _mm256_loadu_ps(v + j), acc);
            }
            float sum = hsum8_ps(acc);
            for (; j < len; ++j)
                sum += q[j] * v[j];
            y[r0 + r] = sum * tau;
        }
    }

    /* Q[:,k:] -= y * v^T  (rank-1, block rows) */
    for (uint16_t r0 = 0; r0 < m; r0 += Jc)
    {
        const uint16_t jr = (uint16_t)((r0 + Jc <= m) ? Jc : (m - r0));
        for (uint16_t r = 0; r < jr; ++r)
        {
            float *q = Q + (size_t)(r0 + r) * m + k;
            const __m256 yr = _mm256_set1_ps(y[r0 + r]);
            uint16_t j = 0;
            for (; j + 7 < len; j += 8)
            {
                __m256 qv = _mm256_loadu_ps(q + j);
                __m256 vv = _mm256_loadu_ps(v + j);
                qv = _mm256_fnmadd_ps(yr, vv, qv);
                _mm256_storeu_ps(q + j, qv);
            }
            for (; j < len; ++j)
                q[j] -= y[r0 + r] * v[j];
        }
    }

    linalg_aligned_free(y);
}
#endif /* LINALG_SIMD_ENABLE */

/**
 * @brief Apply a Householder reflector to Q from the right (scalar fallback).
 *
 * @details
 *  Scalar equivalent of apply_reflector_right_Q_avx(). Computes y = τ Q[:,k:] v
 *  and applies the rank-1 update Q[:,k:] -= y vᵀ using simple loops.
 *
 * @param[in,out] Q    Orthogonal accumulator (m×m), updated in place.
 * @param[in]     m    Order of Q (rows == cols).
 * @param[in]     k    Starting column index for the trailing block.
 * @param[in]     v    Householder vector (length @p len, v[0]=1).
 * @param[in]     len  Number of columns in the trailing block (m-k).
 * @param[in]     tau  Householder scalar τ.
 *
 * @note Used when LINALG_SIMD_ENABLE is false at build time.
 */
#if !LINALG_SIMD_ENABLE
static void apply_reflector_right_Q_scalar(float *RESTRICT Q, uint16_t m,
                                           uint16_t k,
                                           const float *RESTRICT v, uint16_t len,
                                           float tau)
{
    if (tau == 0.0f)
        return;
    float *y = (float *)malloc((size_t)m * sizeof(float));
    if (!y)
        return;
    memset(y, 0, (size_t)m * sizeof(float));

    /* y = Q[:,k:] * v */
    for (uint16_t r = 0; r < m; ++r)
    {
        const float *q = Q + (size_t)r * m + k;
        float sum = 0.0f;
        for (uint16_t j = 0; j < len; ++j)
            sum += q[j] * v[j];
        y[r] = sum * tau;
    }
    /* Q[:,k:] -= y * v^T */
    for (uint16_t r = 0; r < m; ++r)
    {
        float *q = Q + (size_t)r * m + k;
        const float yr = y[r];
        for (uint16_t j = 0; j < len; ++j)
            q[j] -= yr * v[j];
    }
    free(y);
}
#endif

/**
 * @brief Single-precision QR decomposition via Householder reflections.
 *
 * @details
 *  Computes the QR factorization of A (m×n), producing R (upper-triangular).
 *  If @p only_R is false, also forms Q explicitly. The implementation chooses
 *  between a blocked AVX2/FMA path and a scalar reference path:
 *   - **Vectorized path**: Robust Householder vector generation, blocked left
 *     application on R, and right application on Q (if requested). Tuned with
 *     LINALG_BLOCK_{JC,KC} for cache residency and uses aligned workspaces.
 *   - **Scalar path**: Portable, small-matrix implementation that constructs
 *     reflectors and multiplies them using mul()/inv() helpers.
 *
 * @param[in]  A       Input matrix (m×n), row-major.
 * @param[out] Q       Output orthogonal matrix (m×m). Ignored if @p only_R=true.
 * @param[out] R       Output upper-triangular factor (m×n).
 * @param[in]  m       Number of rows of A.
 * @param[in]  n       Number of columns of A.
 * @param[in]  only_R  If true, compute only R (skip forming Q).
 *
 * @retval 0          Success.
 * @retval -EINVAL    Invalid sizes (m==0 or n==0).
 * @retval -ENOMEM    Allocation failure for temporary buffers (vector path).
 *
 * @warning Q and R must not alias A. Buffers must be valid and sized.
 * @note The vector path is selected when AVX2 is available and the problem
 *       is large enough (mn ≥ LINALG_SMALL_N_THRESH).
 */
int qr(const float *RESTRICT A, float *RESTRICT Q, float *RESTRICT R,
       uint16_t m, uint16_t n, bool only_R)
{
    if (m == 0 || n == 0)
        return -EINVAL;

    const uint16_t mn = (m < n) ? m : n;

    /* Small-matrix or no-AVX fallback */
    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2()
#if !LINALG_SIMD_ENABLE
        || 1
#endif
    )
    {
        return qr_scalar(A, Q, R, m, n, only_R);
    }

    /* R ← A, Q ← I (if needed) */
    memcpy(R, A, (size_t)m * n * sizeof(float));
    if (!only_R)
    {
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i)
            Q[(size_t)i * m + i] = 1.0f;
    }

#if LINALG_SIMD_ENABLE
    /* workspace: v (len ≤ m) */
    float *v = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(float));
    if (!v)
        return -ENOMEM;

    const uint16_t l = (m - 1 < n) ? (m - 1) : n;
    for (uint16_t k = 0; k < l; ++k)
    {
        float *x = R + (size_t)k * n + k; /* column segment R[k:,k] */
        uint16_t len = (uint16_t)(m - k);
        uint16_t nc = (uint16_t)(n - k);

        float tau = householder_vec_avx(v, x, len);
        if (tau != 0.0f)
        {
            apply_reflector_left_blocked_avx(x, n, v, len, tau, nc);

            for (uint16_t i = k + 1; i < m; ++i)
                R[(size_t)i * n + k] = 0.0f;

            if (!only_R)
            {
                apply_reflector_right_Q_avx(Q, m, k, v, len, tau);
            }
        }
    }
    linalg_aligned_free(v);
#else
    /* Should not reach here thanks to the guard above */
    return qr_scalar(A, Q, R, m, n, only_R);
#endif

    return 0;
}
