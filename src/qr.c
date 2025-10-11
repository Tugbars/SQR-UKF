// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <immintrin.h>

#include "linalg_simd.h"   // RESTRICT, linalg_has_avx2(), LINALG_* knobs
// also exports mul(), inv() that qr_scalar uses

/* ---------------- small scalar QR (original path, tidied) ---------------- */
static int qr_scalar(const float *RESTRICT A, float *RESTRICT Q,
                     float *RESTRICT R, uint16_t m, uint16_t n, bool only_R)
{
    const uint16_t l = (m - 1 < n) ? (m - 1) : n;

    memcpy(R, A, (size_t)m * n * sizeof(float));

    float *H   = (float*)malloc((size_t)m * m * sizeof(float));
    float *W   = (float*)malloc((size_t)m * sizeof(float));
    float *WW  = (float*)malloc((size_t)m * m * sizeof(float));
    float *Hi  = (float*)malloc((size_t)m * m * sizeof(float));
    float *HiH = (float*)malloc((size_t)m * m * sizeof(float));
    float *HiR = (float*)malloc((size_t)m * n * sizeof(float));
    if (!H || !W || !WW || !Hi || !HiH || !HiR) {
        free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
        return -ENOMEM;
    }

    memset(H, 0, (size_t)m * m * sizeof(float));
    for (uint16_t i = 0; i < m; ++i) H[(size_t)i * m + i] = 1.0f;

    for (uint16_t k = 0; k < l; ++k) {
        float s = 0.0f;
        for (uint16_t i = k; i < m; ++i) {
            float x = R[(size_t)i * n + k];
            s += x * x;
        }
        s = sqrtf(s);
        float Rk = R[(size_t)k * n + k];
        if (Rk < 0.0f) s = -s;
        float r = sqrtf(2.0f * s * (Rk + s));

        memset(W, 0, (size_t)m * sizeof(float));
        W[k] = (Rk + s) / r;
        for (uint16_t i = k + 1; i < m; ++i) W[i] = R[(size_t)i * n + k] / r;

        mul(WW, W, W, m, 1, 1, m);                   // WW = W * Wᵀ
        for (size_t i = 0; i < (size_t)m * m; ++i) Hi[i] = -2.0f * WW[i];
        for (uint16_t i = 0; i < m; ++i) Hi[(size_t)i * m + i] += 1.0f;

        if (!only_R) {
            mul(HiH, Hi, H, m, m, m, m);
            memcpy(H, HiH, (size_t)m * m * sizeof(float));
        }
        mul(HiR, Hi, R, m, m, m, n);
        memcpy(R, HiR, (size_t)m * n * sizeof(float));
    }

    if (!only_R) {
        if (inv(H, H, m) != 0) { free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR); return -ENOTSUP; }
        memcpy(Q, H, (size_t)m * m * sizeof(float));
    }

    free(H); free(W); free(WW); free(Hi); free(HiH); free(HiR);
    return 0;
}

#if LINALG_SIMD_ENABLE
/* ---------------- AVX helpers (compiled only with AVX2/FMA) ---------------- */

static inline float hsum8_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline __m256 bcast_lane(__m256 pack, int lane) {
    const __m256i idx = _mm256_set1_epi32(lane);
    return _mm256_permutevar8x32_ps(pack, idx);
}

/* Householder for x (len): returns tau, writes v (v[0]=1, v[1:]=x/beta), and sets x[0] = -beta */
static float householder_vec_avx(float *RESTRICT v, float *RESTRICT x, uint16_t len)
{
    __m256 acc = _mm256_setzero_ps();
    uint16_t i = 0;
    for (; i + 7 < len; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    float norm2 = hsum8_ps(acc);
    for (; i < len; ++i) norm2 += x[i] * x[i];

    float alpha = x[0];
    float sigma = norm2 - alpha * alpha;
    if (sigma <= 0.0f) { v[0] = 1.0f; for (uint16_t t = 1; t < len; ++t) v[t] = 0.0f; return 0.0f; }

    float normx = sqrtf(alpha * alpha + sigma);
    float beta  = (alpha <= 0.0f) ? (alpha - normx) : (-sigma / (alpha + normx));
    float tau   = (2.0f * beta * beta) / (sigma + beta * beta);

    v[0] = 1.0f;
    for (uint16_t t = 1; t < len; ++t) v[t] = x[t] / beta;
    x[0] = -beta;
    return tau;
}

/**
 * Apply reflector to R trailing block with **row(Jc)/col(Kc) blocking**.
 *
 * Rk     : pointer to R[k,k] (top-left of trailing block), size len x nc, row-major with ld = n
 * ld     : leading dimension of R (n)
 * v,len  : Householder vector (len), with v[0]=1
 * tau    : scalar
 * nc     : number of columns in the trailing block (n - k)
 *
 * We compute:
 *   t = tau * (v^T * Rk)   [length nc]     (accumulate in column blocks Kc)
 *   Rk -= v * t            [rank-1 update] (apply in row blocks Jc, column blocks Kc)
 *
 * Blocking keeps slices hot in L1 and reduces cache thrash on big matrices.
 */
static void apply_reflector_left_blocked_avx(float *RESTRICT Rk, uint16_t ld,
                                             const float *RESTRICT v, uint16_t len,
                                             float tau, uint16_t nc)
{
    if (tau == 0.0f) return;

    const uint16_t Jc = (uint16_t)LINALG_BLOCK_JC;
    const uint16_t Kc = (uint16_t)LINALG_BLOCK_KC;

    float *t = (float*)malloc((size_t)nc * sizeof(float));
    if (!t) return;

    /* t = tau * (v^T * Rk) in column blocks */
    for (uint16_t c0 = 0; c0 < nc; c0 += Kc) {
        const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
        float *tc = t + c0;
        memset(tc, 0, (size_t)kc * sizeof(float));

        for (uint16_t r0 = 0; r0 < len; r0 += Jc) {
            const uint16_t jr = (uint16_t)((r0 + Jc <= len) ? Jc : (len - r0));
            uint16_t r = r0;

            for (; r + 7 < r0 + jr; r += 8) {
                __m256 vpack = _mm256_loadu_ps(v + r);
                __m256 v0 = bcast_lane(vpack, 0);
                __m256 v1 = bcast_lane(vpack, 1);
                __m256 v2 = bcast_lane(vpack, 2);
                __m256 v3 = bcast_lane(vpack, 3);
                __m256 v4 = bcast_lane(vpack, 4);
                __m256 v5 = bcast_lane(vpack, 5);
                __m256 v6 = bcast_lane(vpack, 6);
                __m256 v7 = bcast_lane(vpack, 7);

                const float *row0 = Rk + (size_t)(r+0) * ld + c0;
                const float *row1 = Rk + (size_t)(r+1) * ld + c0;
                const float *row2 = Rk + (size_t)(r+2) * ld + c0;
                const float *row3 = Rk + (size_t)(r+3) * ld + c0;
                const float *row4 = Rk + (size_t)(r+4) * ld + c0;
                const float *row5 = Rk + (size_t)(r+5) * ld + c0;
                const float *row6 = Rk + (size_t)(r+6) * ld + c0;
                const float *row7 = Rk + (size_t)(r+7) * ld + c0;

                uint16_t c = 0;
                for (; c + 7 < kc; c += 8) {
                    __m256 acc = _mm256_loadu_ps(tc + c);
                    acc = _mm256_fmadd_ps(v0, _mm256_loadu_ps(row0 + c), acc);
                    acc = _mm256_fmadd_ps(v1, _mm256_loadu_ps(row1 + c), acc);
                    acc = _mm256_fmadd_ps(v2, _mm256_loadu_ps(row2 + c), acc);
                    acc = _mm256_fmadd_ps(v3, _mm256_loadu_ps(row3 + c), acc);
                    acc = _mm256_fmadd_ps(v4, _mm256_loadu_ps(row4 + c), acc);
                    acc = _mm256_fmadd_ps(v5, _mm256_loadu_ps(row5 + c), acc);
                    acc = _mm256_fmadd_ps(v6, _mm256_loadu_ps(row6 + c), acc);
                    acc = _mm256_fmadd_ps(v7, _mm256_loadu_ps(row7 + c), acc);
                    _mm256_storeu_ps(tc + c, acc);
                }
                for (; c < kc; ++c) {
                    tc[c] += v[r+0]*row0[c] + v[r+1]*row1[c] + v[r+2]*row2[c] + v[r+3]*row3[c]
                           +  v[r+4]*row4[c] + v[r+5]*row5[c] + v[r+6]*row6[c] + v[r+7]*row7[c];
                }
            }

            for (; r < r0 + jr; ++r) {
                const float vr = v[r];
                const float *row = Rk + (size_t)r * ld + c0;
                uint16_t c = 0;
                for (; c + 7 < kc; c += 8) {
                    __m256 acc = _mm256_loadu_ps(tc + c);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(vr), _mm256_loadu_ps(row + c), acc);
                    _mm256_storeu_ps(tc + c, acc);
                }
                for (; c < kc; ++c) tc[c] += vr * row[c];
            }
        }

        /* scale by tau */
        __m256 tv = _mm256_set1_ps(tau);
        uint16_t c = 0;
        for (; c + 7 < kc; c += 8) {
            __m256 vv = _mm256_loadu_ps(tc + c);
            _mm256_storeu_ps(tc + c, _mm256_mul_ps(vv, tv));
        }
        for (; c < kc; ++c) tc[c] *= tau;
    }

    /* Rk -= v * t  (Jc x Kc tiles; 8-row unroll) */
    for (uint16_t r0 = 0; r0 < len; r0 += Jc) {
        const uint16_t jr = (uint16_t)((r0 + Jc <= len) ? Jc : (len - r0));
        uint16_t r = r0;

        for (; r + 7 < r0 + jr; r += 8) {
            __m256 vpack = _mm256_loadu_ps(v + r);
            __m256 v0 = bcast_lane(vpack, 0);
            __m256 v1 = bcast_lane(vpack, 1);
            __m256 v2 = bcast_lane(vpack, 2);
            __m256 v3 = bcast_lane(vpack, 3);
            __m256 v4 = bcast_lane(vpack, 4);
            __m256 v5 = bcast_lane(vpack, 5);
            __m256 v6 = bcast_lane(vpack, 6);
            __m256 v7 = bcast_lane(vpack, 7);

            float *row0 = Rk + (size_t)(r+0) * ld;
            float *row1 = Rk + (size_t)(r+1) * ld;
            float *row2 = Rk + (size_t)(r+2) * ld;
            float *row3 = Rk + (size_t)(r+3) * ld;
            float *row4 = Rk + (size_t)(r+4) * ld;
            float *row5 = Rk + (size_t)(r+5) * ld;
            float *row6 = Rk + (size_t)(r+6) * ld;
            float *row7 = Rk + (size_t)(r+7) * ld;

            for (uint16_t c0 = 0; c0 < nc; c0 += Kc) {
                const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
                float *tc = t + c0;

                uint16_t c = 0;
                for (; c + 7 < kc; c += 8) {
                    __m256 t8 = _mm256_loadu_ps(tc + c);

                    __m256 r0v = _mm256_loadu_ps(row0 + c0 + c);
                    __m256 r1v = _mm256_loadu_ps(row1 + c0 + c);
                    __m256 r2v = _mm256_loadu_ps(row2 + c0 + c);
                    __m256 r3v = _mm256_loadu_ps(row3 + c0 + c);
                    __m256 r4v = _mm256_loadu_ps(row4 + c0 + c);
                    __m256 r5v = _mm256_loadu_ps(row5 + c0 + c);
                    __m256 r6v = _mm256_loadu_ps(row6 + c0 + c);
                    __m256 r7v = _mm256_loadu_ps(row7 + c0 + c);

                    r0v = _mm256_fnmadd_ps(v0, t8, r0v);
                    r1v = _mm256_fnmadd_ps(v1, t8, r1v);
                    r2v = _mm256_fnmadd_ps(v2, t8, r2v);
                    r3v = _mm256_fnmadd_ps(v3, t8, r3v);
                    r4v = _mm256_fnmadd_ps(v4, t8, r4v);
                    r5v = _mm256_fnmadd_ps(v5, t8, r5v);
                    r6v = _mm256_fnmadd_ps(v6, t8, r6v);
                    r7v = _mm256_fnmadd_ps(v7, t8, r7v);

                    _mm256_storeu_ps(row0 + c0 + c, r0v);
                    _mm256_storeu_ps(row1 + c0 + c, r1v);
                    _mm256_storeu_ps(row2 + c0 + c, r2v);
                    _mm256_storeu_ps(row3 + c0 + c, r3v);
                    _mm256_storeu_ps(row4 + c0 + c, r4v);
                    _mm256_storeu_ps(row5 + c0 + c, r5v);
                    _mm256_storeu_ps(row6 + c0 + c, r6v);
                    _mm256_storeu_ps(row7 + c0 + c, r7v);
                }
                for (; c < kc; ++c) {
                    float tc1 = tc[c];
                    row0[c0 + c] -= v[r+0] * tc1;
                    row1[c0 + c] -= v[r+1] * tc1;
                    row2[c0 + c] -= v[r+2] * tc1;
                    row3[c0 + c] -= v[r+3] * tc1;
                    row4[c0 + c] -= v[r+4] * tc1;
                    row5[c0 + c] -= v[r+5] * tc1;
                    row6[c0 + c] -= v[r+6] * tc1;
                    row7[c0 + c] -= v[r+7] * tc1;
                }
            }
        }

        for (; r < r0 + jr; ++r) {
            const float vr = v[r];
            float *row = Rk + (size_t)r * ld;
            for (uint16_t c0 = 0; c0 < nc; c0 += Kc) {
                const uint16_t kc = (uint16_t)((c0 + Kc <= nc) ? Kc : (nc - c0));
                float *tc = t + c0;
                uint16_t c = 0;
                for (; c + 7 < kc; c += 8) {
                    __m256 rv = _mm256_loadu_ps(row + c0 + c);
                    rv = _mm256_fnmadd_ps(_mm256_set1_ps(vr), _mm256_loadu_ps(tc + c), rv);
                    _mm256_storeu_ps(row + c0 + c, rv);
                }
                for (; c < kc; ++c) row[c0 + c] -= vr * tc[c];
            }
        }
    }

    free(t);
}
#endif /* LINALG_SIMD_ENABLE */

/* Apply reflector to Q on the right (scalar, simple & correct). */
static void apply_reflector_right_Q_scalar(float *RESTRICT Q, uint16_t m,
                                           uint16_t k,
                                           const float *RESTRICT v, uint16_t len,
                                           float tau)
{
    if (tau == 0.0f) return;
    float *y = (float*)malloc((size_t)m * sizeof(float));
    if (!y) return;
    memset(y, 0, (size_t)m * sizeof(float));

    /* y = Q[:,k:] * v */
    for (uint16_t r = 0; r < m; ++r) {
        const float *q = Q + (size_t)r * m + k;
        float sum = 0.0f;
        for (uint16_t j = 0; j < len; ++j) sum += q[j] * v[j];
        y[r] = sum * tau;
    }
    /* Q[:,k:] -= y * v^T */
    for (uint16_t r = 0; r < m; ++r) {
        float *q = Q + (size_t)r * m + k;
        const float yr = y[r];
        for (uint16_t j = 0; j < len; ++j) q[j] -= yr * v[j];
    }
    free(y);
}

/**
 * @brief      Single-precision QR decomposition via Householder reflections.
 *             Vectorized path uses blocked AVX2/FMA; tiny or non-AVX falls back to scalar.
 */
int qr(const float *RESTRICT A, float *RESTRICT Q, float *RESTRICT R,
       uint16_t m, uint16_t n, bool only_R)
{
    if (m == 0 || n == 0) return -EINVAL;

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
    if (!only_R) {
        memset(Q, 0, (size_t)m * m * sizeof(float));
        for (uint16_t i = 0; i < m; ++i) Q[(size_t)i * m + i] = 1.0f;
    }

#if LINALG_SIMD_ENABLE
    /* workspace: v (len ≤ m) */
    float *v = (float*)aligned_alloc(32, (size_t)m * sizeof(float));
    if (!v) return -ENOMEM;

    const uint16_t l = (m - 1 < n) ? (m - 1) : n;
    for (uint16_t k = 0; k < l; ++k) {
        float *x   = R + (size_t)k * n + k;  /* column segment R[k:,k] */
        uint16_t len = (uint16_t)(m - k);
        uint16_t nc  = (uint16_t)(n - k);

        float tau = householder_vec_avx(v, x, len);
        if (tau != 0.0f) {
            apply_reflector_left_blocked_avx(x, n, v, len, tau, nc);

            for (uint16_t i = k + 1; i < m; ++i) R[(size_t)i * n + k] = 0.0f;

            if (!only_R) {
                apply_reflector_right_Q_scalar(Q, m, k, v, len, tau);
            }
        }
    }
    free(v);
#else
    /* Should not reach here thanks to the guard above */
    return qr_scalar(A, Q, R, m, n, only_R);
#endif

    return 0;
}
