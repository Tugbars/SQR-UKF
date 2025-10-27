#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include "linalg_simd.h" 

// Compile AVX2 helpers only when the TU is built with AVX2/FMA enabled
#if LINALG_SIMD_ENABLE
static inline __m256 bcast_lane(__m256 pack, int lane) {
    const __m256i idx = _mm256_set1_epi32(lane);
    return _mm256_permutevar8x32_ps(pack, idx);
}
#endif

static int solve_scalar(const float *RESTRICT LU, float *RESTRICT x,
                        const uint8_t *RESTRICT P, uint16_t n, uint16_t col)
{
    // forward
    for (uint16_t i = 0; i < n; ++i) {
        size_t rbase = (size_t)n * P[i];
        float xi = (P[i] == col) ? 1.0f : 0.0f;
        for (uint16_t j = 0; j < i; ++j)
            xi -= LU[rbase + (size_t)j] * x[j];
        x[i] = xi;
    }
    // backward
    for (int i = (int)n - 1; i >= 0; --i) {
        size_t rbase = (size_t)n * P[(uint16_t)i];

        float s = 0.0f;
        for (uint16_t j = (uint16_t)i + 1; j < n; ++j)
            s += LU[rbase + (size_t)j] * x[j];

        /* relative tolerance: scale by max |U(row, k)| for k>=i */
        float scale = 0.0f;
        for (uint16_t k = (uint16_t)i; k < n; ++k) {
            float ak = fabsf(LU[rbase + (size_t)k]);
            if (ak > scale) scale = ak;
        }
        float d = LU[rbase + (size_t)i];
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(d) <= tol) return -ENOTSUP;

        x[i] = (x[i] - s) / d;
    }
    return 0;
}

/* Process up to 8 RHS columns (col0..col0+jb-1). Writes directly to Ai_out (row-major). */
static int solve_block8_avx2_rowmajor(const float *RESTRICT LU,
                                      const uint8_t *RESTRICT P,
                                      uint16_t n, uint16_t col0, uint16_t jb,
                                      float *RESTRICT Ai_out)
{
#if !LINALG_SIMD_ENABLE
    (void)LU; (void)P; (void)n; (void)col0; (void)jb; (void)Ai_out;
    return -ENOTSUP; // should never be called without AVX2; caller guards this
#else
    // n x 8 scratch block for RHS solutions
    float *X = (float*)linalg_aligned_alloc(32, (size_t)n * 8 * sizeof(float));
    if (!X) return -ENOMEM;
    memset(X, 0, (size_t)n * 8 * sizeof(float));

    // Build RHS mask lanes for e_{col0..col0+7}
    int32_t cols_i32[8] = {0};
    for (int t = 0; t < 8; ++t) cols_i32[t] = (int32_t)(col0 + t);
    const __m256i cols = _mm256_loadu_si256((const __m256i*)cols_i32);

    // Blocking over accumulation dimension (Jc)
    const uint16_t Jc = 256; // tune 192–320

    /* ---------------- forward substitution ---------------- */
    for (uint16_t i = 0; i < n; ++i) {
        const size_t rbase = (size_t)n * P[i];

        // Start with RHS mask (1 where P[i] == col, else 0)
        const __m256 pivv = _mm256_cvtepi32_ps(_mm256_cmpeq_epi32(_mm256_set1_epi32((int)P[i]), cols));
        __m256 xi = pivv;

        // Accumulate sum_{j<i} L(row, j) * X[j,:] in blocks
        for (uint16_t j0 = 0; j0 < i; j0 += Jc) {
            const uint16_t j1 = (uint16_t)((j0 + Jc < i) ? (j0 + Jc) : i);
            uint16_t j = j0;

            // unroll by 8 with broadcast hoisting
            for (; j + 7 < j1; j += 8) {
                // load 8 L coefficients at once
                __m256 lpack = _mm256_loadu_ps(&LU[rbase + (size_t)j]);

                // l0..l7 as broadcasted via permute (no set1/bcast)
                __m256 l0 = bcast_lane(lpack, 0);
                __m256 l1 = bcast_lane(lpack, 1);
                __m256 l2 = bcast_lane(lpack, 2);
                __m256 l3 = bcast_lane(lpack, 3);
                __m256 l4 = bcast_lane(lpack, 4);
                __m256 l5 = bcast_lane(lpack, 5);
                __m256 l6 = bcast_lane(lpack, 6);
                __m256 l7 = bcast_lane(lpack, 7);

                __m256 x0 = _mm256_load_ps(&X[(size_t)(j+0)*8]);
                __m256 x1 = _mm256_load_ps(&X[(size_t)(j+1)*8]);
                __m256 x2 = _mm256_load_ps(&X[(size_t)(j+2)*8]);
                __m256 x3 = _mm256_load_ps(&X[(size_t)(j+3)*8]);
                __m256 x4 = _mm256_load_ps(&X[(size_t)(j+4)*8]);
                __m256 x5 = _mm256_load_ps(&X[(size_t)(j+5)*8]);
                __m256 x6 = _mm256_load_ps(&X[(size_t)(j+6)*8]);
                __m256 x7 = _mm256_load_ps(&X[(size_t)(j+7)*8]);

                xi = _mm256_fnmadd_ps(l0, x0, xi);
                xi = _mm256_fnmadd_ps(l1, x1, xi);
                xi = _mm256_fnmadd_ps(l2, x2, xi);
                xi = _mm256_fnmadd_ps(l3, x3, xi);
                xi = _mm256_fnmadd_ps(l4, x4, xi);
                xi = _mm256_fnmadd_ps(l5, x5, xi);
                xi = _mm256_fnmadd_ps(l6, x6, xi);
                xi = _mm256_fnmadd_ps(l7, x7, xi);
            }
            for (; j < j1; ++j) {
                __m256 lj = _mm256_set1_ps(LU[rbase + (size_t)j]);
                __m256 xj = _mm256_load_ps(&X[(size_t)j*8]);
                xi = _mm256_fnmadd_ps(lj, xj, xi);
            }
        }

        // Store xi row to X[i,:] – jb may be < 8 → partial store
        if (jb == 8) {
            _mm256_store_ps(&X[(size_t)i*8], xi);
        } else {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, xi);
            for (uint16_t t = 0; t < jb; ++t)
                X[(size_t)i*8 + t] = tmp[t];
            // zero the inactive lanes to keep later loads clean
            for (uint16_t t = jb; t < 8; ++t)
                X[(size_t)i*8 + t] = 0.0f;
        }
    }

    /* ---------------- backward substitution ---------------- */
    for (int i = (int)n - 1; i >= 0; --i) {
        const size_t rbase = (size_t)n * P[(uint16_t)i];

        __m256 acc = _mm256_setzero_ps();
        // sum over j>i in blocks
        for (uint16_t j0 = (uint16_t)i + 1; j0 < n; j0 += Jc) {
            const uint16_t j1 = (uint16_t)((j0 + Jc < n) ? (j0 + Jc) : n);
            uint16_t j = j0;

            for (; j + 7 < j1; j += 8) {
                __m256 upack = _mm256_loadu_ps(&LU[rbase + (size_t)j]);

                __m256 u0 = bcast_lane(upack, 0);
                __m256 u1 = bcast_lane(upack, 1);
                __m256 u2 = bcast_lane(upack, 2);
                __m256 u3 = bcast_lane(upack, 3);
                __m256 u4 = bcast_lane(upack, 4);
                __m256 u5 = bcast_lane(upack, 5);
                __m256 u6 = bcast_lane(upack, 6);
                __m256 u7 = bcast_lane(upack, 7);

                __m256 x0 = _mm256_load_ps(&X[(size_t)(j+0)*8]);
                __m256 x1 = _mm256_load_ps(&X[(size_t)(j+1)*8]);
                __m256 x2 = _mm256_load_ps(&X[(size_t)(j+2)*8]);
                __m256 x3 = _mm256_load_ps(&X[(size_t)(j+3)*8]);
                __m256 x4 = _mm256_load_ps(&X[(size_t)(j+4)*8]);
                __m256 x5 = _mm256_load_ps(&X[(size_t)(j+5)*8]);
                __m256 x6 = _mm256_load_ps(&X[(size_t)(j+6)*8]);
                __m256 x7 = _mm256_load_ps(&X[(size_t)(j+7)*8]);

                acc = _mm256_fmadd_ps(u0, x0, acc);
                acc = _mm256_fmadd_ps(u1, x1, acc);
                acc = _mm256_fmadd_ps(u2, x2, acc);
                acc = _mm256_fmadd_ps(u3, x3, acc);
                acc = _mm256_fmadd_ps(u4, x4, acc);
                acc = _mm256_fmadd_ps(u5, x5, acc);
                acc = _mm256_fmadd_ps(u6, x6, acc);
                acc = _mm256_fmadd_ps(u7, x7, acc);
            }
            for (; j < j1; ++j) {
                __m256 uj = _mm256_set1_ps(LU[rbase + (size_t)j]);
                __m256 xj = _mm256_load_ps(&X[(size_t)j*8]);
                acc = _mm256_fmadd_ps(uj, xj, acc);
            }
        }

        // Relative tolerance using row max |U|
        float scale = 0.0f;
        for (uint16_t k = (uint16_t)i; k < n; ++k) {
            float ak = fabsf(LU[rbase + (size_t)k]);
            if (ak > scale) scale = ak;
        }
        float d = LU[rbase + (size_t)i];
        float tol = (float)n * FLT_EPSILON * scale;
        if (fabsf(d) <= tol) { free(X); return -ENOTSUP; }

        __m256 invd = _mm256_set1_ps(1.0f / d);
        __m256 xi = _mm256_mul_ps(_mm256_sub_ps(_mm256_load_ps(&X[(size_t)i*8]), acc), invd);

        // Store back to X[i,:], respecting jb
        if (jb == 8) {
            _mm256_store_ps(&X[(size_t)i*8], xi);
        } else {
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, xi);
            for (uint16_t t = 0; t < jb; ++t)
                X[(size_t)i*8 + t] = tmp[t];
        }
    }

    /* ---------------- write to Ai_out (row-major) ---------------- */
    for (uint16_t r = 0; r < n; ++r) {
        const float *rowv = &X[(size_t)r * 8];
        for (uint16_t t = 0; t < jb; ++t)
            Ai_out[(size_t)r * n + (col0 + t)] = rowv[t];
    }

    linalg_aligned_free(X);
    return 0;
#endif
}

int inv(float *RESTRICT Ai_out, const float *RESTRICT A, uint16_t n)
{
    if (n == 0) return -EINVAL;

    // Optional tiny-matrix fast path (keeps code consistent with other kernels)
    if (n < LINALG_SMALL_N_THRESH) {
        float  LU[(size_t)n * n];
        uint8_t P[n];
        if (lup(A, LU, P, n) != 0) return -ENOTSUP;
        float x[n];
        for (uint16_t col = 0; col < n; ++col) {
            if (solve_scalar(LU, x, P, n, col) != 0) return -ENOTSUP;
            for (uint16_t r = 0; r < n; ++r)
                Ai_out[(size_t)r * n + col] = x[r];
        }
        return 0;
    }

    float  LU[(size_t)n * n];
    uint8_t P[n];
    if (lup(A, LU, P, n) != 0) return -ENOTSUP;

    // Runtime gate that also respects compile-time ISA
    const int use_avx =
    #if LINALG_SIMD_ENABLE
        linalg_has_avx2();
    #else
        0;
    #endif

    if (!use_avx) {
        // scalar fallback (unchanged)
        float x[n];
        for (uint16_t col = 0; col < n; ++col) {
            if (solve_scalar(LU, x, P, n, col) != 0) return -ENOTSUP;
            for (uint16_t r = 0; r < n; ++r)
                Ai_out[(size_t)r * n + col] = x[r];
        }
        return 0;
    }

    // AVX2 block-RHS path (unchanged)
    for (uint16_t col0 = 0; col0 < n; col0 += 8) {
        uint16_t jb = (uint16_t)((col0 + 8 <= n) ? 8 : (n - col0));
        int rc = solve_block8_avx2_rowmajor(LU, P, n, col0, jb, Ai_out);
        if (rc) return rc;
    }
    return 0;
}
