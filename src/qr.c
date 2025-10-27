/**
 * @file
 * @brief Blocked compact-WY QR (single-precision) with AVX2/FMA kernels.
 *
 * @details
 * This implementation factors an m×n row-major matrix A into Q·R using
 * Householder reflections. It follows the LAPACK/BLAS pattern:
 *  - **Panel factorization (unblocked)**: GEQR2 over a panel of width @ref QRW_IB_DEFAULT.
 *  - **Form T (compact-WY)**: LARFT builds the ib×ib triangular T for the panel V.
 *  - **Blocked application to trailing matrix**: LARFB-style update via three BLAS-3 shaped
 *    steps: Y = Vᵀ·C, Z = T·Y, C ← C − V·Z. These are implemented with small packers and
 *    AVX2/FMA vectorized kernels (dual accumulators, contiguous loads across kc).
 *
 * **Data layout and outputs**
 *  - Input is row-major A (m×n). The routine copies A→R and factors **in-place**.
 *  - On return, the **upper triangle of R** is the R factor. The **strict lower triangle**
 *    stores the Householder reflectors V; the corresponding scalars τ are kept internally.
 *  - Q is **not** formed unless requested. When needed, ORGQR builds Q (m×m) using the same
 *    blocked machinery (no per-reflector rank-1 updates).
 *
 * **Dispatch**
 *  - For small problems (mn < @ref LINALG_SMALL_N_THRESH) or when AVX2 is unavailable,
 *    a scalar reference QR path is used.
 *  - Otherwise, the blocked compact-WY path is selected.
 *
 * **SIMD & alignment**
 *  - AVX2/FMA kernels assume 32-byte alignment for workspace allocations
 *    (enforced by linalg_aligned_alloc). Unaligned loads are used where layout
 *    prohibits alignment guarantees (e.g., packed tiles), but hot buffers are aligned.
 *
 * **Tuning knobs**
 *  - @ref QRW_IB_DEFAULT : Panel width (ib). Try 64–96 on Intel 14900KF.
 *  - @ref LINALG_BLOCK_KC : Trailing-block tile width (kc) for packed updates, e.g., 256–320.
 *  - @ref LINALG_SMALL_N_THRESH : Switch to scalar path for small mn.
 *
 * **API overview**
 *  - `int qr(const float* A, float* Q, float* R, uint16_t m, uint16_t n, bool only_R);`
 *      - Copies A→R, computes R and (optionally) Q.
 *      - Returns 0 on success, negative errno on failure.
 *  - Internal helpers: blocked GEQRF (in-place reflectors + τ), ORGQR (forms Q on demand),
 *    tiny pack/unpack, and AVX2 kernels for Y/Z/VZ.
 *  - Optional: a minimal CPQR (`geqp3_blocked`) is provided but not wired into `qr()`.
 *
 * **Numerics**
 *  - Householder vectors use a robust constructor with scaling by ‖x‖∞ to avoid
 *    overflow/underflow and Parlett’s choice for β to minimize cancellation.
 *  - Compact-WY preserves the numerical stability of classical Householder QR while
 *    improving performance via BLAS-3-like updates.
 *
 * @note  Single-precision build by default. Hooks exist to mirror s/d and add c/z variants.
 * @note  This file is single-threaded by design; parallelization can be layered around
 *        the GEMM-shaped updates if needed.
 * @warning Q and R must not alias A. All buffers must be valid and sized.
 * @warning The reflectors (V) overwrite the strict lower triangle of R; if you need A later,
 *          keep your own copy.
 *
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <immintrin.h>

#include "linalg_simd.h" // RESTRICT, linalg_has_avx2(), LINALG_* knobs, linalg_aligned_alloc/free
// also exports mul(), inv() that qr_scalar uses

#ifndef LINALG_SMALL_N_THRESH
#define LINALG_SMALL_N_THRESH 48
#endif

#ifndef LINALG_BLOCK_KC
#define LINALG_BLOCK_KC 256
#endif

#ifndef LINALG_BLOCK_JC
#define LINALG_BLOCK_JC 64
#endif

#ifndef QRW_IB_DEFAULT
#define QRW_IB_DEFAULT 64 // try 64 or 96 on 14900KF
#endif

_Static_assert(LINALG_DEFAULT_ALIGNMENT >= 32, "Need 32B alignment for AVX loads");

/* ===========================================================================================
 * Scalar (reference) QR (unchanged, small matrices or no-AVX fallback)
 * ===========================================================================================
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
        if (s == 0.0f)
            continue; // guard: nothing to do on this column
        if (Rk < 0.0f)
            s = -s;
        float r = sqrtf(2.0f * s * (Rk + s));
        if (r == 0.0f)
            continue; // guard: avoid division by zero

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

/* ===========================================================================================
 * Blocked compact-WY QR (single precision; scalar + AVX2 kernels)
 * ===========================================================================================
 */

typedef float qrw_t;

/* ------------------ Householder + Panel (unblocked) ------------------ */

// robust Householder for a contiguous vector x[0..len-1]
static qrw_t qrw_householder_robust(qrw_t *RESTRICT x, uint16_t len, qrw_t *beta_out)
{
    if (len == 0)
    {
        *beta_out = 0;
        return 0;
    }

    qrw_t amax = 0;
    for (uint16_t i = 0; i < len; ++i)
    {
        qrw_t ax = (qrw_t)fabs((double)x[i]);
        if (ax > amax)
            amax = ax;
    }
    if (amax == 0)
    {
        *beta_out = 0;
        x[0] = 1;
        return 0;
    }

    qrw_t alpha = x[0] / amax;
    qrw_t normy2 = 0;
    for (uint16_t i = 0; i < len; ++i)
    {
        qrw_t yi = x[i] / amax;
        normy2 += yi * yi;
    }
    qrw_t sigma = normy2 - alpha * alpha;
    if (sigma <= 0)
    {
        *beta_out = -x[0];
        x[0] = 1;
        return 0;
    }

    qrw_t normy = (qrw_t)sqrt((double)(alpha * alpha + sigma));
    qrw_t beta_scaled = (alpha <= 0) ? (alpha - normy) : (-sigma / (alpha + normy));
    qrw_t beta = beta_scaled * amax;
    qrw_t b2 = beta_scaled * beta_scaled;
    qrw_t tau = (qrw_t)2.0 * b2 / (sigma + b2);

    qrw_t invb = 1.0f / beta;
    for (uint16_t i = 1; i < len; ++i)
        x[i] *= invb;
    x[0] = 1.0f;
    *beta_out = beta;
    return tau;
}

// Panel QR (unblocked Householders)
static void qrw_panel_geqr2(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                            uint16_t k, uint16_t ib, qrw_t *RESTRICT tau_panel,
                            qrw_t *RESTRICT tmp /* len >= m */)
{
    const uint16_t end = (uint16_t)((k + ib <= n) ? (k + ib) : n);

    for (uint16_t j = k; j < end; ++j)
    {
        uint16_t rows = (uint16_t)(m - j);
        qrw_t *colj0 = A + (size_t)j * n + j; // A[j,j] (row-major, down column uses stride n)
        for (uint16_t r = 0; r < rows; ++r)
            tmp[r] = colj0[(size_t)r * n];

        qrw_t beta;
        qrw_t tauj = qrw_householder_robust(tmp, rows, &beta);
        tau_panel[j - k] = tauj;

        for (uint16_t r = 0; r < rows; ++r)
            colj0[(size_t)r * n] = tmp[r];
        *(A + (size_t)j * n + j) = -beta;

        if (tauj != 0 && j + 1 < end)
        {
            for (uint16_t c = (uint16_t)(j + 1); c < end; ++c)
            {
                qrw_t sum = 0;
                for (uint16_t r = 0; r < rows; ++r)
                    sum += colj0[(size_t)r * n] * A[(size_t)(j + r) * n + c];
                sum *= tauj;
                for (uint16_t r = 0; r < rows; ++r)
                    A[(size_t)(j + r) * n + c] -= colj0[(size_t)r * n] * sum;
            }
        }
    }
}

/* ------------------ Build T (LARFT) ------------------ */

static void qrw_larft(qrw_t *RESTRICT T, uint16_t ib,
                      const qrw_t *RESTRICT A, uint16_t m, uint16_t n, uint16_t k,
                      const qrw_t *RESTRICT tau_panel)
{
    for (uint16_t i = 0; i < ib; ++i)
        for (uint16_t j = 0; j < ib; ++j)
            T[(size_t)i * ib + j] = 0;

    for (uint16_t j = 0; j < ib; ++j)
    {
        for (uint16_t i = 0; i < j; ++i)
        {
            const qrw_t *vi = A + (size_t)(k + i) * n + (k + i);
            const qrw_t *vj = A + (size_t)(k + j) * n + (k + j);
            uint16_t len_j = (uint16_t)(m - (k + j));
            qrw_t sum = vi[(size_t)(j - i) * n]; // vj[0] == 1
            for (uint16_t r = 1; r < len_j; ++r)
                sum += vi[(size_t)(j - i + r) * n] * vj[(size_t)r * n];
            T[(size_t)i * ib + j] = -tau_panel[j] * sum;
        }
        T[(size_t)j * ib + j] = tau_panel[j];

        for (int i = (int)j - 1; i >= 0; --i)
        {
            qrw_t acc = T[(size_t)i * ib + j];
            for (uint16_t p = (uint16_t)(i + 1); p < j; ++p)
                acc += T[(size_t)i * ib + p] * T[(size_t)p * ib + j];
            T[(size_t)i * ib + j] = acc;
        }
    }
}

/* ------------------ Packers ------------------ */

static void qrw_pack_C(const qrw_t *RESTRICT C, uint16_t ld, uint16_t m_sub,
                       uint16_t c0, uint16_t kc, qrw_t *RESTRICT Cp)
{
    for (uint16_t r = 0; r < m_sub; ++r)
    {
        const qrw_t *src = C + (size_t)r * ld + c0;
        memcpy(Cp + (size_t)r * kc, src, (size_t)kc * sizeof(qrw_t));
    }
}

static void qrw_unpack_C(qrw_t *RESTRICT C, uint16_t ld, uint16_t m_sub,
                         uint16_t c0, uint16_t kc, const qrw_t *RESTRICT Cp)
{
    for (uint16_t r = 0; r < m_sub; ++r)
    {
        qrw_t *dst = C + (size_t)r * ld + c0;
        memcpy(dst, Cp + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
    }
}

/* ------------------ Scalar Level-3 (fallback) ------------------ */

// Y = V^T * Cpack   (ib × kc)
static void qrw_compute_Y_scalar(const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                 uint16_t k, uint16_t ib,
                                 const qrw_t *RESTRICT Cpack, uint16_t m_sub,
                                 uint16_t kc, qrw_t *RESTRICT Y)
{
    for (uint16_t j = 0; j < kc; ++j)
    {
        for (uint16_t p = 0; p < ib; ++p)
        {
            const qrw_t *vp = A + (size_t)(k + p) * n + (k + p);
            uint16_t len = (uint16_t)(m - (k + p));
            qrw_t sum = 0;
            for (uint16_t r = 0; r < len; ++r)
                sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
            Y[(size_t)p * kc + j] = sum;
        }
    }
}

// Z = T * Y   (ib × kc)
static void qrw_compute_Z_scalar(const qrw_t *RESTRICT T, uint16_t ib,
                                 const qrw_t *RESTRICT Y, uint16_t kc,
                                 qrw_t *RESTRICT Z)
{
    for (uint16_t i = 0; i < ib; ++i)
    {
        for (uint16_t j = 0; j < kc; ++j)
        {
            qrw_t sum = 0;
            for (uint16_t p = 0; p < ib; ++p)
                sum += T[(size_t)i * ib + p] * Y[(size_t)p * kc + j];
            Z[(size_t)i * kc + j] = sum;
        }
    }
}

// Cpack = Cpack − V * Z
static void qrw_apply_VZ_scalar(qrw_t *RESTRICT Cpack, uint16_t m_sub, uint16_t kc,
                                const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                uint16_t k, uint16_t ib,
                                const qrw_t *RESTRICT Z)
{
    for (uint16_t j = 0; j < kc; ++j)
    {
        for (uint16_t p = 0; p < ib; ++p)
        {
            const qrw_t *vp = A + (size_t)(k + p) * n + (k + p);
            uint16_t len = (uint16_t)(m - (k + p));
            qrw_t zp = Z[(size_t)p * kc + j];
            for (uint16_t r = 0; r < len; ++r)
                Cpack[(size_t)(r + p) * kc + j] -= vp[(size_t)r * n] * zp;
        }
    }
}

/* ------------------ AVX2 Level-3 (vectorized) ------------------ */

#if LINALG_SIMD_ENABLE
// horizontal sum for __m256
static inline float qrw_hsum8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

// Vectorized: Y = V^T * Cpack  (ib × kc)
// We vectorize across columns j in chunks of 16 (two 8-lane accumulators).
static void qrw_compute_Y_avx(const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                              uint16_t k, uint16_t ib,
                              const qrw_t *RESTRICT Cpack, uint16_t m_sub,
                              uint16_t kc, qrw_t *RESTRICT Y)
{
    (void)m_sub; // not needed, we derive from m,k,p
    for (uint16_t p = 0; p < ib; ++p)
    {
        const float *vp = A + (size_t)(k + p) * n + (k + p);
        const uint16_t len = (uint16_t)(m - (k + p));

        uint16_t j = 0;
        // alignment peel for Cpack row start (row offset p)
        // Each row r contributes Cpack[(r+p)*kc + j]
        // We'll just use unaligned loads for simplicity on Cpack; we still peel to make stores aligned if desired.
        for (; j + 15 < kc; j += 16)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            // accumulate over r
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                acc0 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 0), acc0);
                acc1 = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr + 8), acc1);
            }
            _mm256_storeu_ps(Y + (size_t)p * kc + j + 0, acc0);
            _mm256_storeu_ps(Y + (size_t)p * kc + j + 8, acc1);
        }
        for (; j + 7 < kc; j += 8)
        {
            __m256 acc = _mm256_setzero_ps();
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vv = _mm256_set1_ps(vp[(size_t)r * n]);
                const float *cptr = Cpack + (size_t)(r + p) * kc + j;
                acc = _mm256_fmadd_ps(vv, _mm256_loadu_ps(cptr), acc);
            }
            _mm256_storeu_ps(Y + (size_t)p * kc + j, acc);
        }
        for (; j < kc; ++j)
        {
            float sum = 0.0f;
            for (uint16_t r = 0; r < len; ++r)
                sum += vp[(size_t)r * n] * Cpack[(size_t)(r + p) * kc + j];
            Y[(size_t)p * kc + j] = sum;
        }
    }
}

// Vectorized: Z = T * Y  (ib × kc)
// Vectorize across kc columns with 16-wide chunks; broadcast T(i,p).
static void qrw_compute_Z_avx(const qrw_t *RESTRICT T, uint16_t ib,
                              const qrw_t *RESTRICT Y, uint16_t kc,
                              qrw_t *RESTRICT Z)
{
    for (uint16_t i = 0; i < ib; ++i)
    {
        uint16_t j = 0;
        for (; j + 15 < kc; j += 16)
        {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (uint16_t p = 0; p < ib; ++p)
            {
                const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                const float *y = Y + (size_t)p * kc + j;
                acc0 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 0), acc0);
                acc1 = _mm256_fmadd_ps(t, _mm256_loadu_ps(y + 8), acc1);
            }
            _mm256_storeu_ps(Z + (size_t)i * kc + j + 0, acc0);
            _mm256_storeu_ps(Z + (size_t)i * kc + j + 8, acc1);
        }
        for (; j + 7 < kc; j += 8)
        {
            __m256 acc = _mm256_setzero_ps();
            for (uint16_t p = 0; p < ib; ++p)
            {
                const __m256 t = _mm256_set1_ps(T[(size_t)i * ib + p]);
                const float *y = Y + (size_t)p * kc + j;
                acc = _mm256_fmadd_ps(t, _mm256_loadu_ps(y), acc);
            }
            _mm256_storeu_ps(Z + (size_t)i * kc + j, acc);
        }
        for (; j < kc; ++j)
        {
            float sum = 0.0f;
            for (uint16_t p = 0; p < ib; ++p)
                sum += T[(size_t)i * ib + p] * Y[(size_t)p * kc + j];
            Z[(size_t)i * kc + j] = sum;
        }
    }
}

// Vectorized: Cpack = Cpack − V * Z
// Vectorize across kc columns similarly; broadcast each v_p[r] and subtract v*z row by row.
static void qrw_apply_VZ_avx(qrw_t *RESTRICT Cpack, uint16_t m_sub, uint16_t kc,
                             const qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                             uint16_t k, uint16_t ib,
                             const qrw_t *RESTRICT Z)
{
    for (uint16_t p = 0; p < ib; ++p)
    {
        const float *vp = A + (size_t)(k + p) * n + (k + p);
        const uint16_t len = (uint16_t)(m - (k + p));

        uint16_t j = 0;
        for (; j + 15 < kc; j += 16)
        {
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vz0 = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                 _mm256_loadu_ps(Z + (size_t)p * kc + j + 0));
                const __m256 vz1 = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                 _mm256_loadu_ps(Z + (size_t)p * kc + j + 8));
                float *cptr = Cpack + (size_t)(r + p) * kc + j;
                __m256 c0 = _mm256_loadu_ps(cptr + 0);
                __m256 c1 = _mm256_loadu_ps(cptr + 8);
                c0 = _mm256_sub_ps(c0, vz0);
                c1 = _mm256_sub_ps(c1, vz1);
                _mm256_storeu_ps(cptr + 0, c0);
                _mm256_storeu_ps(cptr + 8, c1);
            }
        }
        for (; j + 7 < kc; j += 8)
        {
            for (uint16_t r = 0; r < len; ++r)
            {
                const __m256 vz = _mm256_mul_ps(_mm256_set1_ps(vp[(size_t)r * n]),
                                                _mm256_loadu_ps(Z + (size_t)p * kc + j));
                float *cptr = Cpack + (size_t)(r + p) * kc + j;
                __m256 c = _mm256_loadu_ps(cptr);
                c = _mm256_sub_ps(c, vz);
                _mm256_storeu_ps(cptr, c);
            }
        }
        for (; j < kc; ++j)
        {
            for (uint16_t r = 0; r < len; ++r)
                Cpack[(size_t)(r + p) * kc + j] -= vp[(size_t)r * n] * Z[(size_t)p * kc + j];
        }
    }
}
#endif /* LINALG_SIMD_ENABLE */

/* ------------------ Blocked driver: GEQRF (in-place reflectors + tau) ------------------ */

static int qrw_geqrf_blocked_wy(qrw_t *RESTRICT A, uint16_t m, uint16_t n,
                                uint16_t ib, qrw_t *RESTRICT tau_out)
{
    if (m == 0 || n == 0)
        return 0;
    if (ib == 0)
        ib = QRW_IB_DEFAULT;

    const uint16_t kmax = (m < n) ? m : n;
    qrw_t *tmp = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(qrw_t));
    if (!tmp)
        return -ENOMEM;

    uint16_t k = 0;
    while (k < kmax)
    {
        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        qrw_t *tau_panel = tau_out + k;

        // 1) Panel factorization
        qrw_panel_geqr2(A, m, n, k, ib_k, tau_panel, tmp);

        // 2) Build T
        qrw_t *T = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(qrw_t));
        if (!T)
        {
            linalg_aligned_free(tmp);
            return -ENOMEM;
        }
        qrw_larft(T, ib_k, A, m, n, k, tau_panel);

        // 3) Apply block reflector to trailing matrix C = A[k:m, k+ib_k:n]
        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t nc = (uint16_t)(n - (k + ib_k));
        if (nc)
        {
            const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;
            qrw_t *Cpack = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_sub * kc_tile * sizeof(qrw_t));
            qrw_t *Y = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            qrw_t *Z = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            if (!Cpack || !Y || !Z)
            {
                if (Cpack)
                    linalg_aligned_free(Cpack);
                if (Y)
                    linalg_aligned_free(Y);
                if (Z)
                    linalg_aligned_free(Z);
                linalg_aligned_free(T);
                linalg_aligned_free(tmp);
                return -ENOMEM;
            }

            qrw_t *C = A + (size_t)k * n + (k + ib_k);
            for (uint16_t c0 = 0; c0 < nc; c0 += kc_tile)
            {
                uint16_t kc = (uint16_t)((c0 + kc_tile <= nc) ? kc_tile : (nc - c0));
                qrw_pack_C(C, n, m_sub, c0, kc, Cpack);
#if LINALG_SIMD_ENABLE
                qrw_compute_Y_avx(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_avx(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_avx(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#else
                qrw_compute_Y_scalar(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#endif
                qrw_unpack_C(C, n, m_sub, c0, kc, Cpack);
            }

            linalg_aligned_free(Cpack);
            linalg_aligned_free(Y);
            linalg_aligned_free(Z);
        }

        linalg_aligned_free(T);
        k = (uint16_t)(k + ib_k);
    }

    linalg_aligned_free(tmp);
    return 0;
}

/* ------------------ ORGQR: form Q explicitly from (A,V,tau) ------------------ */

static int qrw_orgqr_full(qrw_t *RESTRICT Q, uint16_t m,
                          const qrw_t *RESTRICT A, uint16_t n,
                          const qrw_t *RESTRICT tau, uint16_t kreflect)
{
    for (uint16_t r = 0; r < m; ++r)
    {
        for (uint16_t c = 0; c < m; ++c)
            Q[(size_t)r * m + c] = (r == c) ? 1.0f : 0.0f;
    }

    if (kreflect == 0)
        return 0;

    const uint16_t ib_def = QRW_IB_DEFAULT;

    int64_t kk = (int64_t)kreflect;
    while (kk > 0)
    {
        uint16_t ib_k = (uint16_t)((kk >= ib_def) ? ib_def : kk);
        uint16_t kstart = (uint16_t)(kk - ib_k);

        qrw_t *T = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(qrw_t));
        if (!T)
            return -ENOMEM;
        qrw_larft(T, ib_k, A, m, n, kstart, tau + kstart);

        const uint16_t m_sub = (uint16_t)(m - kstart);
        const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;

        qrw_t *Cpack = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_sub * kc_tile * sizeof(qrw_t));
        qrw_t *Y = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
        qrw_t *Z = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
        if (!Cpack || !Y || !Z)
        {
            if (Cpack)
                linalg_aligned_free(Cpack);
            if (Y)
                linalg_aligned_free(Y);
            if (Z)
                linalg_aligned_free(Z);
            linalg_aligned_free(T);
            return -ENOMEM;
        }

        for (uint16_t c0 = 0; c0 < m; c0 += kc_tile)
        {
            uint16_t kc = (uint16_t)((c0 + kc_tile <= m) ? kc_tile : (m - c0));
            qrw_t *C = Q + (size_t)kstart * m;
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(Cpack + (size_t)r * kc, C + (size_t)r * m + c0, (size_t)kc * sizeof(qrw_t));
#if LINALG_SIMD_ENABLE
            qrw_compute_Y_avx(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_avx(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_avx(Cpack, m_sub, kc, A, m, n, kstart, ib_k, Z);
#else
            qrw_compute_Y_scalar(A, m, n, kstart, ib_k, Cpack, m_sub, kc, Y);
            qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
            qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, kstart, ib_k, Z);
#endif
            for (uint16_t r = 0; r < m_sub; ++r)
                memcpy(C + (size_t)r * m + c0, Cpack + (size_t)r * kc, (size_t)kc * sizeof(qrw_t));
        }

        linalg_aligned_free(Cpack);
        linalg_aligned_free(Y);
        linalg_aligned_free(Z);
        linalg_aligned_free(T);
        kk -= ib_k;
    }
    return 0;
}

/* ===========================================================================================
 * Optional: CPQR (GEQP3) minimal (unchanged from previous drop)
 * ===========================================================================================
 */

static void qrw_swap_cols(qrw_t *A, uint16_t m, uint16_t n, uint16_t j1, uint16_t j2)
{
    if (j1 == j2)
        return;
    for (uint16_t r = 0; r < m; ++r)
    {
        qrw_t tmp = A[(size_t)r * n + j1];
        A[(size_t)r * n + j1] = A[(size_t)r * n + j2];
        A[(size_t)r * n + j2] = tmp;
    }
}

static void qrw_colnorms(const qrw_t *RESTRICT A, uint16_t m, uint16_t n, uint16_t k,
                         qrw_t *RESTRICT nrms)
{
    for (uint16_t j = k; j < n; ++j)
    {
        qrw_t s = 0;
        for (uint16_t r = k; r < m; ++r)
        {
            qrw_t v = A[(size_t)r * n + j];
            s += v * v;
        }
        nrms[j] = (qrw_t)sqrt((double)s);
    }
}

int geqp3_blocked(float *RESTRICT A, uint16_t m, uint16_t n,
                  uint16_t ib, float *RESTRICT tau, int *RESTRICT jpvt)
{
    for (uint16_t j = 0; j < n; ++j)
        jpvt[j] = j;

    qrw_t *nrm = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(qrw_t));
    qrw_t *nrm_ref = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)n * sizeof(qrw_t));
    qrw_t *tmp = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m * sizeof(qrw_t));
    if (!nrm || !nrm_ref || !tmp)
    {
        if (nrm)
            linalg_aligned_free(nrm);
        if (nrm_ref)
            linalg_aligned_free(nrm_ref);
        if (tmp)
            linalg_aligned_free(tmp);
        return -ENOMEM;
    }

    qrw_colnorms(A, m, n, 0, nrm);
    memcpy(nrm_ref, nrm, (size_t)n * sizeof(qrw_t));

    const uint16_t kmax = (m < n) ? m : n;
    uint16_t k = 0;
    while (k < kmax)
    {
        uint16_t pvt = k;
        qrw_t best = nrm[pvt];
        for (uint16_t j = (uint16_t)(k + 1); j < n; ++j)
            if (nrm[j] > best)
            {
                best = nrm[j];
                pvt = j;
            }
        qrw_swap_cols(A, m, n, k, pvt);
        int tmpi = jpvt[k];
        jpvt[k] = jpvt[pvt];
        jpvt[pvt] = tmpi;
        qrw_t tn = nrm[k];
        nrm[k] = nrm[pvt];
        nrm[pvt] = tn;
        tn = nrm_ref[k];
        nrm_ref[k] = nrm_ref[pvt];
        nrm_ref[pvt] = tn;

        uint16_t ib_k = (uint16_t)((k + ib <= kmax) ? ib : (kmax - k));
        qrw_panel_geqr2(A, m, n, k, ib_k, tau + k, tmp);

        qrw_t *T = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * ib_k * sizeof(qrw_t));
        if (!T)
        {
            linalg_aligned_free(nrm);
            linalg_aligned_free(nrm_ref);
            linalg_aligned_free(tmp);
            return -ENOMEM;
        }
        qrw_larft(T, ib_k, A, m, n, k, tau + k);

        const uint16_t m_sub = (uint16_t)(m - k);
        const uint16_t nc = (uint16_t)(n - (k + ib_k));
        if (nc)
        {
            const uint16_t kc_tile = (uint16_t)LINALG_BLOCK_KC;
            qrw_t *Cpack = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)m_sub * kc_tile * sizeof(qrw_t));
            qrw_t *Y = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            qrw_t *Z = (qrw_t *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)ib_k * kc_tile * sizeof(qrw_t));
            if (!Cpack || !Y || !Z)
            {
                if (Cpack)
                    linalg_aligned_free(Cpack);
                if (Y)
                    linalg_aligned_free(Y);
                if (Z)
                    linalg_aligned_free(Z);
                linalg_aligned_free(T);
                linalg_aligned_free(nrm);
                linalg_aligned_free(nrm_ref);
                linalg_aligned_free(tmp);
                return -ENOMEM;
            }
            qrw_t *C = A + (size_t)k * n + (k + ib_k);
            for (uint16_t c0 = 0; c0 < nc; c0 += kc_tile)
            {
                uint16_t kc = (uint16_t)((c0 + kc_tile <= nc) ? kc_tile : (nc - c0));
                qrw_pack_C(C, n, m_sub, c0, kc, Cpack);
#if LINALG_SIMD_ENABLE
                qrw_compute_Y_avx(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_avx(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_avx(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#else
                qrw_compute_Y_scalar(A, m, n, k, ib_k, Cpack, m_sub, kc, Y);
                qrw_compute_Z_scalar(T, ib_k, Y, kc, Z);
                qrw_apply_VZ_scalar(Cpack, m_sub, kc, A, m, n, k, ib_k, Z);
#endif
                qrw_unpack_C(C, n, m_sub, c0, kc, Cpack);
            }
            linalg_aligned_free(Cpack);
            linalg_aligned_free(Y);
            linalg_aligned_free(Z);
        }
        linalg_aligned_free(T);

        k = (uint16_t)(k + ib_k);
    }

    linalg_aligned_free(nrm);
    linalg_aligned_free(nrm_ref);
    linalg_aligned_free(tmp);
    return 0;
}

/* ===========================================================================================
 * Public entry: qr() — chooses blocked WY or scalar path; forms Q only if requested
 * ===========================================================================================
 */

int qr(const float *RESTRICT A, float *RESTRICT Q, float *RESTRICT R,
       uint16_t m, uint16_t n, bool only_R)
{
    if (m == 0 || n == 0)
        return -EINVAL;

    const uint16_t mn = (m < n) ? m : n;

    if (mn < LINALG_SMALL_N_THRESH || !linalg_has_avx2())
    {
        return qr_scalar(A, Q, R, m, n, only_R);
    }

    memcpy(R, A, (size_t)m * n * sizeof(float)); // Factor in R
    float *tau = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, (size_t)mn * sizeof(float));
    if (!tau)
        return -ENOMEM;

    int rc = qrw_geqrf_blocked_wy(R, m, n, QRW_IB_DEFAULT, tau);
    if (rc)
    {
        linalg_aligned_free(tau);
        return rc;
    }

    if (!only_R)
    {
        rc = qrw_orgqr_full(Q, m, R, n, tau, mn);
        if (rc)
        {
            linalg_aligned_free(tau);
            return rc;
        }
    }

    linalg_aligned_free(tau);
    return 0;
}
