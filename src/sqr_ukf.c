// --- drop-in replacements for: create_weights, create_sigma_point_matrix, compute_transistion_function

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "linalg_simd.h"

/* define to 1 if you want the 8-way batching; keep 0 by default */
#ifndef SQR_UKF_ENABLE_BATCH8
#define SQR_UKF_ENABLE_BATCH8 0
#endif

#ifndef UKF_PREFETCH_ROWS_AHEAD
#define UKF_PREFETCH_ROWS_AHEAD 1 /* 0..2 are sensible */
#endif
#ifndef UKF_PREFETCH_DIST_BYTES
#define UKF_PREFETCH_DIST_BYTES 128 /* 64 or 128 */
#endif
#ifndef UKF_PREFETCH_MIN_L
#define UKF_PREFETCH_MIN_L 128
#endif

#ifndef UKF_TRANS_PF_MIN_L
#define UKF_TRANS_PF_MIN_L 128 /* enable prefetch when L >= this */
#endif
#ifndef UKF_TRANS_PF_ROWS_AHEAD
#define UKF_TRANS_PF_ROWS_AHEAD 1 /* 0..2 sensible; 1 is safe default */
#endif

#ifndef UKF_MEAN_PF_MIN_ROWS
#define UKF_MEAN_PF_MIN_ROWS 128 /* enable row-ahead prefetch when L >= this */
#endif
#ifndef UKF_MEAN_PF_ROWS_AHEAD
#define UKF_MEAN_PF_ROWS_AHEAD 1 /* 0..2 are reasonable */
#endif

#ifndef UKF_APRIME_PF_MIN_L
#define UKF_APRIME_PF_MIN_L 128 /* enable prefetch when L >= this */
#endif
#ifndef UKF_APRIME_PF_ROWS_AHEAD
#define UKF_APRIME_PF_ROWS_AHEAD 1 /* 0..2 are sensible */
#endif
#ifndef UKF_APRIME_PF_DIST_BYTES
#define UKF_APRIME_PF_DIST_BYTES 128 /* cache-line distance (64 or 128) */
#endif

#ifndef UKF_PXY_PF_MIN_N
#define UKF_PXY_PF_MIN_N 256 /* enable prefetch when N >= this */
#endif
#ifndef UKF_PXY_PF_ROWS_AHEAD
#define UKF_PXY_PF_ROWS_AHEAD 1 /* prefetch this many future Y rows (0..2 sensible) */
#endif
#ifndef UKF_PXY_PF_DIST_BYTES
#define UKF_PXY_PF_DIST_BYTES 128 /* stream prefetch distance within a row: 64 or 128 */
#endif

#ifndef UKF_UPD_COLBLOCK
#define UKF_UPD_COLBLOCK 64 /* RHS column block for triangular solves */
#endif
#ifndef UKF_UPD_PF_MIN_N
#define UKF_UPD_PF_MIN_N 128 /* enable prefetch in solves when n >= this */
#endif
#ifndef UKF_UPD_PF_DIST_BYTES
#define UKF_UPD_PF_DIST_BYTES 128 /* prefetch distance along RHS rows */
#endif

#ifndef UKF_PXY_PF_MIN_L
#define UKF_PXY_PF_MIN_L 16 /* enable row-ahead prefetch when L >= this */
#endif
#ifndef UKF_PXY_PF_MIN_N
#define UKF_PXY_PF_MIN_N 32 /* enable within-row prefetch when N >= this */
#endif
#ifndef UKF_PXY_PF_ROWS_AHEAD
#define UKF_PXY_PF_ROWS_AHEAD 1 /* 0..2 sensible */
#endif
#ifndef UKF_PXY_PF_DIST_BYTES
#define UKF_PXY_PF_DIST_BYTES 128 /* 64 or 128 are typical */
#endif

static inline void *ukf_aligned_alloc(size_t nbytes)
{
    return linalg_aligned_alloc(32, nbytes);
}

static inline void ukf_aligned_free(void *p)
{
    linalg_aligned_free(p);
}

/* =================== Reusable workspace for SR-UKF QR step =================== */
typedef struct
{
    float *Aprime; /* (M x L) row-major, M = 3L */
    float *R_;     /* (M x L) row-major */
    float *b;      /* (L) */
    size_t capL;   /* capacity in L */
} ukf_qr_ws_t;

typedef struct
{
    float *Z;     /* n x n, reused as K after backward solve */
    float *Ky;    /* n */
    float *U;     /* n x n */
    float *Uk;    /* n */
    float *yyhat; /* n */
    size_t cap;   /* in elements, for n*n buffers */
} ukf_upd_ws_t;

static inline int ukf_qr_ws_ensure(ukf_qr_ws_t *ws, size_t L)
{
    const size_t M = 3u * L;
    const size_t need_A = M * L;
    const size_t need_R = M * L;
    const size_t need_b = L;

    if (ws->capL >= L && ws->Aprime && ws->R_ && ws->b)
        return 0;

    if (ws->Aprime)
        ukf_aligned_free(ws->Aprime);
    if (ws->R_)
        ukf_aligned_free(ws->R_);
    if (ws->b)
        ukf_aligned_free(ws->b);

    ws->Aprime = (float *)ukf_aligned_alloc(need_A * sizeof(float));
    ws->R_ = (float *)ukf_aligned_alloc(need_R * sizeof(float));
    ws->b = (float *)ukf_aligned_alloc(need_b * sizeof(float));
    ws->capL = (ws->Aprime && ws->R_ && ws->b) ? L : 0;

    return (ws->capL ? 0 : -ENOMEM);
}

static inline void ukf_qr_ws_free(ukf_qr_ws_t *ws)
{
    if (!ws)
        return;
    if (ws->Aprime)
        ukf_aligned_free(ws->Aprime);
    if (ws->R_)
        ukf_aligned_free(ws->R_);
    if (ws->b)
        ukf_aligned_free(ws->b);
    ws->Aprime = ws->R_ = ws->b = NULL;
    ws->capL = 0;
}

static inline int ukf_upd_ws_ensure(ukf_upd_ws_t *ws, uint16_t n)
{
    const size_t nn = (size_t)n * (size_t)n;
    const size_t need = nn; /* for Z and U we each need nn; track capacity by n (symmetric growth) */

    if (ws->cap >= nn && ws->Z && ws->U && ws->Uk && ws->Ky && ws->yyhat)
        return 0;

    if (ws->Z)
        linalg_aligned_free(ws->Z);
    if (ws->U)
        linalg_aligned_free(ws->U);
    if (ws->Uk)
        linalg_aligned_free(ws->Uk);
    if (ws->Ky)
        linalg_aligned_free(ws->Ky);
    if (ws->yyhat)
        linalg_aligned_free(ws->yyhat);

    ws->Z = (float *)linalg_aligned_alloc(32, nn * sizeof(float));
    ws->U = (float *)linalg_aligned_alloc(32, nn * sizeof(float));
    ws->Uk = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->Ky = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->yyhat = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    ws->cap = (ws->Z && ws->U && ws->Uk && ws->Ky && ws->yyhat) ? nn : 0;

    return ws->cap ? 0 : -ENOMEM;
}

static inline void ukf_upd_ws_free(ukf_upd_ws_t *ws)
{
    if (!ws)
        return;
    if (ws->Z)
        linalg_aligned_free(ws->Z);
    if (ws->U)
        linalg_aligned_free(ws->U);
    if (ws->Uk)
        linalg_aligned_free(ws->Uk);
    if (ws->Ky)
        linalg_aligned_free(ws->Ky);
    if (ws->yyhat)
        linalg_aligned_free(ws->yyhat);
    ws->Z = ws->U = ws->Uk = ws->Ky = ws->yyhat = NULL;
    ws->cap = 0;
}

/**
 * @brief Compute Unscented weights for mean (Wm) and covariance (Wc).
 *
 * @details
 *  Builds the standard UKF weights from {alpha,beta,kappa,L}.
 *  Vectorized version uses AVX2 to bulk-fill the constant tail (i>=1) in
 *  8-wide chunks to reduce loop overhead and store traffic.
 *
 *  Improvements over scalar:
 *   - Bulk tail initialization with AVX2 stores (8x per iteration).
 *   - Avoids redundant zeroing; fully writes outputs.
 *
 * @param[out] Wc   Covariance weights, length N=2L+1.
 * @param[out] Wm   Mean weights, length N=2L+1.
 * @param[in]  alpha  UKF spread parameter.
 * @param[in]  beta   Prior knowledge of distribution (e.g., beta=2 for Gaussian).
 * @param[in]  kappa  Secondary scaling parameter.
 * @param[in]  L      State dimension.
 *
 * @note Falls back to scalar if AVX2/FMA is not available or N<9.
 */
static void create_weights(float Wc[], float Wm[],
                           float alpha, float beta, float kappa, uint8_t L)
{
    const size_t N = (size_t)(2u * L + 1u);
    const float Lf = (float)L;
    const float lam = alpha * alpha * (Lf + kappa) - Lf;
    const float den = 1.0f / (Lf + lam);

    /* first element */
    Wm[0] = lam * den;
    Wc[0] = Wm[0] + 1.0f - alpha * alpha + beta;

    /* bulk tail (i >= 1): 0.5 / (L + lambda) */
    const float hv = 0.5f * den;

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 9)
    {
        const __m256 v = _mm256_set1_ps(hv);
        size_t i = 1;
        for (; i + 7 < N; i += 8)
        {
            _mm256_storeu_ps(&Wm[i], v);
            _mm256_storeu_ps(&Wc[i], v);
        }
        for (; i < N; ++i)
        {
            Wm[i] = hv;
            Wc[i] = hv;
        }
        return;
    }
#endif

    for (size_t i = 1; i < N; ++i)
    {
        Wm[i] = hv;
        Wc[i] = hv;
    }
}

/**
 * @brief Build sigma point matrix X from state x and Cholesky factor S.
 *
 * @details
 *  Fills X(:,0)=x, X(:,1..L)=x+gamma*S(:,j), X(:,L+1..2L)=x-gamma*S(:,j),
 *  row-major with stride N. Vectorized path:
 *   - Column 0 copied with AVX2 loads/stores.
 *   - For columns, uses AVX2 gathers to read S down columns (row-major stride L),
 *     and FMA with ±gamma to form each sigma column.
 *
 *  Improvements over scalar:
 *   - 8-wide gathers + FMA reduce scalar address arithmetic and mul/add pairs.
 *   - Clean scalar tails for non-multiples of 8.
 *
 * @param[out] X     Sigma point matrix [L x N], row-major.
 * @param[in]  x     State vector [L].
 * @param[in]  S     Cholesky factor of covariance [L x L], row-major.
 * @param[in]  alpha,kappa  UKF scaling parameters.
 * @param[in]  L     State dimension.
 *
 * @note N=2L+1. Falls back to scalar when AVX2 unavailable or L<8.
 * @warning X must not alias x or S.
 */
static void create_sigma_point_matrix(float X[], const float x[], const float S[],
                                      float alpha, float kappa, uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const float gamma = alpha * sqrtf((float)L + kappa);

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && L >= 8)
    {
        const __m256 g = _mm256_set1_ps(gamma);
        const __m256 ng = _mm256_set1_ps(-gamma);
        const size_t pf_elts = UKF_PREFETCH_DIST_BYTES / sizeof(float);
        const int do_pf = (L >= (size_t)UKF_PREFETCH_MIN_L);
        const int rows_ahead = UKF_PREFETCH_ROWS_AHEAD;

        for (size_t i = 0; i < L; ++i)
        {
            float *Xi = X + i * N;       // row i of X
            const float *Si = S + i * L; // row i of S
            const __m256 xi8 = _mm256_set1_ps(x[i]);

            Xi[0] = x[i];

            // Optional row-ahead prefetch (S and X of next row(s))
            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    if (i + (size_t)ra < L)
                    {
                        const float *Spi = S + (i + (size_t)ra) * L;
                        float *Xpi = X + (i + (size_t)ra) * N;
                        _mm_prefetch((const char *)Spi, _MM_HINT_T0);
                        _mm_prefetch((const char *)Xpi, _MM_HINT_T0);
                    }
                }
            }

            // Alignment fast path (works if both Si and Xi are 32B-aligned)
            const int aligned = ((((uintptr_t)Si | (uintptr_t)(Xi + 1) | (uintptr_t)(Xi + 1 + L)) & 31) == 0);

            size_t j = 0;
            if (aligned)
            {
                for (; j + 7 < L; j += 8)
                {
                    if (do_pf)
                    {
                        _mm_prefetch((const char *)(Si + j + pf_elts), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Xi + 1 + j + pf_elts), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Xi + 1 + L + j + pf_elts), _MM_HINT_T0);
                    }
                    __m256 s8 = _mm256_load_ps(Si + j);
                    __m256 plus = _mm256_fmadd_ps(g, s8, xi8);
                    __m256 minus = _mm256_fmadd_ps(ng, s8, xi8);
                    _mm256_store_ps(Xi + 1 + j, plus);
                    _mm256_store_ps(Xi + 1 + L + j, minus);
                }
            }
            else
            {
                for (; j + 7 < L; j += 8)
                {
                    if (do_pf)
                    {
                        _mm_prefetch((const char *)(Si + j + pf_elts), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Xi + 1 + j + pf_elts), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Xi + 1 + L + j + pf_elts), _MM_HINT_T0);
                    }
                    __m256 s8 = _mm256_loadu_ps(Si + j);
                    __m256 plus = _mm256_fmadd_ps(g, s8, xi8);
                    __m256 minus = _mm256_fmadd_ps(ng, s8, xi8);
                    _mm256_storeu_ps(Xi + 1 + j, plus);
                    _mm256_storeu_ps(Xi + 1 + L + j, minus);
                }
            }

            // scalar tail
            for (; j < L; ++j)
            {
                const float s = Si[j];
                Xi[1 + j] = x[i] + gamma * s;
                Xi[1 + L + j] = x[i] - gamma * s;
            }
        }
        return;
    }
#endif

    // Scalar fallback (contiguous)
    for (size_t i = 0; i < L; ++i)
    {
        float *Xi = X + i * N;
        const float *Si = S + i * L;
        Xi[0] = x[i];
        for (size_t j = 0; j < L; ++j)
        {
            const float s = Si[j];
            Xi[1 + j] = x[i] + gamma * s;
            Xi[1 + L + j] = x[i] - gamma * s;
        }
    }
}

/**
 * @brief Apply transition function F to all sigma points: X*[:, j] = F(X[:, j], u).
 *
 * @details
 *  Vectorized "batch-8" path packs 8 sigma columns into 8 contiguous L-length
 *  slices (SoA: k-major) so you can call F() on contiguous inputs:
 *      x[k*L + i] = X[i*N + (j+k)],  d[k*L + i] = F(x[k*L + :], u)[i]
 *  AVX2 is used only to load/store the 8 contiguous sigma entries per row i;
 *  since AVX2 lacks float scatters, we store the 8 lanes into a tiny stack
 *  buffer and perform 8 scalar lane stores to the SoA buffer.
 *
 *  Notes:
 *   - F typically dominates runtime; this path mainly reduces address
 *     arithmetic and improves cache behavior when L is large and N is big.
 *   - Falls back to scalar if allocation fails or N < 8.
 */
static void compute_transistion_function(float Xstar[], const float X[], const float u[],
                                         void (*F)(float[], float[], float[]), uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;

#if SQR_UKF_ENABLE_BATCH8
    if (ukf_has_avx2() && N >= 8)
    {
        /* SoA buffers: 8 states of length L each (k-major). */
        float *x = (float *)ukf_aligned_alloc((size_t)8 * L * sizeof(float));
        float *d = (float *)ukf_aligned_alloc((size_t)8 * L * sizeof(float));
        if (x && d)
        {
            const int do_pf = (L >= (size_t)UKF_TRANS_PF_MIN_L);
            const int rows_ahead = UKF_TRANS_PF_ROWS_AHEAD;

            /* process in batches of 8 sigmas */
            size_t j = 0;
            for (; j + 7 < N; j += 8)
            {

                /* pack 8 columns (j..j+7) into SoA */
                for (size_t i = 0; i < L; ++i)
                {
                    /* prefetch next row(s) of the same 8-sigma stripe */
                    if (do_pf && rows_ahead > 0)
                    {
                        for (int ra = 1; ra <= rows_ahead; ++ra)
                        {
                            const size_t ip = i + (size_t)ra;
                            if (ip < L)
                            {
                                _mm_prefetch((const char *)(&X[ip * N + j]), _MM_HINT_T0);
                                _mm_prefetch((const char *)(&Xstar[ip * N + j]), _MM_HINT_T0);
                            }
                        }
                    }

                    /* load 8 contiguous sigmas from row i */
                    __m256 v = _mm256_loadu_ps(&X[i * N + j]);
                    /* lane buffer then scalar-lane scatter into SoA x[k*L + i] */
                    alignas(32) float lanes[8];
                    _mm256_store_ps(lanes, v);
#pragma GCC ivdep
                    for (int k = 0; k < 8; ++k)
                        x[(size_t)k * L + i] = lanes[k];
                }

                /* evaluate F on each contiguous L-vector */
                for (int k = 0; k < 8; ++k)
                    F(&d[(size_t)k * L], &x[(size_t)k * L], (float *)u);

                /* unpack back into 8 columns (j..j+7) */
                for (size_t i = 0; i < L; ++i)
                {
#pragma GCC ivdep
                    for (int k = 0; k < 8; ++k)
                        Xstar[i * N + (j + (size_t)k)] = d[(size_t)k * L + i];
                }
            }

            /* scalar tail for remaining sigmas */
            for (; j < N; ++j)
            {
                float *xk = x;
                float *dk = d;
                for (size_t i = 0; i < L; ++i)
                    xk[i] = X[i * N + j];
                F(dk, xk, (float *)u);
                for (size_t i = 0; i < L; ++i)
                    Xstar[i * N + j] = dk[i];
            }

            ukf_aligned_free(x);
            ukf_aligned_free(d);
            return;
        }
        if (x)
            ukf_aligned_free(x);
        if (d)
            ukf_aligned_free(d);
    }
#endif

    /* scalar fallback */
    float *xk = (float *)malloc(L * sizeof(float));
    float *dk = (float *)malloc(L * sizeof(float));
    if (!xk || !dk)
    {
        free(xk);
        free(dk);
        return;
    }

    for (size_t j = 0; j < N; ++j)
    {
        for (size_t i = 0; i < L; ++i)
            xk[i] = X[i * N + j];
        F(dk, xk, (float *)u);
        for (size_t i = 0; i < L; ++i)
            Xstar[i * N + j] = dk[i];
    }

    free(xk);
    free(dk);
}

/**
 * @brief Compute weighted mean x = X * W for sigma points.
 *
 * @details
 *  Computes x[i] = sum_j W[j] * X[i,j], i = 0..L-1.
 *  AVX2 fast path processes two rows at a time and reuses each 8-wide W chunk
 *  for both rows, reducing memory traffic and improving throughput.
 *  Falls back to a single-row AVX2 kernel for odd L, and to scalar when N<8 or AVX2 off.
 *
 *  Why 2-row: reusing W halves its load bandwidth and provides ILP without
 *  pushing AVX2 register pressure (fits under 16 YMM regs comfortably).
 *
 * @param[out] x   Output mean vector [L].
 * @param[in]  X   Sigma matrix [L x N], row-major.
 * @param[in]  W   Weights [N].
 * @param[in]  L   State dimension (N=2L+1).
 */
static void multiply_sigma_point_matrix_to_weights(float x[], float X[], float W[], uint8_t L)
{
    const size_t Ls = (size_t)L;
    const size_t N = 2u * Ls + 1u;

    if (!ukf_has_avx2() || N < 8)
    {
        /* scalar fallback */
        for (size_t i = 0; i < Ls; ++i)
        {
            const float *row = &X[i * N];
            float acc = 0.0f;
            for (size_t j = 0; j < N; ++j)
                acc += W[j] * row[j];
            x[i] = acc;
        }
        return;
    }

#if LINALG_SIMD_ENABLE
    /* ---- AVX2 path ---- */

    /* Optional row-ahead prefetching for big L */
    const int do_pf = (Ls >= (size_t)UKF_MEAN_PF_MIN_ROWS);
    const int rows_ahead = UKF_MEAN_PF_ROWS_AHEAD;

    /* Process rows in pairs: i and i+1 */
    size_t i = 0;
    for (; i + 1 < Ls; i += 2)
    {
        const float *row0 = &X[(i + 0) * N];
        const float *row1 = &X[(i + 1) * N];

        /* Prefetch upcoming rows to warm caches (for the same pattern of W) */
        if (do_pf && rows_ahead > 0)
        {
            for (int ra = 1; ra <= rows_ahead; ++ra)
            {
                const size_t ip = i + (size_t)ra * 2; /* prefetch pairs ahead */
                if (ip < Ls)
                {
                    _mm_prefetch((const char *)(&X[ip * N]), _MM_HINT_T0);
                    if (ip + 1 < Ls)
                        _mm_prefetch((const char *)(&X[(ip + 1) * N]), _MM_HINT_T0);
                }
            }
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 7 < N; j += 8)
        {
            __m256 wv = _mm256_loadu_ps(&W[j]); /* load W once */
            __m256 x0 = _mm256_loadu_ps(&row0[j]);
            __m256 x1 = _mm256_loadu_ps(&row1[j]);
            acc0 = _mm256_fmadd_ps(wv, x0, acc0); /* dot row0 */
            acc1 = _mm256_fmadd_ps(wv, x1, acc1); /* dot row1 */
        }

        /* horizontal sums */
        __m128 lo0 = _mm256_castps256_ps128(acc0);
        __m128 hi0 = _mm256_extractf128_ps(acc0, 1);
        __m128 s0 = _mm_add_ps(lo0, hi0);
        s0 = _mm_hadd_ps(s0, s0);
        s0 = _mm_hadd_ps(s0, s0);
        float sum0 = _mm_cvtss_f32(s0);

        __m128 lo1 = _mm256_castps256_ps128(acc1);
        __m128 hi1 = _mm256_extractf128_ps(acc1, 1);
        __m128 s1 = _mm_add_ps(lo1, hi1);
        s1 = _mm_hadd_ps(s1, s1);
        s1 = _mm_hadd_ps(s1, s1);
        float sum1 = _mm_cvtss_f32(s1);

        /* scalar tails */
        for (; j < N; ++j)
        {
            sum0 += W[j] * row0[j];
            sum1 += W[j] * row1[j];
        }

        x[i + 0] = sum0;
        x[i + 1] = sum1;
    }

    /* leftover last row if L is odd */
    if (i < Ls)
    {
        const float *row = &X[i * N];
        __m256 acc = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 7 < N; j += 8)
        {
            __m256 wv = _mm256_loadu_ps(&W[j]);
            __m256 xv = _mm256_loadu_ps(&row[j]);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; j < N; ++j)
            sum += W[j] * row[j];
        x[i] = sum;
    }
#else
    /* Should not reach; guarded above. Keep scalar for safety. */
    for (size_t i2 = 0; i2 < Ls; ++i2)
    {
        const float *row = &X[i2 * N];
        float acc = 0.0f;
        for (size_t j = 0; j < N; ++j)
            acc += W[j] * row[j];
        x[i2] = acc;
    }
#endif
}

/**
 * @brief Build square-root state covariance S (SR-UKF) from weighted sigma deviations and process noise.
 *
 * @details
 *  Forms A' per SR-UKF: columns from weighted deviations and sqrt of R, then computes QR(A')
 *  and copies the upper LxL of R_ into S. Applies a rank-1 Cholesky update/downdate with
 *  (X(:,0)-x) based on W[0] sign.
 *
 *  Vectorized pieces:
 *   - Construct A (then transpose) with AVX2: X−x subtraction and scaling in 8-wide chunks.
 *   - Vectorized sqrt(R) fill of the right block.
 *   - Vectorized build of b = X(:,0) − x.
 *
 *  Improvements over scalar:
 *   - Fewer loops and address arithmetic via wide loads/stores.
 *   - Avoids forming temporary identities; only needed blocks are built.
 *   - Uses optimized tran(), qr(), and cholupdate().
 *
 * @param[out] S   Square-root covariance [L x L], upper-triangular on output.
 * @param[in]  W   Covariance weights [N].
 * @param[in]  X   Propagated sigma points [L x N], row-major.
 * @param[in]  x   Predicted state mean [L].
 * @param[in]  R   Process/measurement noise (diagonal or full) [L x L].
 * @param[in]  L   Dimension.
 *
 * @warning S, X, R must not alias each other. Uses row-major layout.
 */
static void create_state_estimation_error_covariance_matrix(float S[], float W[], float X[],
                                                            float x[], float R[], uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const size_t K = 2u * L;
    const size_t M = 3u * L;

    const float w1s = sqrtf(fabsf(W[1])); /* for columns 1..K */
    const float w0s = sqrtf(fabsf(W[0])); /* for mean deviation (b) */

    /* Reusable workspace (TLS if available) */
#if defined(__GNUC__) || defined(__clang__)
    static __thread ukf_qr_ws_t ws = {0};
#else
    static ukf_qr_ws_t ws = {0}; /* if multi-threaded, convert to TLS on your platform */
#endif
    if (ukf_qr_ws_ensure(&ws, L) != 0)
        return;

    float *Aprime = ws.Aprime; /* (M x L) */
    float *R_ = ws.R_;         /* (M x L) */
    float *b = ws.b;           /* (L) */

    /* Prefetch control */
    const int do_pf = (int)(L >= (size_t)UKF_APRIME_PF_MIN_L);
    const int rows_ahead = UKF_APRIME_PF_ROWS_AHEAD;
    const size_t pf_elems = (size_t)UKF_APRIME_PF_DIST_BYTES / sizeof(float);

    /* -------------------- Build A′ and b (column i at a time) -------------------- */
    if (ukf_has_avx2() && L >= 8)
    {
        const __m256 w1v = _mm256_set1_ps(w1s);

        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            const float xi = x[i];

            /* row-ahead prefetch for X, R, and Aprime destination stripes (future column i+ra) */
            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X + ip * N), _MM_HINT_T0);
                        _mm_prefetch((const char *)(R + ip * L), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Aprime + ip), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Aprime + (K * L) + ip), _MM_HINT_T0);
                    }
                }
            }

            /* b[i] = sqrt(|W0|) * (X[i,0] - x[i])  (contiguous; no gather) */
            b[i] = w0s * (Xi[0] - xi);

            /* Rows 0..K-1: deviations block. Vector math on Xi[r+1..r+8]; strided store via lane buffer. */
            size_t r = 0;
            const __m256 xi8 = _mm256_set1_ps(xi);
            for (; r + 7 < K; r += 8)
            {
                if (do_pf && r + pf_elems + 8 < K)
                    _mm_prefetch((const char *)(&Xi[r + 1 + pf_elems]), _MM_HINT_T0);

                __m256 Xv = _mm256_loadu_ps(&Xi[r + 1]);
                __m256 diff = _mm256_sub_ps(Xv, xi8);
                __m256 out = _mm256_mul_ps(w1v, diff);

                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, out);

                if (do_pf && r + pf_elems + 8 < K)
                    _mm_prefetch((const char *)(Aprime + (r + pf_elems) * L + i), _MM_HINT_T0);

#pragma GCC ivdep
                for (int k2 = 0; k2 < 8; ++k2)
                    Aprime[(r + (size_t)k2) * L + i] = lanes[k2];
            }
            for (; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);

            /* Rows K..M-1: sqrt(R[i, :]) — contiguous loads in R row, strided stores in Aprime */
            size_t t = 0;
            for (; t + 7 < L; t += 8)
            {
                if (do_pf && t + pf_elems + 8 < L)
                {
                    _mm_prefetch((const char *)(&R[i * L + t + pf_elems]), _MM_HINT_T0);
                    _mm_prefetch((const char *)(Aprime + (K + t + pf_elems) * L + i), _MM_HINT_T0);
                }

                __m256 Rv = _mm256_loadu_ps(&R[i * L + t]);
                __m256 Sv = _mm256_sqrt_ps(Rv);

                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, Sv);
#pragma GCC ivdep
                for (int k2 = 0; k2 < 8; ++k2)
                    Aprime[(K + t + (size_t)k2) * L + i] = lanes[k2];
            }
            for (; t < L; ++t)
                Aprime[(K + t) * L + i] = sqrtf(R[i * L + t]);
        }
    }
    else
    {
        /* Scalar build (direct A′, with light row-ahead prefetch for large L) */
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            const float xi = x[i];

            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X + ip * N), _MM_HINT_T0);
                        _mm_prefetch((const char *)(R + ip * L), _MM_HINT_T0);
                    }
                }
            }

            b[i] = w0s * (Xi[0] - xi);

            for (size_t r = 0; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);
            for (size_t t = 0; t < L; ++t)
                Aprime[(K + t) * L + i] = sqrtf(R[i * L + t]);
        }
    }

    /* -------------------- QR of A′ (M x L); only R needed -------------------- */
    /* Ensure qr() fully skips Q work when only_R=true and Q==NULL. */
    if (qr(Aprime, /*Q=*/NULL, R_, (uint16_t)M, (uint16_t)L, /*only_R=*/true) != 0)
    {
        return; /* optionally signal error */
    }

    /* S = upper LxL block of R_ */
    memcpy(S, R_, (size_t)L * L * sizeof(float));

    /* Rank-one update/downdate with sign(W0) — matches signature exactly */
    cholupdate(/*L=*/S, /*xx=*/(const float *)b, /*n=*/(uint16_t)L, /*rank_one_update=*/(W[0] >= 0.0f));

    bool pd_ok = true;
    for (size_t i = 0; i < L; ++i)
        if (!(S[i * L + i] > 0.0f && isfinite(S[i * L + i])))
        {
            pd_ok = false;
            break;
        }

    if (!pd_ok)
    {
        /* Handle error: e.g., reinitialize S or skip update */
    }
}

/**
 * @brief Identity observation model: Y = X.
 *
 * @details
 *  Copies Y := X for the sigma matrix. Implemented as a single memcpy since
 *  row-major layouts are identical.
 *
 * @param[out] Y  Observation sigma matrix [L x N], row-major.
 * @param[in]  X  State sigma matrix [L x N], row-major.
 * @param[in]  L  Dimension (N=2L+1).
 */
static void H(float Y[], float X[], uint8_t L)
{
    const uint16_t N = (uint16_t)(2 * L + 1);
    memcpy(Y, X, (size_t)L * N * sizeof(float));
}

static inline size_t ukf_round_up8(size_t n) { return (n + 7u) & ~7u; }

#if LINALG_SIMD_ENABLE
static inline void ukf_transpose8x8_ps(__m256 in[8], __m256 out[8])
{
    __m256 t0 = _mm256_unpacklo_ps(in[0], in[1]);
    __m256 t1 = _mm256_unpackhi_ps(in[0], in[1]);
    __m256 t2 = _mm256_unpacklo_ps(in[2], in[3]);
    __m256 t3 = _mm256_unpackhi_ps(in[2], in[3]);
    __m256 t4 = _mm256_unpacklo_ps(in[4], in[5]);
    __m256 t5 = _mm256_unpackhi_ps(in[4], in[5]);
    __m256 t6 = _mm256_unpacklo_ps(in[6], in[7]);
    __m256 t7 = _mm256_unpackhi_ps(in[6], in[7]);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x4E);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xB1);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x4E);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xB1);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x4E);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xB1);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x4E);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xB1);

    out[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    out[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    out[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    out[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    out[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    out[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    out[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    out[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}
#endif

/**
 * @brief Cross-covariance Pxy = X_c * diag(W) * Y_c^T without explicit diag matrix.
 *
 * @details
 *  Centers X and Y row-wise by subtracting x and y, then accumulates
 *  P = sum_j W[j] * X[:,j] * Y[:,j]^T using an 8x8 AVX2 outer-product micro-kernel
 *  with scalar tails.
 *
 *  Improvements over scalar/baseline:
 *   - Avoids building a dense diag(W) and two GEMMs.
 *   - FMA-heavy rank-1 accumulation in cache-friendly tiles.
 *   - Fixes the original memset bug (now zeros L*L entries, not 2L).
 *
 * @param[out] P   Cross-covariance [L x L], row-major.
 * @param[in]  W   Weights [N].
 * @param[in]  X   Sigma matrix [L x N], row-major (overwritten in-place during centering).
 * @param[in]  Y   Sigma matrix [L x N], row-major (overwritten in-place during centering).
 * @param[in]  x   Mean of X [L].
 * @param[in]  y   Mean of Y [L].
 * @param[in]  L   Dimension.
 *
 * @warning X and Y are centered in-place (destructive). If needed elsewhere, pass copies.
 */
static void create_state_cross_covariance_matrix(float *RESTRICT P, float *RESTRICT W,
                                                 const float *RESTRICT X, const float *RESTRICT Y,
                                                 const float *RESTRICT x, const float *RESTRICT y,
                                                 uint8_t L8)
{
    const size_t L = (size_t)L8;
    const size_t N = 2u * L + 1u;
    const size_t N8 = ukf_round_up8(N);

    memset(P, 0, L * L * sizeof(float));

    float *Xc = (float *)linalg_aligned_alloc(32, L * N8 * sizeof(float));
    float *YTc = (float *)linalg_aligned_alloc(32, N8 * L * sizeof(float));
    if (!Xc || !YTc)
    {
        if (Xc)
            linalg_aligned_free(Xc);
        if (YTc)
            linalg_aligned_free(YTc);
        return;
    }

    /* Prefetch policy */
    const int do_pf_rows = (int)(L >= (size_t)UKF_PXY_PF_MIN_L);
    const int do_pf_in = (int)(N >= (size_t)UKF_PXY_PF_MIN_N);
    const int rows_ahead = UKF_PXY_PF_ROWS_AHEAD;
    const size_t pf_elems = (size_t)UKF_PXY_PF_DIST_BYTES / sizeof(float);

    /* ---------------- Xc build: (X - x) ⊙ W, pad to N8 ---------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 8)
    {
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            float *Xci = Xc + i * N8;
            const __m256 xi8 = _mm256_set1_ps(x[i]);

            /* row-ahead prefetch for next X rows and their Xc destinations */
            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X + ip * N), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Xc + ip * N8), _MM_HINT_T0);
                    }
                }
            }

            size_t j = 0;
            for (; j + 7 < N; j += 8)
            {
                if (do_pf_in && j + pf_elems + 8 < N)
                {
                    _mm_prefetch((const char *)(Xi + j + pf_elems), _MM_HINT_T0);
                    _mm_prefetch((const char *)(W + j + pf_elems), _MM_HINT_T0);
                }
                __m256 xv = _mm256_loadu_ps(Xi + j);
                __m256 wv = _mm256_loadu_ps(W + j);
                __m256 diff = _mm256_sub_ps(xv, xi8);
                _mm256_storeu_ps(Xci + j, _mm256_mul_ps(diff, wv));
            }
            for (; j < N; ++j)
                Xci[j] = (Xi[j] - x[i]) * W[j];
            for (; j < N8; ++j)
                Xci[j] = 0.0f;
        }
    }
    else
#endif
    {
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            float *Xci = Xc + i * N8;

            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    size_t ip = i + (size_t)ra;
                    if (ip < L)
                        _mm_prefetch((const char *)(X + ip * N), _MM_HINT_T0);
                }
            }

            size_t j = 0;
            for (; j < N; ++j)
            {
                if (do_pf_in && j + pf_elems + 1 < N)
                {
                    _mm_prefetch((const char *)(Xi + j + pf_elems), _MM_HINT_T0);
                    _mm_prefetch((const char *)(W + j + pf_elems), _MM_HINT_T0);
                }
                Xci[j] = (Xi[j] - x[i]) * W[j];
            }
            for (; j < N8; ++j)
                Xci[j] = 0.0f;
        }
    }

    /* ---------------- YTc build: centered Y^T (N8×L) with AVX2 8×8 transpose ---------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 8 && L >= 8)
    {
        /* zero pad rows j=N..N8-1 up-front (contiguous) */
        for (size_t jp = N; jp < N8; ++jp)
            memset(YTc + jp * L, 0, L * sizeof(float));

        size_t k0 = 0;
        for (; k0 + 7 < L; k0 += 8)
        {
            /* prefetch upcoming Y rows */
            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    size_t kp = k0 + (size_t)ra * 8;
                    if (kp < L)
                        _mm_prefetch((const char *)(Y + kp * N), _MM_HINT_T0);
                }
            }

            size_t j0 = 0;
            for (; j0 + 7 < N; j0 += 8)
            {
                /* within-row stream prefetch */
                if (do_pf_in && j0 + pf_elems + 8 < N)
                {
                    for (int r = 0; r < 8; ++r)
                        _mm_prefetch((const char *)(Y + (k0 + (size_t)r) * N + j0 + pf_elems), _MM_HINT_T0);
                    _mm_prefetch((const char *)(YTc + (j0 + pf_elems) * L + k0), _MM_HINT_T0);
                }

                __m256 row[8];
                for (int r = 0; r < 8; ++r)
                {
                    const float *Yr = Y + (k0 + (size_t)r) * N + j0;
                    __m256 yr = _mm256_set1_ps(y[k0 + (size_t)r]);
                    row[r] = _mm256_sub_ps(_mm256_loadu_ps(Yr), yr);
                }
                __m256 col[8];
                ukf_transpose8x8_ps(row, col);

                for (int c = 0; c < 8; ++c)
                    _mm256_storeu_ps(YTc + (j0 + (size_t)c) * L + k0, col[c]);
            }
            /* N-tail for these 8 rows */
            for (; j0 < N; ++j0)
            {
                float *YTrow = YTc + j0 * L;
                for (int r = 0; r < 8; ++r)
                    YTrow[k0 + (size_t)r] = Y[(k0 + (size_t)r) * N + j0] - y[k0 + (size_t)r];
            }
        }
        /* L-tail rows */
        for (; k0 < L; ++k0)
        {
            size_t j = 0;
            for (; j < N; ++j)
                YTc[j * L + k0] = Y[k0 * N + j] - y[k0];
            for (; j < N8; ++j)
                YTc[j * L + k0] = 0.0f;
        }
    }
    else
#endif
    {
        for (size_t j = 0; j < N; ++j)
        {
            const float *Ycol0 = Y + j; /* Y[k*N + j] */
            float *YTrow = YTc + j * L;
            for (size_t k = 0; k < L; ++k)
                YTrow[k] = Ycol0[k * N] - y[k];
        }
        for (size_t j = N; j < N8; ++j)
            memset(YTc + j * L, 0, L * sizeof(float));
    }

    /* ---------------- GEMM: P = Xc * YTc ----------------
       Optional light prefetch hint for mul()’s first panels. */
    if (do_pf_rows)
    {
        _mm_prefetch((const char *)Xc, _MM_HINT_T0);
        _mm_prefetch((const char *)YTc, _MM_HINT_T0);
        _mm_prefetch((const char *)P, _MM_HINT_T0);
    }
    mul(P, Xc, YTc, (uint16_t)L, (uint16_t)N8, (uint16_t)N8, (uint16_t)L);

    linalg_aligned_free(Xc);
    linalg_aligned_free(YTc);
}

/**
 * @brief Measurement update: compute K, update xhat, and downdate S via Cholesky rank-1.
 *
 * @details
 *  Solves (Sy^T Sy)K = Pxy without explicit inverse:
 *   1) Forward  (lower):  Sy^T Z = Pxy.
 *   2) Backward (upper):  Sy   K = Z  (done in-place: Z becomes K).
 *  Then Ky = K*(y−yhat), xhat += Ky, U = K*Sy, and S ← cholupdate(S, U[:,j], false) ∀j.
 *
 *  Vectorized bits:
 *   - AVX2 axpy updates in solves, blocked over RHS columns (UKF_UPD_COLBLOCK).
 *   - AVX2 reciprocals instead of divides.
 *   - AVX2 for yyhat and xhat updates.
 *   - No gathers: U’s columns are copied with a simple strided loop.
 *
 *  Triangle convention:
 *   - Assumes Sy is upper-triangular (standard SR-UKF). Ensure S matches cholupdate()
 *     expectation (cholupdate() doc says L is lower-triangular).
 */
static void update_state_covarariance_matrix_and_state_estimation_vector(
    float *RESTRICT S,
    float *RESTRICT xhat,
    const float *RESTRICT yhat,
    const float *RESTRICT y,
    const float *RESTRICT Sy,
    const float *RESTRICT Pxy,
    uint8_t L8)
{
    const uint16_t n = (uint16_t)L8;
    const size_t nn = (size_t)n * (size_t)n;

    /* Workspace (aligned) */
    float *Z = (float *)linalg_aligned_alloc(32, nn * sizeof(float)); /* becomes K */
    float *U = (float *)linalg_aligned_alloc(32, nn * sizeof(float));
    float *Uk = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    float *Ky = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    float *yyhat = (float *)linalg_aligned_alloc(32, (size_t)n * sizeof(float));
    if (!Z || !U || !Uk || !Ky || !yyhat)
    {
        if (Z)
            linalg_aligned_free(Z);
        if (U)
            linalg_aligned_free(U);
        if (Uk)
            linalg_aligned_free(Uk);
        if (Ky)
            linalg_aligned_free(Ky);
        if (yyhat)
            linalg_aligned_free(yyhat);
        return;
    }

    memcpy(Z, Pxy, nn * sizeof(float));

    const int do_pf = (n >= (uint16_t)UKF_UPD_PF_MIN_N);
    const size_t pf_elts = (size_t)UKF_UPD_PF_DIST_BYTES / sizeof(float);

    /* ---------------- forward solve: Sy^T Z = Pxy (Sy upper ⇒ Sy^T lower) ---------------- */
    if (ukf_has_avx2() && n >= 8)
    {
        for (uint16_t i = 0; i < n; ++i)
        {
            const float sii = Sy[(size_t)i * n + i];

            /* Block RHS columns to keep a stripe hot */
            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = (uint16_t)((c0 + UKF_UPD_COLBLOCK <= n) ? UKF_UPD_COLBLOCK : (n - c0));

                /* Z[i, c0:c0+bc] -= Σ_{k<i} Sy[k,i] * Z[k, c0:c0+bc] */
                for (uint16_t k = 0; k < i; ++k)
                {
                    const float m = Sy[(size_t)k * n + i];
                    if (m == 0.0f)
                        continue;
                    const __m256 mv = _mm256_set1_ps(m);

                    uint16_t c = 0;
                    for (; (uint16_t)(c + 7) < bc; c = (uint16_t)(c + 8))
                    {
                        if (do_pf && c + pf_elts + 8 < bc)
                        {
                            _mm_prefetch((const char *)(&Z[(size_t)i * n + c0 + c + pf_elts]), _MM_HINT_T0);
                            _mm_prefetch((const char *)(&Z[(size_t)k * n + c0 + c + pf_elts]), _MM_HINT_T0);
                        }
                        __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                        __m256 zk = _mm256_loadu_ps(&Z[(size_t)k * n + c0 + c]);
                        zi = _mm256_fnmadd_ps(mv, zk, zi);
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }
                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* scale block by 1/sii */
                const __m256 rinv = _mm256_set1_ps(1.0f / sii);
                uint16_t c = 0;
                for (; (uint16_t)(c + 7) < bc; c = (uint16_t)(c + 8))
                {
                    __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                    _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], _mm256_mul_ps(zi, rinv));
                }
                for (; c < bc; ++c)
                    Z[(size_t)i * n + c0 + c] /= sii;
            }
        }
    }
    else
    {
        /* scalar fallback (unblocked) */
        for (uint16_t i = 0; i < n; ++i)
        {
            const float sii = Sy[(size_t)i * n + i];
            for (uint16_t k = 0; k < i; ++k)
            {
                const float m = Sy[(size_t)k * n + i];
                for (uint16_t c = 0; c < n; ++c)
                    Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }
            for (uint16_t c = 0; c < n; ++c)
                Z[(size_t)i * n + c] /= sii;
        }
    }

    /* ---------------- backward solve: Sy K = Z (Sy upper). In-place Z→K ---------------- */
    if (ukf_has_avx2() && n >= 8)
    {
        for (int i = (int)n - 1; i >= 0; --i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = (uint16_t)((c0 + UKF_UPD_COLBLOCK <= n) ? UKF_UPD_COLBLOCK : (n - c0));

                /* Z[i, block] -= Σ_{k>i} Sy[i,k] * Z[k, block] */
                for (uint16_t k = (uint16_t)(i + 1); k < n; ++k)
                {
                    const float m = Sy[(size_t)i * n + k];
                    if (m == 0.0f)
                        continue;
                    const __m256 mv = _mm256_set1_ps(m);

                    uint16_t c = 0;
                    for (; (uint16_t)(c + 7) < bc; c = (uint16_t)(c + 8))
                    {
                        if (do_pf && c + pf_elts + 8 < bc)
                        {
                            _mm_prefetch((const char *)(&Z[(size_t)i * n + c0 + c + pf_elts]), _MM_HINT_T0);
                            _mm_prefetch((const char *)(&Z[(size_t)k * n + c0 + c + pf_elts]), _MM_HINT_T0);
                        }
                        __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                        __m256 zk = _mm256_loadu_ps(&Z[(size_t)k * n + c0 + c]);
                        zi = _mm256_fnmadd_ps(mv, zk, zi);
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }
                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* scale block by 1/sii */
                const __m256 rinv = _mm256_set1_ps(1.0f / sii);
                uint16_t c = 0;
                for (; (uint16_t)(c + 7) < bc; c = (uint16_t)(c + 8))
                {
                    __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c0 + c]);
                    _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], _mm256_mul_ps(zi, rinv));
                }
                for (; c < bc; ++c)
                    Z[(size_t)i * n + c0 + c] /= sii;
            }
        }
    }
    else
    {
        for (int i = (int)n - 1; i >= 0; --i)
        {
            const float sii = Sy[(size_t)i * n + i];
            for (uint16_t k = (uint16_t)(i + 1); k < n; ++k)
            {
                const float m = Sy[(size_t)i * n + k];
                for (uint16_t c = 0; c < n; ++c)
                    Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }
            for (uint16_t c = 0; c < n; ++c)
                Z[(size_t)i * n + c] /= sii;
        }
    }
    /* Now Z holds K (n x n) */

    /* yyhat = y - yhat */
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8))
        {
            if (do_pf && i + pf_elts + 8 < n)
            {
                _mm_prefetch((const char *)(y + i + pf_elts), _MM_HINT_T0);
                _mm_prefetch((const char *)(yhat + i + pf_elts), _MM_HINT_T0);
            }
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vyh = _mm256_loadu_ps(yhat + i);
            _mm256_storeu_ps(yyhat + i, _mm256_sub_ps(vy, vyh));
        }
        for (; i < n; ++i)
            yyhat[i] = y[i] - yhat[i];
    }
    else
    {
        for (uint16_t i = 0; i < n; ++i)
            yyhat[i] = y[i] - yhat[i];
    }

    /* Ky = K * (y - yhat) */
    mul(Ky, Z /*K*/, yyhat, n, n, n, 1);

    /* xhat += Ky */
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8))
        {
            __m256 xv = _mm256_loadu_ps(xhat + i);
            __m256 kv = _mm256_loadu_ps(Ky + i);
            _mm256_storeu_ps(xhat + i, _mm256_add_ps(xv, kv));
        }
        for (; i < n; ++i)
            xhat[i] += Ky[i];
    }
    else
    {
        for (uint16_t i = 0; i < n; ++i)
            xhat[i] += Ky[i];
    }

    /* U = K * Sy */
    mul(U, Z /*K*/, Sy, n, n, n, n);

    /* Downdate S with each column of U (no gathers; simple strided copy) */
    for (uint16_t j = 0; j < n; ++j)
    {
        /* optional prefetch of future stripes */
        if (do_pf && (uint16_t)(j + 1) < n)
            _mm_prefetch((const char *)(&U[(size_t)0 * n + (j + 1)]), _MM_HINT_T0);

        for (uint16_t i = 0; i < n; ++i)
            Uk[i] = U[(size_t)i * n + j];

        cholupdate(S, Uk, n, /*rank_one_update=*/false);
    }

    linalg_aligned_free(Z);
    linalg_aligned_free(U);
    linalg_aligned_free(Uk);
    linalg_aligned_free(Ky);
    linalg_aligned_free(yyhat);
}

/**
 * @brief Square-root Unscented Kalman Filter (SR-UKF) step (predict + update).
 *
 * @details
 *  Orchestrates the SR-UKF cycle using vectorized kernels:
 *   - Weights, sigma generation, propagation with F, weighted mean,
 *     SR covariance via QR, identity H, measurement prediction,
 *     measurement SR covariance, cross-cov, and update via triangular solves
 *     + Cholesky downdates.
 *
 *  Improvements over scalar pipeline:
 *   - AVX2 kernels in the hot loops (sigma build, weighted sums, cross-cov).
 *   - No VLAs; uses aligned heap scratch for embedded safety and SIMD alignment.
 *   - Update avoids explicit matrix inverse; uses two triangular solves.
 *
 * @param[in]     y     Measurement vector [L].
 * @param[in,out] xhat  State mean [L]; on return the updated state estimate.
 * @param[in]     Rn    Measurement noise covariance [L x L].
 * @param[in]     Rv    Process noise covariance [L x L].
 * @param[in]     u     Control/input vector passed to F.
 * @param[in]     F     Transition function: F(dx, x, u).
 * @param[in,out] S     State SR covariance [L x L], updated in-place.
 * @param[in]     alpha,beta  UKF parameters.
 * @param[in]     L     State dimension.
 *
 * @retval 0        on success
 * @retval -EINVAL  if L==0
 * @retval -ENOMEM  if scratch allocation fails
 *
 * @note Row-major layout throughout; arrays must not alias unless documented.
 * @warning X/Y are centered in-place inside cross-covariance; pass copies if needed later.
 */
int sqr_ukf(float y[], float xhat[], float Rn[], float Rv[], float u[],
            void (*F)(float[], float[], float[]),
            float S[], float alpha, float beta, uint8_t L8)
{
    if (L8 == 0)
        return -EINVAL;

    const uint16_t L = L8; // promote to avoid overflow
    const uint16_t N = (uint16_t)(2 * L + 1);

    /* --- workspace sizes --- */
    const size_t szW = (size_t)N * sizeof(float);
    const size_t szLN = (size_t)L * N * sizeof(float);
    const size_t szLL = (size_t)L * L * sizeof(float);
    const size_t szL = (size_t)L * sizeof(float);

    /* --- allocate aligned scratch --- */
    float *Wc = (float *)ukf_aligned_alloc(szW);
    float *Wm = (float *)ukf_aligned_alloc(szW);
    float *X = (float *)ukf_aligned_alloc(szLN);
    float *Xst = (float *)ukf_aligned_alloc(szLN);
    float *Y = (float *)ukf_aligned_alloc(szLN);
    float *yhat = (float *)ukf_aligned_alloc(szL);
    float *Sy = (float *)ukf_aligned_alloc(szLL);
    float *Pxy = (float *)ukf_aligned_alloc(szLL);

    if (!Wc || !Wm || !X || !Xst || !Y || !yhat || !Sy || !Pxy)
    {
        ukf_aligned_free(Wc);
        ukf_aligned_free(Wm);
        ukf_aligned_free(X);
        ukf_aligned_free(Xst);
        ukf_aligned_free(Y);
        ukf_aligned_free(yhat);
        ukf_aligned_free(Sy);
        ukf_aligned_free(Pxy);
        return -ENOMEM;
    }

    /* ---------- Predict ---------- */

    // Weights (kappa=0 for state estimation)
    const float kappa = 0.0f;
    // no need to memset; create_weights writes all entries
    create_weights(Wc, Wm, alpha, beta, kappa, (uint8_t)L);

    // Sigma points for F
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);

    // Propagate through F
    compute_transistion_function(Xst, X, u, F, (uint8_t)L);

    // Predicted state: xhat = sum_j Wm[j] * Xst[:,j]
    multiply_sigma_point_matrix_to_weights(xhat, Xst, Wm, (uint8_t)L);

    // Predicted covariance square-root: S from Xst & Rv
    create_state_estimation_error_covariance_matrix(S, Wc, Xst, xhat, Rv, (uint8_t)L);

    // Sigma points for H
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);

    // Observation model (identity): Y = X
    H(Y, X, (uint8_t)L);

    // Predicted measurement
    multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, (uint8_t)L);

    // Measurement covariance square-root Sy
    create_state_estimation_error_covariance_matrix(Sy, Wc, Y, yhat, Rn, (uint8_t)L);

    // Cross-covariance Pxy
    create_state_cross_covariance_matrix(Pxy, Wc, X, Y, xhat, yhat, (uint8_t)L);

    /* ---------- Update ---------- */

    update_state_covarariance_matrix_and_state_estimation_vector(
        S, xhat, yhat, y, Sy, Pxy, (uint8_t)L);

    /* --- free scratch --- */
    ukf_aligned_free(Wc);
    ukf_aligned_free(Wm);
    ukf_aligned_free(X);
    ukf_aligned_free(Xst);
    ukf_aligned_free(Y);
    ukf_aligned_free(yhat);
    ukf_aligned_free(Sy);
    ukf_aligned_free(Pxy);

    return 0;
}
