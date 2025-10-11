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

#if LINALG_SIMD_ENABLE
static inline float avx2_sum_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

/**
 * @brief Compute Unscented Transform weights for mean (Wm) and covariance (Wc).
 *
 * @details
 *  Builds the standard UKF weights from parameters \p alpha, \p beta, \p kappa and
 *  state size \p L. The weights are:
 *  \f[
 *    \lambda = \alpha^2 (L + \kappa) - L,\quad
 *    W_m^{(0)} = \frac{\lambda}{L+\lambda},\quad
 *    W_c^{(0)} = W_m^{(0)} + (1 - \alpha^2 + \beta),\quad
 *    W_m^{(i)} = W_c^{(i)} = \frac{1}{2(L+\lambda)}\ \text{for}\ i=1..2L.
 *  \f]
 *
 *  A fast AVX2 path bulk-fills the constant tail (i ≥ 1) in 8-wide chunks to reduce
 *  loop overhead and memory traffic. For small N or when AVX2 is unavailable, a
 *  scalar loop is used.
 *
 * @param[out] Wc    Covariance weights, length N = 2L + 1.
 * @param[out] Wm    Mean weights, length N = 2L + 1.
 * @param[in]  alpha Spread parameter (typ. 1e-3 ≤ α ≤ 1).
 * @param[in]  beta  Prior distribution knowledge (β=2 for Gaussian optimality).
 * @param[in]  kappa Secondary scaling parameter (often 0 or 3−L).
 * @param[in]  L     State dimension.
 *
 * @note
 *  - Requires L ≥ 1 for meaningful weights.
 *  - When \f$L+\lambda\f$ is very small, denominators can amplify round-off.
 *    Choose \p alpha/\p kappa sensibly for numerical stability.
 *  - Falls back to scalar if AVX2/FMA is not available or N < 9.
 */
static void create_weights(float Wc[],
                           float Wm[],
                           float alpha,
                           float beta,
                           float kappa,
                           uint8_t L)
{
    const size_t N  = (size_t)(2u * L + 1u);    //!< Number of sigma points
    const float  Lf = (float)L;                 //!< State size as float

    /* λ = α^2 (L + κ) − L */
    const float lam = alpha * alpha * (Lf + kappa) - Lf;

    /* Common denominator 1 / (L + λ) */
    const float den = 1.0f / (Lf + lam);

    /* First element (i = 0) */
    Wm[0] = lam * den;
    Wc[0] = Wm[0] + 1.0f - alpha * alpha + beta;

    /* Tail (i ≥ 1): 0.5 / (L + λ) */
    const float hv = 0.5f * den;

#if LINALG_SIMD_ENABLE
    /* AVX2 bulk fill of the constant tail when N >= 9 (i.e., at least one full 8-lane chunk). */
    if (ukf_has_avx2() && N >= 9)
    {
        const __m256 v = _mm256_set1_ps(hv);
        size_t i = 1;                              //!< Start filling from index 1
        for (; i + 7 < N; i += 8)
        {
            _mm256_storeu_ps(&Wm[i], v);          //!< Store 8 identical Wm values
            _mm256_storeu_ps(&Wc[i], v);          //!< Store 8 identical Wc values
        }
        /* Scalar cleanup for the remaining elements (if N is not a multiple of 8). */
        for (; i < N; ++i)
        {
            Wm[i] = hv;
            Wc[i] = hv;
        }
        return;
    }
#endif

    /* Portable scalar tail initialization */
    for (size_t i = 1; i < N; ++i)
    {
        Wm[i] = hv;
        Wc[i] = hv;
    }
}

/**
 * @brief Build the Unscented sigma-point matrix X from mean x and SR-covariance S.
 *
 * @details
 *  Constructs the (L × (2L+1)) sigma matrix in row-major order:
 *  - Column 0:            X(:,0)     = x
 *  - Columns 1..L:        X(:,j)     = x + γ S(:,j)         (j=1..L)
 *  - Columns L+1..2L:     X(:,L+j)   = x - γ S(:,j)         (j=1..L)
 *
 *  where γ = α √(L + κ). The input S is the **square-root covariance** (upper
 *  or lower is fine since we access rows of S here; we simply scale each row’s
 *  entries by ±γ).
 *
 *  SIMD path:
 *   - Uses AVX2/FMA to compute two sigma columns ( +γ and −γ ) in parallel for
 *     8 elements per iteration.
 *   - Optional row-ahead prefetch to reduce cache miss latency for large L.
 *
 *  Scalar path:
 *   - Portable, straightforward loops for all L and (2L+1) columns.
 *
 * @param[out] X      Sigma matrix, row-major, size L × (2L+1).
 * @param[in]  x      State mean vector of length L.
 * @param[in]  S      Square-root covariance (SR) matrix, row-major L × L.
 * @param[in]  alpha  UKF spread parameter (α).
 * @param[in]  kappa  Secondary scaling parameter (κ).
 * @param[in]  L8     State dimension (stored as uint8_t to match surrounding API).
 *
 * @note
 *  - X must not alias x or S.
 *  - This routine assumes S is already a valid SR factor (e.g., from QR/Cholesky).
 *  - Vectorized path requires AVX2+FMA and benefits most when L ≥ 8.
 */
static void create_sigma_point_matrix(float X[],
                                      const float x[],
                                      const float S[],
                                      float alpha,
                                      float kappa,
                                      uint8_t L8)
{
    const size_t L = (size_t)L8;               //!< State dimension
    const size_t N = 2u * L + 1u;              //!< Number of sigma points
    const float  gamma = alpha * sqrtf((float)L + kappa);  //!< σ scaling

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && L >= 8)
    {
        const __m256 g  = _mm256_set1_ps(gamma);   //!< +γ broadcast
        const __m256 ng = _mm256_set1_ps(-gamma);  //!< −γ broadcast

        /* Prefetch policy */
        const size_t pf_elts   = UKF_PREFETCH_DIST_BYTES / sizeof(float);
        const int    do_pf     = (L >= (size_t)UKF_PREFETCH_MIN_L);
        const int    rows_ahead= UKF_PREFETCH_ROWS_AHEAD;

        for (size_t i = 0; i < L; ++i)
        {
            float *Xi        = X + i * N;     //!< Row pointer of X (state i across columns)
            const float *Si  = S + i * L;     //!< Row pointer of S (state i across its L entries)
            const __m256 xi8 = _mm256_set1_ps(x[i]);  //!< Broadcast x[i] to 8 lanes

            /* Column 0 is the mean itself */
            Xi[0] = x[i];

            /* Prefetch a few future rows of S and X to warm caches (optional) */
            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        const float *Spi = S + ip * L;
                        float *Xpi       = X + ip * N;
                        _mm_prefetch((const char *)Spi, _MM_HINT_T0);
                        _mm_prefetch((const char *)Xpi, _MM_HINT_T0);
                    }
                }
            }

            /* Vectorized loop over S row in chunks of 8:
               We produce X[i, 1..L] = x[i] + γ*S[i, :]
                         and X[i, L+1..2L] = x[i] − γ*S[i, :]                 */
            size_t j = 0;
            for (; j + 7 < L; j += 8)
            {
                if (do_pf)
                {
                    _mm_prefetch((const char *)(Si + j + pf_elts),            _MM_HINT_T0);
                    _mm_prefetch((const char *)(Xi + 1 + j + pf_elts),        _MM_HINT_T0);
                    _mm_prefetch((const char *)(Xi + 1 + L + j + pf_elts),    _MM_HINT_T0);
                }

                __m256 s8    = _mm256_loadu_ps(Si + j);          //!< Load 8 entries from S row
                __m256 plus  = _mm256_fmadd_ps(g,  s8, xi8);     //!< x[i] + γ * S[i, j..j+7]
                __m256 minus = _mm256_fmadd_ps(ng, s8, xi8);     //!< x[i] − γ * S[i, j..j+7]

                _mm256_storeu_ps(Xi + 1 + j,       plus);        //!< Write +γ branch
                _mm256_storeu_ps(Xi + 1 + L + j,   minus);       //!< Write −γ branch
            }

            /* Scalar tail for the remaining (L % 8) elements */
            for (; j < L; ++j)
            {
                const float s = Si[j];
                Xi[1 + j]       = x[i] + gamma * s;  //!< +γ column
                Xi[1 + L + j]   = x[i] - gamma * s;  //!< −γ column
            }
        }
        return;
    }
#endif

    /* ------------------------ Scalar fallback ------------------------ */
    for (size_t i = 0; i < L; ++i)
    {
        float *Xi        = X + i * N;     //!< Row pointer of X (state i across columns)
        const float *Si  = S + i * L;     //!< Row pointer of S (state i across its L entries)

        Xi[0] = x[i];                     //!< Mean in column 0

        for (size_t j = 0; j < L; ++j)
        {
            const float s = Si[j];
            Xi[1 + j]       = x[i] + gamma * s;  //!< +γ column
            Xi[1 + L + j]   = x[i] - gamma * s;  //!< −γ column
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
static void compute_transition_function(float Xstar[], const float X[], const float u[],
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
 * @brief Compute the weighted mean of sigma points.
 *
 * @details
 *  Given the sigma point matrix @p X (L × (2L+1)) and weight vector @p W ((2L+1) × 1),
 *  this function computes:
 *  \f[
 *      x_i = \sum_{j=0}^{2L} W_j \, X_{i,j}, \quad i = 0,\dots,L-1
 *  \f]
 *  which corresponds to the weighted mean of each state dimension.
 *
 *  - For small matrices or when AVX2 is unavailable, it falls back to a scalar loop.
 *  - For large matrices, it uses an AVX2/FMA optimized inner loop processing 8 weights at a time.
 *  - The SIMD path computes two rows (state dimensions) per iteration to maximize throughput.
 *
 * @param[out] x  Output mean vector of length L.
 * @param[in]  X  Sigma point matrix of shape (L × (2L+1)), row-major.
 * @param[in]  W  Weights vector of length (2L+1).
 * @param[in]  L  Number of state dimensions.
 *
 * @note
 *  The function automatically detects AVX2 support and falls back to portable scalar code.
 *  Prefetching hints are used to reduce memory stalls when L is large enough.
 */
static void multiply_sigma_point_matrix_to_weights(float x[],
                                                   const float X[],
                                                   const float W[],
                                                   uint8_t L)
{
    const size_t Ls = (size_t)L;          //!< State dimension
    const size_t N  = 2u * Ls + 1u;       //!< Number of sigma points

    /* --------------------- Scalar fallback path --------------------- */
    if (!ukf_has_avx2() || N < 8)
    {
        for (size_t i = 0; i < Ls; ++i)
        {
            const float *row = &X[i * N]; //!< Pointer to i-th row of sigma points
            float acc = 0.0f;
            for (size_t j = 0; j < N; ++j)
                acc += W[j] * row[j];     //!< Accumulate weighted sum for this state dimension
            x[i] = acc;
        }
        return;
    }

#if LINALG_SIMD_ENABLE
    /* --------------------- Vectorized AVX2 path --------------------- */
    const int do_pf = (Ls >= (size_t)UKF_MEAN_PF_MIN_ROWS); //!< Enable prefetch for large matrices
    const int rows_ahead = UKF_MEAN_PF_ROWS_AHEAD;          //!< Number of future rows to prefetch

    size_t i = 0;
    for (; i + 1 < Ls; i += 2)
    {
        /* Process two consecutive rows (state dimensions) together */
        const float *row0 = &X[(i + 0) * N];
        const float *row1 = &X[(i + 1) * N];

        /* Prefetch a few rows ahead to keep data in cache */
        if (do_pf && rows_ahead > 0)
        {
            for (int ra = 1; ra <= rows_ahead; ++ra)
            {
                const size_t ip = i + (size_t)ra * 2;
                if (ip < Ls)
                {
                    _mm_prefetch((const char *)(&X[ip * N]), _MM_HINT_T0);
                    if (ip + 1 < Ls)
                        _mm_prefetch((const char *)(&X[(ip + 1) * N]), _MM_HINT_T0);
                }
            }
        }

        /* Vector accumulators for two rows */
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        size_t j = 0;
        for (; j + 7 < N; j += 8)
        {
            /**
             * Load 8 weights and the corresponding 8 sigma points for each of
             * the two rows, then accumulate using FMA:
             *   acc += W * X
             */
            __m256 wv = _mm256_loadu_ps(&W[j]);   //!< Load 8 weights
            __m256 x0 = _mm256_loadu_ps(&row0[j]);//!< Load 8 sigma values (row 0)
            __m256 x1 = _mm256_loadu_ps(&row1[j]);//!< Load 8 sigma values (row 1)
            acc0 = _mm256_fmadd_ps(wv, x0, acc0); //!< acc0 += W * X0
            acc1 = _mm256_fmadd_ps(wv, x1, acc1); //!< acc1 += W * X1
        }

        /* Reduce 8-lane accumulators to scalar sums */
        float sum0 = avx2_sum_ps(acc0);
        float sum1 = avx2_sum_ps(acc1);

        /* Handle any leftover (non-multiple-of-8) sigma points */
        for (; j < N; ++j)
        {
            sum0 += W[j] * row0[j];
            sum1 += W[j] * row1[j];
        }

        /* Store weighted means for these two state dimensions */
        x[i + 0] = sum0;
        x[i + 1] = sum1;
    }

    /* Handle last row if L is odd */
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
        float sum = avx2_sum_ps(acc);
        for (; j < N; ++j)
            sum += W[j] * row[j];
        x[i] = sum;
    }
#else
    /* --------------------- Portable fallback (no AVX2) --------------------- */
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
 * @brief Build square-root state covariance S (SR-UKF) via QR of weighted deviations.
 *
 * @details
 *  Constructs the augmented matrix A′ (size M×L with M=3L) used by the SR-UKF:
 *
 *  Let X be the propagated sigma points (L × N, N = 2L+1), x the predicted mean (L),
 *  and W the covariance weights. Define:
 *   - K = 2L (number of deviation columns excluding the mean column),
 *   - w1 = sqrt(|W[1]|) (common absolute weight for all non-zero sigma columns),
 *   - w0 = sqrt(|W[0]|) (mean-deviation weight).
 *
 *  Then the columns of A′ are formed as:
 *   - Rows 0..K−1   :  w1 * (X[:, 1..N−1] − x)          (stacked by sigma index)
 *   - Rows K..M−1   :  Rsr                              (per-row copy of SR noise)
 *
 *  Next, compute R from a QR factorization (Householder) of A′:
 *     A′ = Q * R
 *  The upper L×L part of R is a square-root covariance (upper-triangular). Finally,
 *  perform a rank-1 Cholesky update/downdate with (X[:,0] − x), scaled by w0, using
 *  an **upper-triangular** routine:
 *     S ← cholupdate_upper(S,  w0*(X[:,0] − x), update=(W[0] ≥ 0))
 *
 *  Vectorization:
 *   - Deviations (X − x) are built in 8-wide chunks with AVX2 and scaled by w1.
 *   - The SR noise block uses 8-wide loads/stores (Rsr already factored).
 *   - The mean deviation vector b = w0*(X[:,0] − x) is built alongside.
 *
 *  Improvements vs. scalar:
 *   - Fewer loop-carried address computations and better cache residency.
 *   - Avoids forming identity/zero padding explicitly; builds only needed blocks.
 *   - Uses optimized @ref qr and an in-place @ref cholupdate_upper.
 *
 * @param[out] S     Output square-root covariance (L × L), **upper-triangular**.
 * @param[in]  W     Covariance weights, length N = 2L + 1.
 * @param[in]  X     Propagated sigma points (L × N), row-major.
 * @param[in]  x     Predicted state mean (L).
 * @param[in]  Rsr   Square-root of process/measurement noise (L × L), **upper-triangular**.
 * @param[in]  L8    Dimension L (stored as uint8_t to match surrounding API).
 *
 * @retval 0       Success.
 * @retval -ENOMEM Workspace allocation failed.
 * @retval -EIO    QR decomposition failed.
 * @retval -EFAULT Resulting S failed a simple PD sanity check (non-positive/NaN diagonal).
 *
 * @warning S, X, and Rsr must not alias. All matrices are row-major.
 * @note    This routine assumes an **upper** SR convention end-to-end.
 */
static int create_state_estimation_error_covariance_matrix(float S[],
                                                           float W[],
                                                           float X[],
                                                           float x[],
                                                           const float Rsr[],
                                                           uint8_t L8)
{
    const size_t L = (size_t)L8;                 //!< State dimension
    const size_t N = 2u * L + 1u;                //!< # of sigma points
    const size_t K = 2u * L;                     //!< # of deviation columns (excluding mean)
    const size_t M = 3u * L;                     //!< Augmented rows (deviations + SR noise)

    const float w1s = sqrtf(fabsf(W[1]));        //!< |W[1]|^1/2 for columns 1..K
    const float w0s = sqrtf(fabsf(W[0]));        //!< |W[0]|^1/2 for mean deviation

    /* Reusable QR workspace (thread-local if supported) */
#if defined(__GNUC__) || defined(__clang__)
    static __thread ukf_qr_ws_t ws = {0};
#else
    static ukf_qr_ws_t ws = {0};
#endif
    if (ukf_qr_ws_ensure(&ws, L) != 0)
        return -ENOMEM;

    float *Aprime = ws.Aprime;   /* M×L, column i holds stacked blocks for state i */
    float *R_     = ws.R_;       /* M×L, will hold R from QR (we copy upper L×L)  */
    float *b      = ws.b;        /* L, mean deviation vector scaled by w0s        */

    /* Prefetch policy knobs */
    const int    do_pf      = (int)(L >= (size_t)UKF_APRIME_PF_MIN_L);
    const int    rows_ahead = UKF_APRIME_PF_ROWS_AHEAD;
    const size_t pf_elems   = (size_t)UKF_APRIME_PF_DIST_BYTES / sizeof(float);

#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && L >= 8)
    {
        const __m256 w1v = _mm256_set1_ps(w1s);

        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;   //!< Row i of X (state i across all sigmas)
            const float  xi = x[i];

            /* Row-ahead prefetch (optional) */
            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X   + ip * N), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Rsr + ip * L), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Aprime + ip),  _MM_HINT_T0);
                        _mm_prefetch((const char *)(Aprime + (K * L) + ip), _MM_HINT_T0);
                    }
                }
            }

            /* b[i] = w0s * (Xi[0] − x[i])  (contiguous access) */
            b[i] = w0s * (Xi[0] - xi);

            /* Deviations block (rows 0..K−1), 8-wide chunks:
               Aprime[r*L + i] = w1s * (X[i, r+1] − x[i])                               */
            size_t r = 0;
            const __m256 xi8 = _mm256_set1_ps(xi);
            for (; r + 7 < K; r += 8)
            {
                if (do_pf && r + pf_elems + 8 < K)
                    _mm_prefetch((const char *)(&Xi[r + 1 + pf_elems]), _MM_HINT_T0);

                __m256 Xv   = _mm256_loadu_ps(&Xi[r + 1]);       //!< Load 8 sigma values
                __m256 diff = _mm256_sub_ps(Xv, xi8);            //!< (X − x)
                __m256 out  = _mm256_mul_ps(w1v, diff);          //!< Scale by w1

                /* Column-major write into Aprime’s stacked block for column i:
                   (strided stores via a small lane buffer) */
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, out);
#pragma GCC ivdep
                for (int k2 = 0; k2 < 8; ++k2)
                    Aprime[(r + (size_t)k2) * L + i] = lanes[k2];
            }
            for (; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);

            /* SR noise block (rows K..M−1), copy row i of upper SR noise Rsr */
            size_t t = 0;
            for (; t + 7 < L; t += 8)
            {
                if (do_pf && t + pf_elems + 8 < L)
                {
                    _mm_prefetch((const char *)(&Rsr[i * L + t + pf_elems]), _MM_HINT_T0);
                    _mm_prefetch((const char *)(Aprime + (K + t + pf_elems) * L + i), _MM_HINT_T0);
                }

                __m256 Sv = _mm256_loadu_ps(&Rsr[i * L + t]);    //!< Load 8 from SR noise row i
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, Sv);
#pragma GCC ivdep
                for (int k2 = 0; k2 < 8; ++k2)
                    Aprime[(K + t + (size_t)k2) * L + i] = lanes[k2];
            }
            for (; t < L; ++t)
                Aprime[(K + t) * L + i] = Rsr[i * L + t];
        }
    }
    else
#endif
    {
        /* Portable scalar build of A′ and b */
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X + i * N;
            const float  xi = x[i];

            if (do_pf && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X   + ip * N), _MM_HINT_T0);
                        _mm_prefetch((const char *)(Rsr + ip * L), _MM_HINT_T0);
                    }
                }
            }

            b[i] = w0s * (Xi[0] - xi);

            for (size_t r = 0; r < K; ++r)
                Aprime[r * L + i] = w1s * (Xi[r + 1] - xi);

            for (size_t t = 0; t < L; ++t)
                Aprime[(K + t) * L + i] = Rsr[i * L + t];
        }
    }

    /* QR of A′ (M×L); we only need R. qr() must skip Q work when only_compute_R=true. */
    if (qr(Aprime, /*Q=*/NULL, R_, (uint16_t)M, (uint16_t)L, /*only_compute_R=*/true) != 0)
        return -EIO;

    /* Copy the upper L×L block of R_ into S (upper-triangular SR) */
    memcpy(S, R_, (size_t)L * L * sizeof(float));

    int rc = cholupdate(/*L=*/S, /*x=*/(const float *)b, /*n=*/(uint16_t)L,
                    /*is_upper=*/true, /*rank_one_update=*/(W[0] >= 0.0f));
    if (rc != 0) {
        /* SR covariance update failed (should be rare for prediction step) */
        return rc;  /* propagate error (-EDOM or -ENOMEM) */
    }

    /* Simple SPD sanity check: diagonal elements must be positive and finite */
    for (size_t i = 0; i < L; ++i)
        if (!(S[i * L + i] > 0.0f && isfinite(S[i * L + i])))
            return -EFAULT;

    return 0;
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
 * @brief Compute cross-covariance Pxy = X_c · diag(W) · Y_c^T without materializing diag(W).
 *
 * @details
 *  Given sigma-point matrices @p X and @p Y (each L × N, row-major; N = 2L+1) and their means
 *  @p x and @p y (length L), this routine computes the cross-covariance:
 *
 *  \f[
 *      P_{xy} \;=\; \sum_{j=0}^{N-1} W_j \,(X_{\cdot j}-x)\,(Y_{\cdot j}-y)^{\top}
 *                 \;=\; X_c \;\mathrm{diag}(W)\; Y_c^{\top},
 *  \f]
 *
 *  where the centered (but not yet weighted) matrices are:
 *   - \( X_c = X - x \mathbb{1}^{\top} \in \mathbb{R}^{L\times N} \)
 *   - \( Y_c = Y - y \mathbb{1}^{\top} \in \mathbb{R}^{L\times N} \)
 *
 *  Implementation outline (non-destructive):
 *   1) Build a padded/centered/weighted copy \( \tilde{X} \in \mathbb{R}^{L\times N_8} \),
 *      with \( \tilde{X}_{i,j} = (X_{i,j}-x_i)\,W_j \) and zero-pad columns to
 *      \( N_8=\lceil N/8\rceil\cdot 8 \) for vectorization.
 *   2) Build a padded, centered transpose \( \widetilde{Y^T} \in \mathbb{R}^{N_8\times L} \),
 *      with \( \widetilde{Y^T}_{j,k} = (Y_{k,j}-y_k) \) and zero-pad rows (j ≥ N).
 *   3) Multiply once: \( P = \tilde{X} \cdot \widetilde{Y^T} \).
 *
 *  Vectorization:
 *   - Step (1) uses AVX2 to compute \((X-x)\odot W\) 8 elements at a time per row.
 *   - Step (2) uses an AVX2 8×8 transpose micro-kernel to efficiently build \(Y_c^T\).
 *   - Step (3) relies on the project’s optimized @ref mul kernel (row-major GEMM-lite).
 *
 *  Advantages:
 *   - Avoids explicitly forming a dense diag(W).
 *   - Performs a single GEMM-like multiply rather than two.
 *   - Keeps inputs @p X and @p Y intact (non-destructive).
 *
 * @param[out] P   Output cross-covariance (L × L), row-major. Fully written.
 * @param[in]  W   Weights vector of length N = 2L + 1.
 * @param[in]  X   Sigma matrix for state (L × N), row-major. **Not** modified.
 * @param[in]  Y   Sigma matrix for measurement (L × N), row-major. **Not** modified.
 * @param[in]  x   Mean of X (length L).
 * @param[in]  y   Mean of Y (length L).
 * @param[in]  L8  Dimension L (stored as uint8_t to match surrounding API).
 *
 * @note
 *  - This function is **non-destructive**: inputs @p X and @p Y are not modified.
 *  - Temporary buffers are 32B-aligned and padded to a multiple of 8 columns/rows.
 *  - All matrices are row-major. @ref mul is expected to handle the provided shapes.
 */
static void create_state_cross_covariance_matrix(float *RESTRICT P,
                                                 float *RESTRICT W,
                                                 const float *RESTRICT X,
                                                 const float *RESTRICT Y,
                                                 const float *RESTRICT x,
                                                 const float *RESTRICT y,
                                                 uint8_t L8)
{
    const size_t L  = (size_t)L8;                  //!< State / measurement dimension
    const size_t N  = 2u * L + 1u;                 //!< Number of sigma points
    const size_t N8 = ukf_round_up8(N);            //!< Padded to next multiple of 8 for SIMD

    /* Zero the output cross-covariance. */
    memset(P, 0, L * L * sizeof(float));

    /* Allocate centered/weighted temporaries:
       - Xc : L × N8  (row-major)
       - YTc: N8 × L  (row-major) == (centered Y)^T with padding on rows j ≥ N
     */
    float *Xc  = (float *)linalg_aligned_alloc(32, L * N8 * sizeof(float));
    float *YTc = (float *)linalg_aligned_alloc(32, N8 * L * sizeof(float));
    if (!Xc || !YTc)
    {
        if (Xc)  linalg_aligned_free(Xc);
        if (YTc) linalg_aligned_free(YTc);
        return; /* Out-of-memory; leave P as zeros */
    }

    /* Prefetch / tiling knobs */
    const int    do_pf_rows  = (int)(L >= (size_t)UKF_PXY_PF_MIN_L);
    const int    do_pf_in    = (int)(N >= (size_t)UKF_PXY_PF_MIN_N);
    const int    rows_ahead  = UKF_PXY_PF_ROWS_AHEAD;
    const size_t pf_elems    = (size_t)UKF_PXY_PF_DIST_BYTES / sizeof(float);

    /* ----------------------------------------------------------------
     * Step 1: Build Xc = (X - x) ⊙ W, with column-padding to N8.
     * ---------------------------------------------------------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 8)
    {
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X  + i * N;      //!< Row i of X
            float       *Xci = Xc + i * N8;    //!< Row i of Xc
            const __m256 xi8 = _mm256_set1_ps(x[i]);

            /* Row-ahead prefetch for X rows and Xc destinations */
            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
                    if (ip < L)
                    {
                        _mm_prefetch((const char *)(X  + ip * N),  _MM_HINT_T0);
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
                    _mm_prefetch((const char *)(W  + j + pf_elems), _MM_HINT_T0);
                }
                __m256 xv   = _mm256_loadu_ps(Xi + j);          //!< X entries
                __m256 wv   = _mm256_loadu_ps(W  + j);          //!< weights
                __m256 diff = _mm256_sub_ps(xv, xi8);           //!< (X - x)
                _mm256_storeu_ps(Xci + j, _mm256_mul_ps(diff, wv)); //!< (X - x) ⊙ W
            }
            /* Scalar remainder and padding */
            for (; j < N; ++j)  Xci[j] = (Xi[j] - x[i]) * W[j];
            for (; j < N8; ++j) Xci[j] = 0.0f;
        }
    }
    else
#endif
    {
        for (size_t i = 0; i < L; ++i)
        {
            const float *Xi = X  + i * N;
            float       *Xci = Xc + i * N8;

            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t ip = i + (size_t)ra;
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
                    _mm_prefetch((const char *)(W  + j + pf_elems), _MM_HINT_T0);
                }
                Xci[j] = (Xi[j] - x[i]) * W[j];
            }
            for (; j < N8; ++j) Xci[j] = 0.0f;
        }
    }

    /* ----------------------------------------------------------------
     * Step 2: Build YTc = (Y - y)^T with row-padding to N8.
     *         YTc has shape (N8 × L), row-major, i.e., each row is one
     *         centered sigma column from Y (or zeros for padded rows).
     * ---------------------------------------------------------------- */
#if LINALG_SIMD_ENABLE
    if (ukf_has_avx2() && N >= 8 && L >= 8)
    {
        /* Zero-pad rows j = N..N8-1 (these entire rows are 0) */
        for (size_t jp = N; jp < N8; ++jp)
            memset(YTc + jp * L, 0, L * sizeof(float));

        size_t k0 = 0;
        for (; k0 + 7 < L; k0 += 8)        /* Process Y in 8-row panels */
        {
            /* Prefetch upcoming 8-row panels of Y */
            if (do_pf_rows && rows_ahead > 0)
            {
                for (int ra = 1; ra <= rows_ahead; ++ra)
                {
                    const size_t kp = k0 + (size_t)ra * 8;
                    if (kp < L)
                        _mm_prefetch((const char *)(Y + kp * N), _MM_HINT_T0);
                }
            }

            size_t j0 = 0;
            for (; j0 + 7 < N; j0 += 8)   /* For each 8-wide column block */
            {
                if (do_pf_in && j0 + pf_elems + 8 < N)
                {
                    for (int r = 0; r < 8; ++r)
                        _mm_prefetch((const char *)(Y + (k0 + (size_t)r) * N + j0 + pf_elems), _MM_HINT_T0);
                    _mm_prefetch((const char *)(YTc + (j0 + pf_elems) * L + k0), _MM_HINT_T0);
                }

                /* Load 8 rows × 8 cols tile of Y and center by y */
                __m256 row[8];
                for (int r = 0; r < 8; ++r)
                {
                    const float *Yr = Y + (k0 + (size_t)r) * N + j0;
                    __m256 yr = _mm256_set1_ps(y[k0 + (size_t)r]);
                    row[r] = _mm256_sub_ps(_mm256_loadu_ps(Yr), yr);
                }

                /* Transpose 8×8 block so we can write contiguous rows into YTc */
                __m256 col[8];
                ukf_transpose8x8_ps(row, col);

                for (int c = 0; c < 8; ++c)
                    _mm256_storeu_ps(YTc + (j0 + (size_t)c) * L + k0, col[c]);
            }

            /* Handle N-tail for these 8 rows */
            for (; j0 < N; ++j0)
            {
                float *YTrow = YTc + j0 * L;   /* row j0 in YTc */
                for (int r = 0; r < 8; ++r)
                    YTrow[k0 + (size_t)r] = Y[(k0 + (size_t)r) * N + j0] - y[k0 + (size_t)r];
            }
        }

        /* Handle remaining rows (L-tail) one-by-one */
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
        /* Portable path: build centered Y^T directly */
        for (size_t j = 0; j < N; ++j)
        {
            const float *Ycol0 = Y + j;    /* Address Y[0,j] in row-major: Y[k*N + j] */
            float *YTrow = YTc + j * L;    /* Row j in YTc (since YTc is N8×L row-major) */
            for (size_t k = 0; k < L; ++k)
                YTrow[k] = Ycol0[k * N] - y[k];
        }
        /* Zero-pad extra rows (j = N..N8-1) */
        for (size_t j = N; j < N8; ++j)
            memset(YTc + j * L, 0, L * sizeof(float));
    }

    /* ----------------------------------------------------------------
     * Step 3: Multiply once — P = Xc · YTc
     * ---------------------------------------------------------------- */
    if (do_pf_rows)
    {
        _mm_prefetch((const char *)Xc, _MM_HINT_T0);
        _mm_prefetch((const char *)YTc, _MM_HINT_T0);
        _mm_prefetch((const char *)P,  _MM_HINT_T0);
    }
    /* Shapes: Xc (L×N8), YTc (N8×L) → P (L×L) */
    (void)mul(P, Xc, YTc, (uint16_t)L, (uint16_t)N8, (uint16_t)N8, (uint16_t)L);

    /* Free temporaries */
    linalg_aligned_free(Xc);
    linalg_aligned_free(YTc);
}

/**
 * @brief Measurement update: compute Kalman gain, update state, and downdate SR covariance.
 *
 * @details
 *  Solves the linear system for the Kalman gain without forming any explicit inverses.
 *  With the measurement SR factor @p Sy (upper-triangular) and cross-covariance @p Pxy:
 *
 *  1) Forward solve (lower):   \( S_y^\top Z = P_{xy} \)
 *  2) Backward solve (upper):  \( S_y K     = Z \)     (in-place: Z becomes K)
 *
 *  Then:
 *   - \( \delta y = y - \hat{y} \)
 *   - \( K \delta y \) is accumulated into @p xhat
 *   - \( U = K S_y \) is formed and @p S is downdated via rank-1 Cholesky for each column of U:
 *       \( S \leftarrow \mathrm{cholupdate\_upper}(S, U_{\cdot j}, \mathrm{update}=false) \)
 *
 *  Vectorization:
 *   - AVX2/FMA AXPY-like updates inside both triangular solves, blocked over RHS columns
 *     (size controlled by UKF_UPD_COLBLOCK).
 *   - AVX2 used for building \( \delta y \) and for accumulating @p xhat.
 *   - Prefetching along RHS panels to reduce cache miss latency when n is large.
 *
 *  Conventions:
 *   - @p Sy is **upper-triangular** (SR of the measurement covariance).
 *   - @p S is **upper-triangular** (SR of the state covariance) and is downdated in-place.
 *
 * @param[in,out] S     State SR covariance (n × n), upper-triangular, updated in-place (downdated).
 * @param[in,out] xhat  State estimate (length n); on return, \( \hat{x}^+ = \hat{x} + K (y-\hat{y}) \).
 * @param[in]     yhat  Predicted measurement (length n).
 * @param[in]     y     Actual measurement (length n).
 * @param[in]     Sy    Measurement SR covariance (n × n), **upper-triangular**.
 * @param[in]     Pxy   Cross-covariance between state and measurement (n × n).
 * @param[in]     L8    Dimension n (stored as uint8_t to match surrounding API).
 *
 * @retval 0        Success.
 * @retval -ENOMEM  Workspace allocation failed.
 * @retval -EIO     Underlying GEMM (mul) failed.
 *
 * @note
 *  Uses a thread-local workspace (see ukf_upd_ws_t). All matrices are row-major.
 */
static int update_state_covariance_matrix_and_state_estimation_vector(
    float *RESTRICT S,
    float *RESTRICT xhat,
    const float *RESTRICT yhat,
    const float *RESTRICT y,
    const float *RESTRICT Sy,
    const float *RESTRICT Pxy,
    uint8_t L8)
{
    const uint16_t n  = (uint16_t)L8;
    const size_t   nn = (size_t)n * (size_t)n;

    /* Thread-local reusable workspace (Z, U, Uk, Ky, yyhat). */
    static __thread ukf_upd_ws_t uws = {0};
    if (ukf_upd_ws_ensure(&uws, n) != 0)
        return -ENOMEM;

    float *Z     = uws.Z;     /* (n×n) RHS workspace → becomes K */
    float *U     = uws.U;     /* (n×n) temporary for K*Sy */
    float *Uk    = uws.Uk;    /* (n)   single column buffer for chol downdate */
    float *Ky    = uws.Ky;    /* (n)   K*(y−yhat) */
    float *yyhat = uws.yyhat; /* (n)   innovation v = y − yhat */

    /* Z := Pxy (we’ll overwrite it into K after the triangular solves) */
    memcpy(Z, Pxy, nn * sizeof(float));

    /* Prefetch knobs for large problems */
    const int    do_pf   = (n >= (uint16_t)UKF_UPD_PF_MIN_N);
    const size_t pf_elts = (size_t)UKF_UPD_PF_DIST_BYTES / sizeof(float);

    /* --------------------------------------------------------------
     * Forward solve:  S_y^T · Z = P_xy      (S_y upper ⇒ S_y^T lower)
     * Solve row-by-row (i ascending). Within each row, block over
     * RHS columns to improve temporal locality.
     * -------------------------------------------------------------- */
    if (ukf_has_avx2() && n >= 8)
    {
        for (uint16_t i = 0; i < n; ++i)
        {
            const float sii = Sy[(size_t)i * n + i];  /* diagonal (scalar) */

            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = (uint16_t)((c0 + UKF_UPD_COLBLOCK <= n)
                                    ? UKF_UPD_COLBLOCK : (n - c0));

                /* Z[i,*] -= Σ_{k<i} Sy[k,i] * Z[k,*] */
                for (uint16_t k = 0; k < i; ++k)
                {
                    const float m = Sy[(size_t)k * n + i];
                    if (m == 0.0f) continue;
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
                        zi = _mm256_fnmadd_ps(mv, zk, zi); /* zi -= m * zk */
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }
                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* Divide the row block by the diagonal (scalar) */
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
        /* Portable scalar forward solve */
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

    /* --------------------------------------------------------------
     * Backward solve:  S_y · K = Z           (S_y upper; overwrite Z→K)
     * Solve row-by-row (i descending). Block RHS columns as above.
     * -------------------------------------------------------------- */
    if (ukf_has_avx2() && n >= 8)
    {
        for (int i = (int)n - 1; i >= 0; --i)
        {
            const float sii = Sy[(size_t)i * n + i];

            for (uint16_t c0 = 0; c0 < n; c0 += UKF_UPD_COLBLOCK)
            {
                const uint16_t bc = (uint16_t)((c0 + UKF_UPD_COLBLOCK <= n)
                                    ? UKF_UPD_COLBLOCK : (n - c0));

                /* Z[i,*] -= Σ_{k>i} Sy[i,k] * Z[k,*] */
                for (uint16_t k = (uint16_t)(i + 1); k < n; ++k)
                {
                    const float m = Sy[(size_t)i * n + k];
                    if (m == 0.0f) continue;
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
                        zi = _mm256_fnmadd_ps(mv, zk, zi); /* zi -= m * zk */
                        _mm256_storeu_ps(&Z[(size_t)i * n + c0 + c], zi);
                    }
                    for (; c < bc; ++c)
                        Z[(size_t)i * n + c0 + c] -= m * Z[(size_t)k * n + c0 + c];
                }

                /* Divide the row block by the diagonal */
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
        /* Portable scalar backward solve */
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
    /* Z now holds K (n×n). */

    /* --------------------------------------------------------------
     * Innovation: v = y − ŷ
     * -------------------------------------------------------------- */
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8))
        {
            if (do_pf && i + pf_elts + 8 < n)
            {
                _mm_prefetch((const char *)(y    + i + pf_elts), _MM_HINT_T0);
                _mm_prefetch((const char *)(yhat + i + pf_elts), _MM_HINT_T0);
            }
            __m256 vy  = _mm256_loadu_ps(y    + i);
            __m256 vyh = _mm256_loadu_ps(yhat + i);
            _mm256_storeu_ps(yyhat + i, _mm256_sub_ps(vy, vyh));
        }
        for (; i < n; ++i) yyhat[i] = y[i] - yhat[i];
    }
    else
    {
        for (uint16_t i = 0; i < n; ++i) yyhat[i] = y[i] - yhat[i];
    }

    /* Gain-times-innovation: Ky = K * (y − ŷ) */
    if (mul(Ky, Z /*K*/, yyhat, n, n, n, 1) != 0)
        return -EIO;

    /* State update: x̂ ← x̂ + Ky */
    if (ukf_has_avx2() && n >= 8)
    {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8))
        {
            __m256 xv = _mm256_loadu_ps(xhat + i);
            __m256 kv = _mm256_loadu_ps(Ky   + i);
            _mm256_storeu_ps(xhat + i, _mm256_add_ps(xv, kv));
        }
        for (; i < n; ++i) xhat[i] += Ky[i];
    }
    else
    {
        for (uint16_t i = 0; i < n; ++i) xhat[i] += Ky[i];
    }

    /* Helper matrix: U = K * S_y (n×n) */
    if (mul(U, Z /*K*/, Sy, n, n, n, n) != 0)
        return -EIO;

    /* Downdate S (upper) by each column of U */
    for (uint16_t j = 0; j < n; ++j)
    {
        /* Prefetch next column (optional) */
        if (do_pf && (uint16_t)(j + 1) < n)
            _mm_prefetch((const char *)(&U[(size_t)0 * n + (j + 1)]), _MM_HINT_T0);

        /* Extract column j of U into contiguous vector */
        for (uint16_t i = 0; i < n; ++i)
            Uk[i] = U[(size_t)i * n + j];

        /* Downdate S with column j */
        int rc = cholupdate(S, Uk, n, /*is_upper=*/true, /*rank_one_update=*/false);
        if (rc != 0)
        {
            /* Downdate failed: measurement update caused filter divergence.
            This can happen if:
            - Measurement is wildly inconsistent (outlier)
            - R is too small (overconfident measurement model)
            - Numerical issues accumulated                              */
            return rc;  /* Let caller handle divergence */
        }
    }

    return 0;
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
int sqr_ukf(float y[], float xhat[],
            const float Rn_sr[], const float Rv_sr[], /* upper SR noise */
            float u[],
            void (*F)(float[], float[], float[]),
            float S[], float alpha, float beta, uint8_t L8)
{
    if (L8 == 0)
        return -EINVAL;

    int status = 0;
    const uint16_t L = L8;
    const uint16_t N = (uint16_t)(2 * L + 1);

    const size_t szW = (size_t)N * sizeof(float);
    const size_t szLN = (size_t)L * N * sizeof(float);
    const size_t szLL = (size_t)L * L * sizeof(float);
    const size_t szL = (size_t)L * sizeof(float);

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
        status = -ENOMEM;
        goto Cleanup;
    }

    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, (uint8_t)L);

    // 2) sigma → X  (state)
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);
    
    // 3) propagate through F → X*
    compute_transition_function(Xst, X, u, F, (uint8_t)L);

    // 4) predicted state mean
    multiply_sigma_point_matrix_to_weights(xhat, Xst, Wm, (uint8_t)L);

    {
        // 5) predicted SR covariance (upper): uses QR + upper chol update/downdate
        int rc = create_state_estimation_error_covariance_matrix(S, Wc, Xst, xhat, Rv_sr, (uint8_t)L);
        if (rc)
        {
            status = rc;
            goto Cleanup;
        }
    }
    // 6) re-sigma from new (x̂, S)
    create_sigma_point_matrix(X, xhat, S, alpha, kappa, (uint8_t)L);
    // 7) identity measurement model: Y := X
    H(Y, X, (uint8_t)L);

    // 8) predicted measurement mean
    multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, (uint8_t)L);

    {
        // 9) measurement SR covariance (upper)
        int rc = create_state_estimation_error_covariance_matrix(Sy, Wc, Y, yhat, Rn_sr, (uint8_t)L);
        if (rc)
        {
            status = rc;
            goto Cleanup;
        }
    }
    // 10) cross-covariance P_xy
    create_state_cross_covariance_matrix(Pxy, Wc, X, Y, xhat, yhat, (uint8_t)L);

    {
        // 11) SR update (upper): gain, x̂ += K(y−ŷ), S downdate by columns of U=K·S_y
        int rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y, Sy, Pxy, (uint8_t)L);
        if (rc)
        {
            status = rc;
            goto Cleanup;
        }
    }

Cleanup:
    ukf_aligned_free(Wc);
    ukf_aligned_free(Wm);
    ukf_aligned_free(X);
    ukf_aligned_free(Xst);
    ukf_aligned_free(Y);
    ukf_aligned_free(yhat);
    ukf_aligned_free(Sy);
    ukf_aligned_free(Pxy);
    return status;
}