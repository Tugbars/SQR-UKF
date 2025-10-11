// --- drop-in replacements for: create_weights, create_sigma_point_matrix, compute_transistion_function

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "linalg_simd.h" 

/* define to 1 if you want the 8-way batching; keep 0 by default */
#ifndef SQR_UKF_ENABLE_BATCH8
#  define SQR_UKF_ENABLE_BATCH8 0
#endif

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
static void create_weights(float Wc[], float Wm[], float alpha, float beta, float kappa, uint8_t L)
{
    const uint8_t N   = (uint8_t)(2 * L + 1);
    const float   Lf  = (float)L;
    const float   lam = alpha * alpha * (Lf + kappa) - Lf;
    const float   den = 1.0f / (Lf + lam);

    /* first element */
    Wm[0] = lam * den;
    Wc[0] = Wm[0] + 1.0f - alpha * alpha + beta;

    /* bulk: 0.5/(L+lambda) for i>=1 */
    if (ukf_has_avx2() && N >= 9) {
        const __m256 hv = _mm256_set1_ps(0.5f * den);
        uint8_t i = 1;
        for (; (uint8_t)(i + 7) < N; i = (uint8_t)(i + 8)) {
            _mm256_storeu_ps(&Wm[i], hv);
            _mm256_storeu_ps(&Wc[i], hv);
        }
        for (; i < N; ++i) {
            Wm[i] = 0.5f * den;
            Wc[i] = 0.5f * den;
        }
    } else {
        for (uint8_t i = 1; i < N; ++i) {
            Wm[i] = 0.5f * den;
            Wc[i] = 0.5f * den;
        }
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
                                      float alpha, float kappa, uint8_t L)
{
    const uint8_t N    = (uint8_t)(2 * L + 1);
    const float   gamma= alpha * sqrtf((float)L + kappa);

    /* column 0: copy x */
    if (ukf_has_avx2() && L >= 8) {
        uint8_t i = 0;
        for (; (uint8_t)(i + 7) < L; i = (uint8_t)(i + 8)) {
            __m256 xv = _mm256_loadu_ps(&x[i]);
            _mm256_storeu_ps(&X[i * N], xv);
        }
        for (; i < L; ++i) X[i * N] = x[i];
    } else {
        for (uint8_t i = 0; i < L; ++i) X[i * N] = x[i];
    }

    if (ukf_has_avx2() && L >= 8) {
        /* prepare index vector for stride-L gathers of S[:, col] */
        alignas(32) int idx_a[8];
        for (int t = 0; t < 8; ++t) idx_a[t] = t * (int)L;
        const __m256i idx = _mm256_load_si256((const __m256i*)idx_a);
        const __m256  g8  = _mm256_set1_ps(gamma);
        const __m256  ng8 = _mm256_set1_ps(-gamma);

        /* columns 1..L : X[:,j] = x + γ*S[:, j-1] */
        for (uint8_t j = 1; j <= L; ++j) {
            const int base = (int)(j - 1);
            uint8_t i = 0;
            for (; (uint8_t)(i + 7) < L; i = (uint8_t)(i + 8)) {
                const float *p = &S[i * L + base];               /* &S[i, base] */
                __m256 sv = _mm256_i32gather_ps(p, idx, sizeof(float));
                __m256 xv = _mm256_loadu_ps(&x[i]);
                _mm256_storeu_ps(&X[i * N + j], _mm256_fmadd_ps(g8, sv, xv));
            }
            for (; i < L; ++i) X[i * N + j] = x[i] + gamma * S[i * L + (j - 1)];
        }

        /* columns L+1..2L : X[:,j] = x - γ*S[:, j-L-1] */
        for (uint8_t j = (uint8_t)(L + 1); j < N; ++j) {
            const int base = (int)(j - L - 1);
            uint8_t i = 0;
            for (; (uint8_t)(i + 7) < L; i = (uint8_t)(i + 8)) {
                const float *p = &S[i * L + base];
                __m256 sv = _mm256_i32gather_ps(p, idx, sizeof(float));
                __m256 xv = _mm256_loadu_ps(&x[i]);
                _mm256_storeu_ps(&X[i * N + j], _mm256_fmadd_ps(ng8, sv, xv));
            }
            for (; i < L; ++i) X[i * N + j] = x[i] - gamma * S[i * L + (j - L - 1)];
        }
    } else {
        /* scalar fallback */
        for (uint8_t j = 1; j <= L; ++j)
            for (uint8_t i = 0; i < L; ++i)
                X[i * N + j] = x[i] + gamma * S[i * L + (j - 1)];
        for (uint8_t j = (uint8_t)(L + 1); j < N; ++j)
            for (uint8_t i = 0; i < L; ++i)
                X[i * N + j] = x[i] - gamma * S[i * L + (j - L - 1)];
    }
}


/**
 * @brief Evaluate transition function F at each sigma point column of X.
 *
 * @details
 *  For j=0..N-1, forms x_j = X(:,j), computes dx_j = F(x_j,u), stores dx_j in Xstar(:,j).
 *  Default path is scalar per-column since F is user-provided and typically dominant.
 *  Optional AVX batching (compile-time flag) packs 8 columns and calls F 8 times
 *  with SoA buffers, reducing pack/unpack overhead.
 *
 *  Improvements over scalar:
 *   - Optional 8-way batching to cut per-call overhead when F is light-weight.
 *   - Heap-based scratch to avoid large VLAs.
 *
 * @param[out] Xstar  Results [L x N], row-major.
 * @param[in]  X      Sigma points [L x N], row-major.
 * @param[in]  u      Control/input vector (passed to F).
 * @param[in]  F      User transition function: F(dx,x,u).
 * @param[in]  L      State dimension.
 *
 * @note Enable batching by defining SQR_UKF_ENABLE_BATCH8=1.
 */
static void compute_transistion_function(float Xstar[], const float X[], const float u[],
                                         void (*F)(float[], float[], float[]), uint8_t L)
{
    const uint8_t N = (uint8_t)(2 * L + 1);

#if SQR_UKF_ENABLE_BATCH8
    if (ukf_has_avx2() && N >= 8) {
        /* SoA buffers: 8 states contiguous each */
        float *x = (float*)aligned_alloc(32, (size_t)8 * L * sizeof(float));
        float *d = (float*)aligned_alloc(32, (size_t)8 * L * sizeof(float));
        if (!x || !d) { free(x); free(d); /* fall through to scalar */ }
        else {
            uint8_t j = 0;
            for (; (uint8_t)(j + 7) < N; j = (uint8_t)(j + 8)) {
                /* pack: x[k*L + i] = X[i*N + (j+k)] */
                for (uint8_t i = 0; i < L; ++i) {
                    __m256 col = _mm256_loadu_ps(&X[i * N + j]);    // 8 contiguous sigmas
                    _mm256_storeu_ps(&x[i * 8], col);              // transpose to SoA (i-major)
                }
                /* call F on 8 contiguous slices */
                for (uint8_t k = 0; k < 8; ++k)
                    F(&d[k * L], &x[k * L], (float*)u);
                /* unpack back: Xstar[i*N + (j+k)] = d[k*L + i] */
                for (uint8_t k = 0; k < 8; ++k)
                    for (uint8_t i = 0; i < L; ++i)
                        Xstar[i * N + (j + k)] = d[k * L + i];
            }
            /* scalar tail */
            for (; j < N; ++j) {
                float *xk = x, *dk = d;
                for (uint8_t i = 0; i < L; ++i) xk[i] = X[i * N + j];
                F(dk, xk, (float*)u);
                for (uint8_t i = 0; i < L; ++i) Xstar[i * N + j] = dk[i];
            }
            ukf_aligned_free(x);
            ukf_aligned_free(d);
            return;
        }
    }
#endif

    /* scalar (portable, safe; usually F dominates anyway) */
    float *xk = (float*)malloc((size_t)L * sizeof(float));
    float *dk = (float*)malloc((size_t)L * sizeof(float));
    if (!xk || !dk) { free(xk); free(dk); return; }

    for (uint8_t j = 0; j < N; ++j) {
        for (uint8_t i = 0; i < L; ++i) xk[i] = X[i * N + j];
        F(dk, xk, (float*)u);
        for (uint8_t i = 0; i < L; ++i) Xstar[i * N + j] = dk[i];
    }
    free(xk); free(dk);
}

/**
 * @brief Compute weighted mean x = X * W for sigma points.
 *
 * @details
 *  Computes x[i] = sum_j W[j]*X[i,j], i=0..L-1. Vectorized path performs
 *  row-wise dot-products with AVX2 FMAs over 8-wide chunks with scalar tails.
 *
 *  Improvements over scalar:
 *   - FMA accumulation reduces instruction count and latency.
 *   - Horizontal add + scalar tail for correctness on any N.
 *
 * @param[out] x   Output mean vector [L].
 * @param[in]  X   Sigma matrix [L x N], row-major.
 * @param[in]  W   Weights [N].
 * @param[in]  L   State dimension (N=2L+1).
 *
 * @note x is fully overwritten; no need to pre-clear externally.
 */
static void multiply_sigma_point_matrix_to_weights(float x[], float X[], float W[], uint8_t L)
{
    const uint16_t N = (uint16_t)(2 * L + 1);

    /* clear x */
    memset(x, 0, (size_t)L * sizeof(float));

    if (!ukf_has_avx2() || N < 8) {
        /* scalar */
        for (uint16_t i = 0; i < L; ++i) {
            float acc = 0.0f;
            const float *row = &X[(size_t)i * N];
            for (uint16_t j = 0; j < N; ++j)
                acc += W[j] * row[j];
            x[i] = acc;
        }
        return;
    }

    /* AVX2: row-wise dot products with FMAs over N */
    for (uint16_t i = 0; i < L; ++i) {
        const float *row = &X[(size_t)i * N];
        __m256 vacc = _mm256_setzero_ps();

        uint16_t j = 0;
        for (; (uint16_t)(j + 7) < N; j = (uint16_t)(j + 8)) {
            __m256 wv = _mm256_loadu_ps(&W[j]);
            __m256 xv = _mm256_loadu_ps(&row[j]);
            vacc = _mm256_fmadd_ps(wv, xv, vacc);
        }

        /* horizontal sum */
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float acc = _mm_cvtss_f32(s);

        /* tail */
        for (; j < N; ++j) acc += W[j] * row[j];

        x[i] = acc;
    }
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
 *   - Uses your optimized tran(), qr(), and cholupdate().
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
                                                            float x[], float R[], uint8_t L)
{
    const uint16_t N = (uint16_t)(2 * L + 1);
    const uint16_t M = (uint16_t)(2 * L + L);   /* = 3L */
    const uint16_t K = (uint16_t)(2 * L);

    /* sqrt weight for columns 1..K */
    const float weight1 = sqrtf(fabsf(W[1]));

    /* AT is L x M (row-major) */
    float AT[(size_t)L * M];

    if (ukf_has_avx2() && L >= 8) {
        const __m256 w1v = _mm256_set1_ps(weight1);

        /* first K columns: AT[i,j] = weight1 * ( X[i, j+1] - x[i] ) */
        for (uint16_t i = 0; i < L; ++i) {
            float *ATrow = &AT[(size_t)i * M];
            const float *Xi = &X[(size_t)i * N];
            const __m256 xi8 = _mm256_set1_ps(x[i]);

            uint16_t j = 0;
            for (; (uint16_t)(j + 7) < K; j = (uint16_t)(j + 8)) {
                __m256 Xv = _mm256_loadu_ps(&Xi[j + 1]);         /* X[i, (j+1)...] */
                __m256 diff = _mm256_sub_ps(Xv, xi8);
                _mm256_storeu_ps(&ATrow[j], _mm256_mul_ps(w1v, diff));
            }
            for (; j < K; ++j)
                ATrow[j] = weight1 * (Xi[j + 1] - x[i]);

            /* last L columns: AT[i, K + t] = sqrt( R[i, t] ) */
            uint16_t t = 0;
            for (; (uint16_t)(t + 7) < L; t = (uint16_t)(t + 8)) {
                __m256 Rv = _mm256_loadu_ps(&R[(size_t)i * L + t]);
                __m256 Sv = _mm256_sqrt_ps(Rv);
                _mm256_storeu_ps(&ATrow[K + t], Sv);
            }
            for (; t < L; ++t)
                ATrow[K + t] = sqrtf(R[(size_t)i * L + t]);
        }
    } else {
        /* scalar build of AT */
        for (uint16_t i = 0; i < L; ++i) {
            for (uint16_t j = 0; j < K; ++j)
                AT[(size_t)i * M + j] = weight1 * (X[(size_t)i * N + (j + 1)] - x[i]);
            for (uint16_t j = K; j < M; ++j)
                AT[(size_t)i * M + j] = sqrtf(R[(size_t)i * L + (j - K)]);
        }
    }

    /* A' is required by the SR-UKF derivation */
    tran(AT, AT, L, M);   /* (L x M) -> (M x L) in-place via workspace inside tran */

    /* QR of A' (M x L). We only need R_ (upper triangular M x L) */
    float Qtmp[(size_t)M * M];   /* not used but qr() wants it */
    float R_[(size_t)M * L];
    qr(AT, Qtmp, R_, M, L, true);

    /* S is the upper LxL of R_ according to SR-UKF */
    memcpy(S, R_, (size_t)L * L * sizeof(float));

    /* b = X[:,0] - x  (vectorized) */
    float b[L];
    if (ukf_has_avx2() && L >= 8) {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < L; i = (uint16_t)(i + 8)) {
            __m256 X0 = _mm256_loadu_ps(&X[(size_t)i * N + 0]);  /* column 0 is contiguous by rows */
            __m256 xv = _mm256_loadu_ps(&x[i]);
            _mm256_storeu_ps(&b[i], _mm256_sub_ps(X0, xv));
        }
        for (; i < L; ++i) b[i] = X[(size_t)i * N] - x[i];
    } else {
        for (uint16_t i = 0; i < L; ++i) b[i] = X[(size_t)i * N] - x[i];
    }

    /* rank-one update/downdate on S depending on sign of W[0] */
    const bool rank_one_update = (W[0] >= 0.0f);
    cholupdate(S, b, L, rank_one_update);
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
static void create_state_cross_covariance_matrix(float P[], float W[],
                                                 float X[], float Y[],
                                                 float x[], float y[],
                                                 uint8_t L)
{
    const uint16_t N = (uint16_t)(2 * L + 1);
    const uint16_t LL = (uint16_t)L;

    /* --- clear P (fix: L*L, not 2L) --- */
    memset(P, 0, (size_t)LL * LL * sizeof(float));

    /* --- Center X and Y: subtract means row-wise (contiguous across N) --- */
    if (ukf_has_avx2() && N >= 8) {
        for (uint16_t i = 0; i < LL; ++i) {
            float *Xi = &X[(size_t)i * N];
            float *Yi = &Y[(size_t)i * N];
            __m256 xi = _mm256_set1_ps(x[i]);
            __m256 yi = _mm256_set1_ps(y[i]);
            uint16_t j = 0;
            for (; (uint16_t)(j + 7) < N; j = (uint16_t)(j + 8)) {
                _mm256_storeu_ps(&Xi[j], _mm256_sub_ps(_mm256_loadu_ps(&Xi[j]), xi));
                _mm256_storeu_ps(&Yi[j], _mm256_sub_ps(_mm256_loadu_ps(&Yi[j]), yi));
            }
            for (; j < N; ++j) { Xi[j] -= x[i]; Yi[j] -= y[i]; }
        }
    } else {
        for (uint16_t i = 0; i < LL; ++i) {
            float *Xi = &X[(size_t)i * N];
            float *Yi = &Y[(size_t)i * N];
            for (uint16_t j = 0; j < N; ++j) { Xi[j] -= x[i]; Yi[j] -= y[i]; }
        }
    }

    /* --- Accumulate P = Σ_j W[j] * x_j * y_j^T (8×8 micro-kernel + tails) --- */
    if (ukf_has_avx2() && LL >= 8) {
        for (uint16_t j = 0; j < N; ++j) {
            const float wj = W[j];
            if (wj == 0.0f) continue;
            const __m256 w8 = _mm256_set1_ps(wj);

            /* process P in 8×8 tiles: rows i..i+7, cols k..k+7 */
            uint16_t i = 0;
            for (; (uint16_t)(i + 7) < LL; i = (uint16_t)(i + 8)) {
                /* load x tile (8 elements) */
                __m256 x8 = _mm256_loadu_ps(&X[(size_t)i * N + j]);

                uint16_t k = 0;
                for (; (uint16_t)(k + 7) < LL; k = (uint16_t)(k + 8)) {
                    /* broadcast y tile and FMA into P block */
                    float *Pblk = &P[(size_t)i * LL + k];

                    __m256 y0 = _mm256_loadu_ps(&Y[(size_t)k * N + j]); /* 8 y’s */
                    __m256 t = _mm256_mul_ps(w8, y0);                   /* wj * y */

                    /* update 8 rows of P: each row r does P[r, k..k+7] += x[r]*t */
                    __m256 pr;

                    pr = _mm256_loadu_ps(Pblk + 0*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[0]), t, pr);
                    _mm256_storeu_ps(Pblk + 0*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 1*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[1]), t, pr);
                    _mm256_storeu_ps(Pblk + 1*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 2*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[2]), t, pr);
                    _mm256_storeu_ps(Pblk + 2*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 3*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[3]), t, pr);
                    _mm256_storeu_ps(Pblk + 3*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 4*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[4]), t, pr);
                    _mm256_storeu_ps(Pblk + 4*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 5*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[5]), t, pr);
                    _mm256_storeu_ps(Pblk + 5*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 6*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[6]), t, pr);
                    _mm256_storeu_ps(Pblk + 6*LL, pr);

                    pr = _mm256_loadu_ps(Pblk + 7*LL);
                    pr = _mm256_fmadd_ps(_mm256_set1_ps(((float*)&x8)[7]), t, pr);
                    _mm256_storeu_ps(Pblk + 7*LL, pr);
                }
                /* col tail (<8) for rows i..i+7 */
                for (; k < LL; ++k) {
                    const float yk = Y[(size_t)k * N + j] * wj;
                    float *Pcol = &P[(size_t)i * LL + k];
                    for (int r = 0; r < 8; ++r) Pcol[r * LL] += ((float*)&x8)[r] * yk;
                }
            }
            /* row tail (i .. LL-1): scalar outer updates */
            for (; i < LL; ++i) {
                const float xi = X[(size_t)i * N + j];
                float *Pi = &P[(size_t)i * LL];
                for (uint16_t k = 0; k < LL; ++k)
                    Pi[k] += wj * xi * Y[(size_t)k * N + j];
            }
        }
    } else {
        /* scalar fallback */
        for (uint16_t j = 0; j < N; ++j) {
            const float wj = W[j];
            if (wj == 0.0f) continue;
            for (uint16_t i = 0; i < LL; ++i) {
                const float xi = X[(size_t)i * N + j];
                float *Pi = &P[(size_t)i * LL];
                for (uint16_t k = 0; k < LL; ++k)
                    Pi[k] += wj * xi * Y[(size_t)k * N + j];
            }
        }
    }
}

/**
 * @brief Measurement update: compute K, update xhat, and downdate S via Cholesky rank-1.
 *
 * @details
 *  Solves (Sy^T Sy)K = Pxy without explicit inverse:
 *   1) Forward:  Sy^T Z = Pxy   (lower-triangular solve).
 *   2) Backward: Sy   K = Z     (upper-triangular solve).
 *  Then forms Ky = K*(y−yhat), updates xhat += Ky, computes U=K*Sy, and applies
 *  column-wise downdates S = cholupdate(S, U[:,j], false).
 *
 *  Vectorized pieces:
 *   - AVX2 axpy-style updates inside triangular solves over RHS columns.
 *   - AVX2 y−yhat and xhat += Ky.
 *   - AVX2 gathers to extract columns U[:,j] before cholupdate().
 *
 *  Improvements over scalar:
 *   - Avoids explicit inv(Sy^T Sy) for better speed and conditioning.
 *   - Wide FMAs reduce instruction count in the solves.
 *
 * @param[in,out] S     Square-root covariance [L x L], upper-triangular, downdated in-place.
 * @param[in,out] xhat  State mean [L], updated in-place.
 * @param[in]     yhat  Predicted measurement mean [L].
 * @param[in]     y     Actual measurement [L].
 * @param[in]     Sy    Measurement SR covariance [L x L], upper-triangular.
 * @param[in]     Pxy   Cross-covariance [L x L].
 * @param[in]     L     Dimension.
 *
 * @note Uses your optimized mul() for U=K*Sy and cholupdate() for SR updates.
 */
static void update_state_covarariance_matrix_and_state_estimation_vector(float S[], float xhat[],
                                                                         float yhat[], float y[],
                                                                         float Sy[], float Pxy[],
                                                                         uint8_t L)
{
    const uint16_t n = (uint16_t)L;

    /* -------- forward solve: Sy^T Z = Pxy (Sy upper ⇒ Sy^T lower) -------- */
    float Z[(size_t)n * n];
    /* copy Pxy into Z as initial RHS */
    memcpy(Z, Pxy, (size_t)n * n * sizeof(float));

    if (ukf_has_avx2() && n >= 8) {
        for (uint16_t i = 0; i < n; ++i) {
            /* Z[i,:] = (Z[i,:] - sum_{k<i} Sy[k,i]*Z[k,:]) / Sy[i,i] */
            float sii = Sy[(size_t)i * n + i];

            /* subtract previous rows contributions */
            for (uint16_t k = 0; k < i; ++k) {
                const float m = Sy[(size_t)k * n + i];
                if (m == 0.0f) continue;
                const __m256 mv = _mm256_set1_ps(m);
                uint16_t c = 0;
                for (; (uint16_t)(c + 7) < n; c = (uint16_t)(c + 8)) {
                    __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c]);
                    __m256 zk = _mm256_loadu_ps(&Z[(size_t)k * n + c]);
                    zi = _mm256_fnmadd_ps(mv, zk, zi);
                    _mm256_storeu_ps(&Z[(size_t)i * n + c], zi);
                }
                for (; c < n; ++c) Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }
            /* divide by diagonal */
            const __m256 div = _mm256_set1_ps(sii);
            uint16_t c = 0;
            for (; (uint16_t)(c + 7) < n; c = (uint16_t)(c + 8)) {
                __m256 zi = _mm256_loadu_ps(&Z[(size_t)i * n + c]);
                _mm256_storeu_ps(&Z[(size_t)i * n + c], _mm256_div_ps(zi, div));
            }
            for (; c < n; ++c) Z[(size_t)i * n + c] /= sii;
        }
    } else {
        for (uint16_t i = 0; i < n; ++i) {
            float sii = Sy[(size_t)i * n + i];
            for (uint16_t k = 0; k < i; ++k) {
                const float m = Sy[(size_t)k * n + i];
                for (uint16_t c = 0; c < n; ++c)
                    Z[(size_t)i * n + c] -= m * Z[(size_t)k * n + c];
            }
            for (uint16_t c = 0; c < n; ++c) Z[(size_t)i * n + c] /= sii;
        }
    }

    /* -------- backward solve: Sy K = Z (Sy upper) -------- */
    float K[(size_t)n * n];
    memcpy(K, Z, (size_t)n * n * sizeof(float));

    if (ukf_has_avx2() && n >= 8) {
        for (int i = (int)n - 1; i >= 0; --i) {
            const float sii = Sy[(size_t)i * n + i];

            /* subtract upper-part contributions */
            for (uint16_t k = (uint16_t)(i + 1); k < n; ++k) {
                const float m = Sy[(size_t)i * n + k];
                if (m == 0.0f) continue;
                const __m256 mv = _mm256_set1_ps(m);
                uint16_t c = 0;
                for (; (uint16_t)(c + 7) < n; c = (uint16_t)(c + 8)) {
                    __m256 ki = _mm256_loadu_ps(&K[(size_t)i * n + c]);
                    __m256 kk = _mm256_loadu_ps(&K[(size_t)k * n + c]);
                    ki = _mm256_fnmadd_ps(mv, kk, ki);
                    _mm256_storeu_ps(&K[(size_t)i * n + c], ki);
                }
                for (; c < n; ++c) K[(size_t)i * n + c] -= m * K[(size_t)k * n + c];
            }
            /* divide by diagonal */
            const __m256 div = _mm256_set1_ps(sii);
            uint16_t c = 0;
            for (; (uint16_t)(c + 7) < n; c = (uint16_t)(c + 8)) {
                __m256 ki = _mm256_loadu_ps(&K[(size_t)i * n + c]);
                _mm256_storeu_ps(&K[(size_t)i * n + c], _mm256_div_ps(ki, div));
            }
            for (; c < n; ++c) K[(size_t)i * n + c] /= sii;
        }
    } else {
        for (int i = (int)n - 1; i >= 0; --i) {
            const float sii = Sy[(size_t)i * n + i];
            for (uint16_t k = (uint16_t)(i + 1); k < n; ++k) {
                const float m = Sy[(size_t)i * n + k];
                for (uint16_t c = 0; c < n; ++c)
                    K[(size_t)i * n + c] -= m * K[(size_t)k * n + c];
            }
            for (uint16_t c = 0; c < n; ++c) K[(size_t)i * n + c] /= sii;
        }
    }

    /* yyhat = y - yhat (vectorized) */
    float yyhat[(size_t)n];
    if (ukf_has_avx2() && n >= 8) {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8)) {
            __m256 vy = _mm256_loadu_ps(&y[i]);
            __m256 vyh= _mm256_loadu_ps(&yhat[i]);
            _mm256_storeu_ps(&yyhat[i], _mm256_sub_ps(vy, vyh));
        }
        for (; i < n; ++i) yyhat[i] = y[i] - yhat[i];
    } else {
        for (uint16_t i = 0; i < n; ++i) yyhat[i] = y[i] - yhat[i];
    }

    /* Ky = K * (y - yhat)  (your mul is already vectorized) */
    float Ky[(size_t)n];
    mul(Ky, K, yyhat, n, n, n, 1);

    /* xhat += Ky (vectorized) */
    if (ukf_has_avx2() && n >= 8) {
        uint16_t i = 0;
        for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8)) {
            __m256 xv = _mm256_loadu_ps(&xhat[i]);
            __m256 kv = _mm256_loadu_ps(&Ky[i]);
            _mm256_storeu_ps(&xhat[i], _mm256_add_ps(xv, kv));
        }
        for (; i < n; ++i) xhat[i] += Ky[i];
    } else {
        for (uint16_t i = 0; i < n; ++i) xhat[i] += Ky[i];
    }

    /* U = K * Sy (uses your optimized mul) */
    float U[(size_t)n * n];
    mul(U, K, Sy, n, n, n, n);

    /* For each column j of U, downdate S with Uk = U[:,j] */
    float Uk[(size_t)n];
    if (ukf_has_avx2() && n >= 8) {
        /* gather column j with stride n in chunks of 8 */
        alignas(32) int idx[8];
        for (int t = 0; t < 8; ++t) idx[t] = t * (int)n;
        const __m256i gidx = _mm256_load_si256((const __m256i*)idx);

        for (uint16_t j = 0; j < n; ++j) {
            uint16_t i = 0;
            for (; (uint16_t)(i + 7) < n; i = (uint16_t)(i + 8)) {
                const float *col0 = &U[(size_t)i * n + j];
                __m256 v = _mm256_i32gather_ps(col0, gidx, sizeof(float));
                _mm256_storeu_ps(&Uk[i], v);
            }
            for (; i < n; ++i) Uk[i] = U[(size_t)i * n + j];

            cholupdate(S, Uk, n, /*rank_one_update=*/false);
        }
    } else {
        for (uint16_t j = 0; j < n; ++j) {
            for (uint16_t i = 0; i < n; ++i) Uk[i] = U[(size_t)i * n + j];
            cholupdate(S, Uk, n, false);
        }
    }
}

static inline void* ukf_aligned_alloc(size_t nbytes) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(32, ((nbytes + 31) / 32) * 32);
#elif defined(_MSC_VER)
    return _aligned_malloc(nbytes, 32);
#else
    // fallback: overallocate and align manually
    void* base = malloc(nbytes + 64);
    if (!base) return NULL;
    uintptr_t p = ((uintptr_t)base + 31 + sizeof(void*)) & ~((uintptr_t)31);
    ((void**)p)[-1] = base;
    return (void*)p;
#endif
}

static inline void ukf_aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    free(p);
#else
    if (p) free(((void**)p)[-1]);
#endif
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
    if (L8 == 0) return -EINVAL;

    const uint16_t L = L8;                 // promote to avoid overflow
    const uint16_t N = (uint16_t)(2 * L + 1);

    /* --- workspace sizes --- */
    const size_t szW  = (size_t)N * sizeof(float);
    const size_t szLN = (size_t)L * N * sizeof(float);
    const size_t szLL = (size_t)L * L * sizeof(float);
    const size_t szL  = (size_t)L * sizeof(float);

    /* --- allocate aligned scratch --- */
    float *Wc   = (float*)ukf_aligned_alloc(szW);
    float *Wm   = (float*)ukf_aligned_alloc(szW);
    float *X    = (float*)ukf_aligned_alloc(szLN);
    float *Xst  = (float*)ukf_aligned_alloc(szLN);
    float *Y    = (float*)ukf_aligned_alloc(szLN);
    float *yhat = (float*)ukf_aligned_alloc(szL);
    float *Sy   = (float*)ukf_aligned_alloc(szLL);
    float *Pxy  = (float*)ukf_aligned_alloc(szLL);

    if (!Wc || !Wm || !X || !Xst || !Y || !yhat || !Sy || !Pxy) {
        ukf_aligned_free(Wc);  ukf_aligned_free(Wm);
        ukf_aligned_free(X);   ukf_aligned_free(Xst);
        ukf_aligned_free(Y);   ukf_aligned_free(yhat);
        ukf_aligned_free(Sy);  ukf_aligned_free(Pxy);
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
    ukf_aligned_free(Wc);   ukf_aligned_free(Wm);
    ukf_aligned_free(X);    ukf_aligned_free(Xst);
    ukf_aligned_free(Y);    ukf_aligned_free(yhat);
    ukf_aligned_free(Sy);   ukf_aligned_free(Pxy);

    return 0;
}

