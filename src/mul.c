// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <immintrin.h>
#include <string.h>
#include "linalg_simd.h" // linalg_has_avx2(), linalg_aligned_alloc(), linalg_aligned_free(), LINALG_DEFAULT_ALIGNMENT, RESTRICT

#ifndef LINALG_BLOCK_MC
#define LINALG_BLOCK_MC 128
#endif
#ifndef LINALG_BLOCK_KC
#define LINALG_BLOCK_KC 256
#endif
#ifndef LINALG_BLOCK_JC
#define LINALG_BLOCK_JC 256
#endif
#ifndef LINALG_SMALL_N_THRESH
#define LINALG_SMALL_N_THRESH 64
#endif
#ifndef LINALG_GEMM_PF_DIST
#define LINALG_GEMM_PF_DIST 192 /* bytes ahead within a stream: 64–256 */
#endif
#ifndef LINALG_GEMM_PF_ROWS_AHEAD
#define LINALG_GEMM_PF_ROWS_AHEAD 1 /* row-ahead prefetch for packers: 0..2 */
#endif
#ifndef LINALG_GEMM_PF_MIN_K
#define LINALG_GEMM_PF_MIN_K 128 /* enable within-row prefetch if Kblk ≥ */
#endif

/* === Prefetch kill-switch (compile-time) ============================== */
#ifndef LINALG_GEMM_PREFETCH_ENABLE
#define LINALG_GEMM_PREFETCH_ENABLE 1 /* set to 0 to disable all explicit prefetch */
#endif
#if LINALG_GEMM_PREFETCH_ENABLE
#define PREFETCH_T0(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#define PREFETCH_T1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T1)
#else
#define PREFETCH_T0(ptr) ((void)0)
#define PREFETCH_T1(ptr) ((void)0)
#endif

/* ===== helper: AVX2 mask for N-tail (0..8 lanes) ===== */
#if LINALG_SIMD_ENABLE
// Table for 0..8 lanes (-1 = load/store lane; 0 = masked)
static const alignas(64) int8_t kMask8x8[9][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {-1, 0, 0, 0, 0, 0, 0, 0},
    {-1, -1, 0, 0, 0, 0, 0, 0},
    {-1, -1, -1, 0, 0, 0, 0, 0},
    {-1, -1, -1, -1, 0, 0, 0, 0},
    {-1, -1, -1, -1, -1, 0, 0, 0},
    {-1, -1, -1, -1, -1, -1, 0, 0},
    {-1, -1, -1, -1, -1, -1, -1, 0},
    {-1, -1, -1, -1, -1, -1, -1, -1}};

static inline __m256i avx2_tailmask_fast(int lanes /*0..8*/)
{
    __m128i b8 = _mm_loadl_epi64((const __m128i *)kMask8x8[lanes]);
    return _mm256_cvtepi8_epi32(b8);
}

// n = active columns (≤ NR), NR ∈ {6,8}
static inline __m256i avx2_tailmask_nr(size_t n, size_t NR)
{
    size_t lanes = (n <= NR) ? n : NR;
    return avx2_tailmask_fast((int)lanes);
}

// Build a vector from up to n columns taken from a col-major temp buffer.
// stride is 16 for 16x* kernels, 8 for 8x* kernels. Leaves extra lanes 0.
static inline __m256 load_cols_from_temp(const float *temp, size_t stride, size_t r, size_t n)
{
    // n ∈ [0..8]
    alignas(32) float lane[8] = {0};
    // temp[j*stride + r] is row r, column j (col-major in temp)
    for (size_t j = 0; j < n; ++j)
        lane[j] = temp[j * stride + r];
    return _mm256_load_ps(lane);
}
#endif /* LINALG_SIMD_ENABLE */

/* ======================= Packing ======================= */

/* ---- A packers (col-major for ld=16 and ld=8) ---- */
static inline void
pack_A_block_16row_colmajor(float *RESTRICT Ap,
                            const float *RESTRICT A,
                            size_t M, size_t K,
                            size_t i0, size_t ib,
                            size_t kk, size_t Kblk)
{
    (void)M; /* bounds guaranteed by caller */
    if (ib < 16)
    {
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    }
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 16;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
        {
            dst[r] = A[idx];
        }
    }
}

static inline void
pack_A_block_8row_colmajor(float *RESTRICT Ap,
                           const float *RESTRICT A,
                           size_t M, size_t K,
                           size_t i0, size_t ib,
                           size_t kk, size_t Kblk)
{
    (void)M; /* bounds guaranteed by caller */
    if (ib < 8)
    {
        memset(Ap, 0, Kblk * 8 * sizeof(float));
    }
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 8;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
        {
            dst[r] = A[idx];
        }
    }
}

/* Tail packer for <=16 rows (same layout/ld=16) */
static inline void
pack_A_16row_tile(float *RESTRICT Ap,
                  const float *RESTRICT A,
                  size_t M, size_t K,
                  size_t i0, size_t ib,
                  size_t kk, size_t Kblk)
{
    (void)M; /* bounds are guaranteed by caller */
    if (ib < 16)
    {
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    }
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 16;
        /* Row-ahead prefetch in the same (kk+k) column */
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0)
        {
            const size_t i_pf = i0 + (size_t)LINALG_GEMM_PF_ROWS_AHEAD;
            if (i_pf < i0 + ib)
            {
                const float *pfp = A + i_pf * K + (kk + k);
                PREFETCH_T0(pfp);
            }
        }
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
        {
            dst[r] = A[idx];
        }
    }
}

/* ---- B packers (8-column and 6-column panels) ---- */
static inline void
pack_B_8col_tile(float *RESTRICT Bp,
                 const float *RESTRICT B,
                 size_t K, size_t N,
                 size_t kk, size_t Kblk,
                 size_t j0, size_t jb)
{
    const size_t n_panels = (jb + 7) / 8;
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    size_t off = 0;
    for (size_t p = 0, j = j0; p < n_panels; ++p, j += 8)
    {
        const size_t w = (j + 8 <= j0 + jb) ? 8 : (j0 + jb - j);
        const size_t remain = N - j;
        for (size_t k = 0; k < Kblk; ++k)
        {
            const float *src = B + (kk + k) * N + j; /* row-major */
            float *dst = Bp + off + k * 8;
            /* Prefetch neighboring rows and a small intra-row lookahead */
            if (k + 1 < Kblk)
                PREFETCH_T0(src + N);
            if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
                PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j);
            if (jb >= 16 && pf_elts >= 8 && w == 8 && pf_elts <= remain)
                PREFETCH_T0(src + pf_elts);
            /* copy present cols then pad */
            if (w == 8)
            {
                memcpy(dst, src, 8 * sizeof(float));
            }
            else
            {
                size_t t = 0;
                for (; t < w; ++t)
                    dst[t] = src[t];
                for (; t < 8; ++t)
                    dst[t] = 0.0f;
            }
        }
        off += Kblk * 8;
    }
}

static inline void
pack_B_6col_tile(float *RESTRICT Bp,
                 const float *RESTRICT B,
                 size_t K, size_t N,
                 size_t kk, size_t Kblk,
                 size_t j0, size_t jb)
{
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    for (size_t k = 0; k < Kblk; ++k)
    {
        const float *src = B + (kk + k) * N + j0; /* row-major */
        float *dst = Bp + k * 6;
        /* Prefetch neighboring rows and a small intra-row lookahead */
        if (k + 1 < Kblk)
            PREFETCH_T0(src + N);
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
            PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j0);
        const size_t remain = N - j0;
        if (jb >= 16 && pf_elts >= 6 && pf_elts <= remain)
            PREFETCH_T0(src + pf_elts);
        /* copy present cols */
        memcpy(dst, src, jb * sizeof(float));
        if (jb < 6)
        {
            memset(dst + jb, 0, (6 - jb) * sizeof(float));
        }
    }
}

/* Dispatcher used by mul() based on ker->NR */
static inline void
pack_B_tile(float *RESTRICT Bp,
            const float *RESTRICT B,
            size_t K, size_t N,
            size_t kk, size_t Kblk,
            size_t j0, size_t jb,
            size_t NR)
{
    if (NR == 8)
        pack_B_8col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
    else
        pack_B_6col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
}

/* ======================= Micro-kernels (AVX2/FMA) ======================= */

#if LINALG_SIMD_ENABLE

/* ---- 16x8 (add) ---- */
static inline void
gemm_16x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                            const float *RESTRICT Ap,
                            const float *RESTRICT Bp,
                            size_t Kblk, size_t m, size_t n, __m256i mask /* for n<8 */)
{
    __m256 acc_lo0 = _mm256_setzero_ps();
    __m256 acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps();
    __m256 acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps();
    __m256 acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps();
    __m256 acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps();
    __m256 acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps();
    __m256 acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps();
    __m256 acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps();
    __m256 acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 8;

            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);

            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    alignas(32) float temp[16 * 8];
    _mm256_store_ps(temp + 0 * 16, acc_lo0);
    _mm256_store_ps(temp + 0 * 16 + 8, acc_hi0);
    _mm256_store_ps(temp + 1 * 16, acc_lo1);
    _mm256_store_ps(temp + 1 * 16 + 8, acc_hi1);
    _mm256_store_ps(temp + 2 * 16, acc_lo2);
    _mm256_store_ps(temp + 2 * 16 + 8, acc_hi2);
    _mm256_store_ps(temp + 3 * 16, acc_lo3);
    _mm256_store_ps(temp + 3 * 16 + 8, acc_hi3);
    _mm256_store_ps(temp + 4 * 16, acc_lo4);
    _mm256_store_ps(temp + 4 * 16 + 8, acc_hi4);
    _mm256_store_ps(temp + 5 * 16, acc_lo5);
    _mm256_store_ps(temp + 5 * 16 + 8, acc_hi5);
    _mm256_store_ps(temp + 6 * 16, acc_lo6);
    _mm256_store_ps(temp + 6 * 16 + 8, acc_hi6);
    _mm256_store_ps(temp + 7 * 16, acc_lo7);
    _mm256_store_ps(temp + 7 * 16 + 8, acc_hi7);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 16, r, n);
        __m256 old = _mm256_maskload_ps(cr, mask);
        sum = _mm256_add_ps(old, sum);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 16x8 (store) ---- */
static inline void
gemm_16x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                              const float *RESTRICT Ap,
                              const float *RESTRICT Bp,
                              size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc_lo0 = _mm256_setzero_ps();
    __m256 acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps();
    __m256 acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps();
    __m256 acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps();
    __m256 acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps();
    __m256 acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps();
    __m256 acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps();
    __m256 acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps();
    __m256 acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 8;

            __m256 b0 = _mm256_broadcast_ss(b_row + 0);
            __m256 b1 = _mm256_broadcast_ss(b_row + 1);
            __m256 b2 = _mm256_broadcast_ss(b_row + 2);
            __m256 b3 = _mm256_broadcast_ss(b_row + 3);
            acc_lo0 = _mm256_fmadd_ps(a_lo, b0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(a_hi, b0, acc_hi0);
            acc_lo1 = _mm256_fmadd_ps(a_lo, b1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(a_hi, b1, acc_hi1);
            acc_lo2 = _mm256_fmadd_ps(a_lo, b2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(a_hi, b2, acc_hi2);
            acc_lo3 = _mm256_fmadd_ps(a_lo, b3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(a_hi, b3, acc_hi3);

            __m256 b4 = _mm256_broadcast_ss(b_row + 4);
            __m256 b5 = _mm256_broadcast_ss(b_row + 5);
            __m256 b6 = _mm256_broadcast_ss(b_row + 6);
            __m256 b7 = _mm256_broadcast_ss(b_row + 7);
            acc_lo4 = _mm256_fmadd_ps(a_lo, b4, acc_lo4);
            acc_hi4 = _mm256_fmadd_ps(a_hi, b4, acc_hi4);
            acc_lo5 = _mm256_fmadd_ps(a_lo, b5, acc_lo5);
            acc_hi5 = _mm256_fmadd_ps(a_hi, b5, acc_hi5);
            acc_lo6 = _mm256_fmadd_ps(a_lo, b6, acc_lo6);
            acc_hi6 = _mm256_fmadd_ps(a_hi, b6, acc_hi6);
            acc_lo7 = _mm256_fmadd_ps(a_lo, b7, acc_lo7);
            acc_hi7 = _mm256_fmadd_ps(a_hi, b7, acc_hi7);
        }
    }

    alignas(32) float temp[16 * 8];
    _mm256_store_ps(temp + 0 * 16, acc_lo0);
    _mm256_store_ps(temp + 0 * 16 + 8, acc_hi0);
    _mm256_store_ps(temp + 1 * 16, acc_lo1);
    _mm256_store_ps(temp + 1 * 16 + 8, acc_hi1);
    _mm256_store_ps(temp + 2 * 16, acc_lo2);
    _mm256_store_ps(temp + 2 * 16 + 8, acc_hi2);
    _mm256_store_ps(temp + 3 * 16, acc_lo3);
    _mm256_store_ps(temp + 3 * 16 + 8, acc_hi3);
    _mm256_store_ps(temp + 4 * 16, acc_lo4);
    _mm256_store_ps(temp + 4 * 16 + 8, acc_hi4);
    _mm256_store_ps(temp + 5 * 16, acc_lo5);
    _mm256_store_ps(temp + 5 * 16 + 8, acc_hi5);
    _mm256_store_ps(temp + 6 * 16, acc_lo6);
    _mm256_store_ps(temp + 6 * 16 + 8, acc_hi6);
    _mm256_store_ps(temp + 7 * 16, acc_lo7);
    _mm256_store_ps(temp + 7 * 16 + 8, acc_hi7);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 16, r, n);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 8x8 (add) ---- */
static inline void
gemm_8x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc[8];
    for (int j = 0; j < 8; ++j)
        acc[j] = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 8;
            for (int j = 0; j < 8; ++j)
            {
                __m256 bcast = _mm256_broadcast_ss(b_row + j);
                acc[j] = _mm256_fmadd_ps(a, bcast, acc[j]);
            }
        }
    }

    alignas(32) float temp[8 * 8];
    for (int j = 0; j < 8; ++j)
        _mm256_store_ps(temp + j * 8, acc[j]);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = _mm256_setzero_ps();
        for (size_t jj = 0; jj < n; ++jj)
            ((float *)&sum)[jj] = temp[jj * 8 + r];
        __m256 old = _mm256_maskload_ps(cr, mask);
        sum = _mm256_add_ps(old, sum);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 8x8 (store) ---- */
static inline void
gemm_8x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc[8];
    for (int j = 0; j < 8; ++j)
        acc[j] = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 8);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 8);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 8;
            for (int j = 0; j < 8; ++j)
            {
                __m256 bcast = _mm256_broadcast_ss(b_row + j);
                acc[j] = _mm256_fmadd_ps(a, bcast, acc[j]);
            }
        }
    }

    alignas(32) float temp[8 * 8];
    for (int j = 0; j < 8; ++j)
        _mm256_store_ps(temp + j * 8, acc[j]);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 8, r, n);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 16x6 (add) ---- */
static inline void
gemm_16x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                            const float *RESTRICT Ap,
                            const float *RESTRICT Bp,
                            size_t Kblk, size_t m, size_t n, __m256i mask /* for n<6 */)
{
    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 6;
            for (int j = 0; j < 6; ++j)
            {
                __m256 bcast = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, bcast, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, bcast, acc_hi[j]);
            }
        }
    }

    alignas(32) float temp[16 * 6];
    for (int j = 0; j < 6; ++j)
    {
        _mm256_store_ps(temp + j * 16, acc_lo[j]);
        _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
    }

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 16, r, n);
        __m256 old = _mm256_maskload_ps(cr, mask);
        sum = _mm256_add_ps(old, sum);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 16x6 (store) ---- */
static inline void
gemm_16x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                              const float *RESTRICT Ap,
                              const float *RESTRICT Bp,
                              size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc_lo[6], acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 16);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a_lo = _mm256_load_ps(Ap + kk * 16);
            __m256 a_hi = _mm256_load_ps(Ap + kk * 16 + 8);
            const float *b_row = Bp + kk * 6;
            for (int j = 0; j < 6; ++j)
            {
                __m256 bcast = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, bcast, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, bcast, acc_hi[j]);
            }
        }
    }

    alignas(32) float temp[16 * 6];
    for (int j = 0; j < 6; ++j)
    {
        _mm256_store_ps(temp + j * 16, acc_lo[j]);
        _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
    }

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 16, r, n);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 8x6 (add) ---- */
static inline void
gemm_8x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 6;
            __m256 bcast = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, bcast, acc0);
            bcast = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, bcast, acc1);
            bcast = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, bcast, acc2);
            bcast = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, bcast, acc3);
            bcast = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, bcast, acc4);
            bcast = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, bcast, acc5);
        }
    }

    alignas(32) float temp[8 * 6];
    _mm256_store_ps(temp + 0 * 8, acc0);
    _mm256_store_ps(temp + 1 * 8, acc1);
    _mm256_store_ps(temp + 2 * 8, acc2);
    _mm256_store_ps(temp + 3 * 8, acc3);
    _mm256_store_ps(temp + 4 * 8, acc4);
    _mm256_store_ps(temp + 5 * 8, acc5);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 8, r, n);
        __m256 old = _mm256_maskload_ps(cr, mask);
        sum = _mm256_add_ps(old, sum);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

/* ---- 8x6 (store) ---- */
static inline void
gemm_8x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t m, size_t n, __m256i mask)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;
            size_t kpf_l = k + PF_LONG;
            if (kpf_s < Kblk)
                PREFETCH_T0(Bp + kpf_s * 6);
            if (kpf_l < Kblk)
                PREFETCH_T0(Bp + kpf_l * 6);
#if LINALG_GEMM_PREFETCH_A_LONG
            if (kpf_l < Kblk)
                PREFETCH_T0(Ap + kpf_l * 8);
#endif
        }
        for (int u = 0; u < 8; ++u)
        {
            size_t kk = k + u;
            if (kk >= Kblk)
                break;
            __m256 a = _mm256_load_ps(Ap + kk * 8);
            const float *b_row = Bp + kk * 6;
            __m256 bcast = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, bcast, acc0);
            bcast = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, bcast, acc1);
            bcast = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, bcast, acc2);
            bcast = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, bcast, acc3);
            bcast = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, bcast, acc4);
            bcast = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, bcast, acc5);
        }
    }

    alignas(32) float temp[8 * 6];
    _mm256_store_ps(temp + 0 * 8, acc0);
    _mm256_store_ps(temp + 1 * 8, acc1);
    _mm256_store_ps(temp + 2 * 8, acc2);
    _mm256_store_ps(temp + 3 * 8, acc3);
    _mm256_store_ps(temp + 4 * 8, acc4);
    _mm256_store_ps(temp + 5 * 8, acc5);

    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        __m256 sum = load_cols_from_temp(temp, 8, r, n);
        _mm256_maskstore_ps(cr, mask, sum);
    }
}

#endif /* LINALG_SIMD_ENABLE */

/* ======================= Kernel selection ======================= */
enum kernel_shape
{
    K16x6,
    K8x6,
    K16x8,
    K8x8
};

static inline enum kernel_shape pick_kernel(size_t Mblk, size_t Nblk, size_t Kblk)
{
    (void)Kblk; /* currently used only for a guard in caller */
    if (Nblk >= 8 && (Nblk % 8 >= 6 || Nblk >= 3 * (size_t)LINALG_SMALL_N_THRESH))
    {
        // prefer 8-wide if tails are small-ish or N is very wide
        if (Mblk >= 16)
            return K16x8;
        if (Mblk >= 8)
            return K8x8;
    }
    // else 6-wide
    if (Mblk >= 16)
        return K16x6;
    return K8x6;
}

/* ======================= Top-level GEMM ======================= */
/**
 * @brief GEMM-lite: compute C = A * B (row-major), AVX2/FMA-optimized.
 *
 * @details
 * Three-level blocking (Nc/Mc/Kc) around AVX2/FMA micro-kernels. B is packed
 * into 6-column or 8-column panels, A into col-major with ld=16/8. Kernels:
 * - 16×6, 8×6, 16×8, 8×8 (add/store variants).
 *
 * Scalar fallback triggers for tiny matrices or when AVX2 is unavailable.
 * Explicit prefetching is controlled by `LINALG_GEMM_PREFETCH_ENABLE`.
 */
int mul(float *RESTRICT C,
        const float *RESTRICT A,
        const float *RESTRICT B,
        uint16_t row_a, uint16_t column_a,
        uint16_t row_b, uint16_t column_b)
{
    if (column_a != row_b)
        return -EINVAL;

    const size_t M = row_a, K = column_a, N = column_b;

    if (!linalg_has_avx2() || M == 0 || N == 0 || K == 0 ||
        (M < LINALG_SMALL_N_THRESH && N < LINALG_SMALL_N_THRESH))
    {
        /* scalar fallback */
        for (size_t i = 0; i < M; ++i)
        {
            const float *ai = A + i * K;
            for (size_t j = 0; j < N; ++j)
            {
                const float *bj = B + j;
                float s = 0.f;
                for (size_t k = 0; k < K; ++k)
                    s += ai[k] * bj[k * N];
                C[i * N + j] = s;
            }
        }
        return 0;
    }

#if LINALG_SIMD_ENABLE
    const size_t Kc = (size_t)LINALG_BLOCK_KC;
    const size_t Nc = (size_t)LINALG_BLOCK_JC;
    const size_t Mc = (size_t)LINALG_BLOCK_MC;

    struct ker
    {
        void (*packA_blk)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t);
        void (*packA_tail)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t);
        void (*gemm_add)(float *, size_t, const float *, const float *, size_t, size_t, size_t, __m256i);
        void (*gemm_store)(float *, size_t, const float *, const float *, size_t, size_t, size_t, __m256i);
        size_t MR, NR, A_ld;
    };

    static const struct ker KERS[4] = {
        {// K16x6
         pack_A_block_16row_colmajor, pack_A_16row_tile,
         gemm_16x6_panel_avx2fma_add, gemm_16x6_panel_avx2fma_store,
         16, 6, 16},
        {// K8x6
         pack_A_block_8row_colmajor, pack_A_block_8row_colmajor,
         gemm_8x6_panel_avx2fma_add, gemm_8x6_panel_avx2fma_store,
         8, 6, 8},
        {// K16x8
         pack_A_block_16row_colmajor, pack_A_16row_tile,
         gemm_16x8_panel_avx2fma_add, gemm_16x8_panel_avx2fma_store,
         16, 8, 16},
        {// K8x8
         pack_A_block_8row_colmajor, pack_A_block_8row_colmajor,
         gemm_8x8_panel_avx2fma_add, gemm_8x8_panel_avx2fma_store,
         8, 8, 8}};

    const size_t max_nr = 8; // for buffer calc
    const size_t max_n_panels = (Nc + max_nr - 1) / max_nr;
    const size_t max_Bp_elems = Kc * max_n_panels * max_nr;
    float *Bp = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, max_Bp_elems * sizeof(float));
    if (!Bp)
        return -ENOMEM;

    const size_t max_Ap_elems = Kc * 16;
    float *Ap = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, max_Ap_elems * sizeof(float));
    if (!Ap)
    {
        linalg_aligned_free(Bp);
        return -ENOMEM;
    }

    for (size_t j0 = 0; j0 < N; j0 += Nc)
    {
        const size_t jb_tile = (j0 + Nc <= N) ? Nc : (N - j0);

        for (size_t kk = 0; kk < K; kk += Kc)
        {
            const size_t Kblk = (kk + Kc <= K) ? Kc : (K - kk);

            /* L2 hints for next B slab within the same j-tile */
            if (kk + Kblk < K && jb_tile >= 64)
            {
                const size_t kk_next = kk + Kblk;
                const size_t step = (size_t)(64 / sizeof(float));
                for (size_t jpf = j0, jpf_end = j0 + jb_tile; jpf < jpf_end; jpf += step)
                    PREFETCH_T1(B + kk_next * N + jpf);
            }

            enum kernel_shape shape = pick_kernel(Mc, jb_tile, Kblk); // Mc ≈ Mblk
            const struct ker *ker = &KERS[shape];
            const size_t NR = ker->NR;
            const size_t n_panels_tile = (jb_tile + NR - 1) / NR;

            /* pack B for this shape */
            size_t panel_off = 0;
            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off += Kblk * NR)
            {
                const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                pack_B_tile(Bp + panel_off, B, K, N, kk, Kblk, j, n_block, NR);
            }

            for (size_t i0 = 0; i0 < M; i0 += Mc)
            {
                const size_t ib_tile = (i0 + Mc <= M) ? Mc : (M - i0);

                /* L2 hints for A rows of this i-tile & current kk slab */
                if (ib_tile >= 64)
                {
                    for (size_t ipf = i0, ipf_end = i0 + ib_tile; ipf < ipf_end; ipf += 8)
                        PREFETCH_T1(A + ipf * K + kk);
                }

                size_t i = 0;
                const size_t mr = ker->MR;

                for (; i + mr - 1 < ib_tile; i += mr)
                {
                    ker->packA_blk(Ap, A, M, K, i0 + i, mr, kk, Kblk);

                    size_t panel_off2 = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off2 += Kblk * NR)
                    {
                        const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                        __m256i mask = avx2_tailmask_nr(n_block, NR);
                        if (kk == 0)
                            ker->gemm_store(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, mr, n_block, mask);
                        else
                            ker->gemm_add(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, mr, n_block, mask);
                    }
                }

                /* leftover rows in this i-tile */
                if (i < ib_tile)
                {
                    const size_t m_block = ib_tile - i;
                    ker->packA_tail(Ap, A, M, K, i0 + i, m_block, kk, Kblk);

                    size_t panel_off2 = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off2 += Kblk * NR)
                    {
                        const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                        __m256i mask = avx2_tailmask_nr(n_block, NR);
                        if (kk == 0)
                            ker->gemm_store(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, m_block, n_block, mask);
                        else
                            ker->gemm_add(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, m_block, n_block, mask);
                    }
                }
            }
        }
    }

    linalg_aligned_free(Ap);
    linalg_aligned_free(Bp);
    return 0;
#else
    (void)C;
    (void)A;
    (void)B;
    (void)row_a;
    (void)column_a;
    (void)row_b;
    (void)column_b;
    return -ENOTSUP;
#endif
}
