// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <immintrin.h>
#include <string.h>
#include "linalg_simd.h" // linalg_has_avx2(), LINALG_BLOCK_KC, LINALG_BLOCK_JC, RESTRICT
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
/* ===== helper: AVX2 mask for N-tail (jb < 8) ===== */
#if LINALG_SIMD_ENABLE
static inline __m256i avx2_tailmask(int jb)
{
    /* lanes in ascending address order (setr) */
    return _mm256_setr_epi32(
        (jb > 0) ? -1 : 0,
        (jb > 1) ? -1 : 0,
        (jb > 2) ? -1 : 0,
        (jb > 3) ? -1 : 0,
        (jb > 4) ? -1 : 0,
        (jb > 5) ? -1 : 0,
        (jb > 6) ? -1 : 0,
        (jb > 7) ? -1 : 0);
}
#endif
/* ======================= Packing ======================= */
/**
 * @brief Pack an A tile into 8-row–striped layout for AVX2 micro-kernels.
 *
 * @details
 * Packs an (ib × Kblk) submatrix of row-major A into stripes of 8 rows.
 * Within each 8-row stripe `s`, elements are stored as a dense (Kblk × 8)
 * block with layout:
 *
 * Ap_off = s*(Kblk*8) + k*8 + r
 * ^stripe ^row-lane (0..7)
 *
 * where k∈[0,Kblk), r∈[0,7]. The last short stripe (ib % 8 != 0) is
 * zero-padded in the trailing row lanes, so kernels can always read 8 lanes.
 *
 * This layout feeds the 8×8 / 4×8 / 1×8 AVX2 panels using simple
 * broadcasts of contiguous scalars and aligned loads from packed B.
 *
 * @param[out] Ap Packed destination buffer of size n_stripes*Kblk*8 floats,
 * 32B-aligned. n_stripes = ceil(ib/8).
 * @param[in] A Source matrix (row-major), shape (M × K).
 * @param[in] M Rows of A (bounds are guaranteed by caller).
 * @param[in] K Cols of A (leading dimension of row-major A).
 * @param[in] i0 Row offset in A for this tile (start row).
 * @param[in] ib Number of tile rows to pack (tile height).
 * @param[in] kk Column offset in A for this K-slab (start col).
 * @param[in] Kblk Number of columns in this K-slab (tile width).
 *
 * @note The function assumes Ap is 32B-aligned and large enough.
 * @warning No bounds checks; caller guarantees (i0+ib)≤M and (kk+Kblk)≤K.
 */
static inline void
pack_A_block_8row_striped(float *RESTRICT Ap,
                          const float *RESTRICT A,
                          size_t M, size_t K,
                          size_t i0, size_t ib,
                          size_t kk, size_t Kblk)
{
    (void)M; /* bounds guaranteed by caller */
    const size_t n_stripes = (ib + 7) / 8;
    for (size_t s = 0; s < n_stripes; ++s)
    {
        const size_t stripe_rows = ((s + 1) * 8 <= ib) ? 8 : (ib - s * 8);
        const size_t Ap_base = s * (Kblk * 8);
        if (stripe_rows < 8)
        {
            memset(Ap + Ap_base, 0, Kblk * 8 * sizeof(float));
        }
        for (size_t k = 0; k < Kblk; ++k)
        {
            float *dst = Ap + Ap_base + k * 8;
            /* fill present rows of this stripe (then pad remaining lanes with 0) */
            size_t r = 0;
            for (; r < stripe_rows; ++r)
            {
                const size_t i = i0 + s * 8 + r;
                dst[r] = A[i * K + (kk + k)];
            }
        }
    }
}
/**
 * @brief Pack a B tile into contiguous 8-column panels for AVX2 kernels.
 *
 * @details
 * Packs a (Kblk × jb) submatrix of row-major B into `n_panels` panels
 * of width 8 (last panel may be <8 and is zero-padded). Within each panel,
 * row k (0..Kblk-1) is stored contiguously at:
 *
 * Bp_off = panel_base + k*8 + c
 * ^col-lane (0..7)
 *
 * This gives aligned `_mm256_load_ps` on Bp and broadcast-from-A for FMA.
 *
 * @param[out] Bp Packed destination buffer, size Kblk*n_panels*8 floats,
 * 32B-aligned.
 * @param[in] B Source matrix (row-major), shape (K × N).
 * @param[in] K Rows of B (shared inner dimension).
 * @param[in] N Cols of B (leading dimension of row-major B).
 * @param[in] kk Row offset in B for this K-slab.
 * @param[in] Kblk Number of rows in this K-slab.
 * @param[in] j0 Column offset in B for this N-tile.
 * @param[in] jb Number of columns in this N-tile.
 *
 * @note Uses optional prefetch (PREFETCH_T0) guarded by LINALG_GEMM_PREFETCH_ENABLE.
 * @warning Caller ensures Bp is aligned and large enough.
 */
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
        for (size_t k = 0; k < Kblk; ++k)
        {
            const float *src = B + (kk + k) * N + j; /* row-major */
            float *dst = Bp + off + k * 8;
            /* Prefetch neighboring rows and a small intra-row lookahead */
            if (k + 1 < Kblk)
                PREFETCH_T0(src + N);
            if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
                PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j);
            if (jb >= 16 && pf_elts >= 8 && w == 8)
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
/**
 * @brief Pack an A (ib × Kblk) tile into a simple 8-row interleaved form.
 *
 * @details
 * Simpler variant used by some kernels: for each k, store up to 8 row scalars
 * contiguously: `Ap[k*8 + r] = A[(i0+r), (kk+k)]`. Short row-blocks are padded
 * with zeros. This is compatible with broadcast patterns in the micro-kernels.
 *
 * @param[out] Ap Packed dest buffer (Kblk*8 floats), 32B-aligned.
 * @param[in] A Source (row-major), shape (M × K).
 * @param[in] M Rows of A (unused; bounds guaranteed by caller).
 * @param[in] K Cols of A (leading dimension).
 * @param[in] i0 Start row in A.
 * @param[in] ib Number of rows to pack (≤8).
 * @param[in] kk Start col in A.
 * @param[in] Kblk Number of cols to pack.
 *
 * @note Used when building smaller row-block paths.
 */
static inline void
pack_A_8row_tile(float *RESTRICT Ap,
                 const float *RESTRICT A,
                 size_t M, size_t K,
                 size_t i0, size_t ib,
                 size_t kk, size_t Kblk)
{
    (void)M; /* bounds are guaranteed by caller */
    if (ib < 8)
    {
        memset(Ap, 0, Kblk * 8 * sizeof(float));
    }
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 8;
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
        size_t r = 0;
        for (; r < ib; ++r)
        {
            const size_t i = i0 + r;
            dst[r] = A[i * K + (kk + k)];
        }
    }
}
/* ======================= Micro-kernels (AVX2/FMA) ======================= */
/**
 * @brief AVX2/FMA micro-kernel: accumulate C(8×jb) += Ap(…)*Bp(…), jb≤8.
 *
 * @details
 * Computes an 8-row block of C against one 8-wide B panel using a dual-tile
 * unroll (16 steps) to hide FMA latency. Loads Bp with aligned 256-bit loads
 * and broadcasts scalars from Ap. Handles jb<8 via masked loads/stores.
 *
 * Accumulator mapping: acc0..acc7 correspond to the 8 C rows. Each is a YMM
 * holding up to 8 columns. A second set acc*b accumulates an independent tile
 * to increase ILP.
 *
 * @param[in,out] c Pointer to C block top-left (row-major), stride ldc.
 * @param[in] ldc Leading dimension of C (N of full matrix).
 * @param[in] Ap Packed A stripe for these 8 rows (Kblk×8 scalars).
 * @param[in] Bp Packed B panel (Kblk×8), aligned.
 * @param[in] Kblk Shared inner dimension for this slab.
 * @param[in] jb Active columns in this panel (1..8).
 * @param[in] m Tail mask for jb <8.
 */
#if LINALG_SIMD_ENABLE
static inline void
gemm_8x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    /* second tile accumulators (to hide FMA latency) */
    __m256 acc0b = _mm256_setzero_ps();
    __m256 acc1b = _mm256_setzero_ps();
    __m256 acc2b = _mm256_setzero_ps();
    __m256 acc3b = _mm256_setzero_ps();
    __m256 acc4b = _mm256_setzero_ps();
    __m256 acc5b = _mm256_setzero_ps();
    __m256 acc6b = _mm256_setzero_ps();
    __m256 acc7b = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    /* L1/L2 prefetch for C rows touched by this kernel */
    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T0(c + 4 * ldc);
    PREFETCH_T0(c + 5 * ldc);
    PREFETCH_T0(c + 6 * ldc);
    PREFETCH_T0(c + 7 * ldc);
    PREFETCH_T1(c + 0 * ldc + 2 * ldc);
    PREFETCH_T1(c + 4 * ldc + 2 * ldc);
    size_t k = 0;
    /* dual-tile unroll by 16 */
    for (; k + 15 < Kblk; k += 16)
    {
        if (do_pf)
        {
            const size_t kpf = k + 16;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        /* tile 0: k..k+7 */
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7);
            bptr += 8;
            aptr += 8;
        }
        /* tile 1: k+8..k+15 (independent acc set) */
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0b);
            acc1b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1b);
            acc2b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2b);
            acc3b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3b);
            acc4b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4b);
            acc5b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5b);
            acc6b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6b);
            acc7b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7b);
            bptr += 8;
            aptr += 8;
        }
    }
    /* finish remainder ≥8 with single-tile path */
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7);
            bptr += 8;
            aptr += 8;
        }
    }
    /* scalar tail (k … Kblk-1) */
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 4), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 5), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 6), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 7), b, acc7);
    }
    /* fold dual-tile accumulators */
    acc0 = _mm256_add_ps(acc0, acc0b);
    acc1 = _mm256_add_ps(acc1, acc1b);
    acc2 = _mm256_add_ps(acc2, acc2b);
    acc3 = _mm256_add_ps(acc3, acc3b);
    acc4 = _mm256_add_ps(acc4, acc4b);
    acc5 = _mm256_add_ps(acc5, acc5b);
    acc6 = _mm256_add_ps(acc6, acc6b);
    acc7 = _mm256_add_ps(acc7, acc7b);
    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 0 * ldc), acc0));
        _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 1 * ldc), acc1));
        _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), acc2));
        _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), acc3));
        _mm256_storeu_ps(c + 4 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 4 * ldc), acc4));
        _mm256_storeu_ps(c + 5 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 5 * ldc), acc5));
        _mm256_storeu_ps(c + 6 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 6 * ldc), acc6));
        _mm256_storeu_ps(c + 7 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 7 * ldc), acc7));
    }
    else
    {
#define MASKED_ADD_STORE(ROW, ACC)                            \
    do                                                        \
    {                                                         \
        __m256 oldv = _mm256_maskload_ps(c + (ROW) * ldc, m); \
        __m256 sum = _mm256_add_ps(oldv, (ACC));              \
        _mm256_maskstore_ps(c + (ROW) * ldc, m, sum);         \
    } while (0)
        MASKED_ADD_STORE(0, acc0);
        MASKED_ADD_STORE(1, acc1);
        MASKED_ADD_STORE(2, acc2);
        MASKED_ADD_STORE(3, acc3);
        MASKED_ADD_STORE(4, acc4);
        MASKED_ADD_STORE(5, acc5);
        MASKED_ADD_STORE(6, acc6);
        MASKED_ADD_STORE(7, acc7);
#undef MASKED_ADD_STORE
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_8x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    /* second tile accumulators (to hide FMA latency) */
    __m256 acc0b = _mm256_setzero_ps();
    __m256 acc1b = _mm256_setzero_ps();
    __m256 acc2b = _mm256_setzero_ps();
    __m256 acc3b = _mm256_setzero_ps();
    __m256 acc4b = _mm256_setzero_ps();
    __m256 acc5b = _mm256_setzero_ps();
    __m256 acc6b = _mm256_setzero_ps();
    __m256 acc7b = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    /* L1/L2 prefetch for C rows touched by this kernel */
    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T0(c + 4 * ldc);
    PREFETCH_T0(c + 5 * ldc);
    PREFETCH_T0(c + 6 * ldc);
    PREFETCH_T0(c + 7 * ldc);
    PREFETCH_T1(c + 0 * ldc + 2 * ldc);
    PREFETCH_T1(c + 4 * ldc + 2 * ldc);
    size_t k = 0;
    /* dual-tile unroll by 16 */
    for (; k + 15 < Kblk; k += 16)
    {
        if (do_pf)
        {
            const size_t kpf = k + 16;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        /* tile 0: k..k+7 */
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7);
            bptr += 8;
            aptr += 8;
        }
        /* tile 1: k+8..k+15 (independent acc set) */
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0b);
            acc1b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1b);
            acc2b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2b);
            acc3b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3b);
            acc4b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4b);
            acc5b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5b);
            acc6b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6b);
            acc7b = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7b);
            bptr += 8;
            aptr += 8;
        }
    }
    /* finish remainder ≥8 with single-tile path */
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 4), b, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 5), b, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 6), b, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 7), b, acc7);
            bptr += 8;
            aptr += 8;
        }
    }
    /* scalar tail (k … Kblk-1) */
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 4), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 5), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 6), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 7), b, acc7);
    }
    /* fold dual-tile accumulators */
    acc0 = _mm256_add_ps(acc0, acc0b);
    acc1 = _mm256_add_ps(acc1, acc1b);
    acc2 = _mm256_add_ps(acc2, acc2b);
    acc3 = _mm256_add_ps(acc3, acc3b);
    acc4 = _mm256_add_ps(acc4, acc4b);
    acc5 = _mm256_add_ps(acc5, acc5b);
    acc6 = _mm256_add_ps(acc6, acc6b);
    acc7 = _mm256_add_ps(acc7, acc7b);
    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, acc0);
        _mm256_storeu_ps(c + 1 * ldc, acc1);
        _mm256_storeu_ps(c + 2 * ldc, acc2);
        _mm256_storeu_ps(c + 3 * ldc, acc3);
        _mm256_storeu_ps(c + 4 * ldc, acc4);
        _mm256_storeu_ps(c + 5 * ldc, acc5);
        _mm256_storeu_ps(c + 6 * ldc, acc6);
        _mm256_storeu_ps(c + 7 * ldc, acc7);
    }
    else
    {
#define MASKED_STORE(ROW, ACC)                          \
    do                                                  \
    {                                                   \
        _mm256_maskstore_ps(c + (ROW) * ldc, m, (ACC)); \
    } while (0)
        MASKED_STORE(0, acc0);
        MASKED_STORE(1, acc1);
        MASKED_STORE(2, acc2);
        MASKED_STORE(3, acc3);
        MASKED_STORE(4, acc4);
        MASKED_STORE(5, acc5);
        MASKED_STORE(6, acc6);
        MASKED_STORE(7, acc7);
#undef MASKED_STORE
    }
}
#endif
/**
 * @brief AVX2/FMA micro-kernel: accumulate C(4×jb) += Ap(…)*Bp(…), jb≤8.
 *
 * @details
 * Same idea as the 8×8 kernel but for 4 rows. Uses aligned Bp loads,
 * broadcast-from-A, and masked tail for jb<8. Intended for handling
 * leftover row blocks (4..7).
 *
 * @param[in,out] c Pointer to C block top-left, row-major.
 * @param[in] ldc Leading dimension of C.
 * @param[in] Ap Packed A rows (Kblk×8, with row offset embedded by caller).
 * @param[in] Bp Packed B panel (Kblk×8).
 * @param[in] Kblk Shared inner dimension.
 * @param[in] jb Active columns in this panel (1..8).
 * @param[in] m Tail mask for jb <8.
 */
#if LINALG_SIMD_ENABLE
static inline void
gemm_4x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 2 * ldc + 2 * ldc);
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += 8;
            aptr += 8;
        }
    }
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
    }
    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 0 * ldc), acc0));
        _mm256_storeu_ps(c + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 1 * ldc), acc1));
        _mm256_storeu_ps(c + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 2 * ldc), acc2));
        _mm256_storeu_ps(c + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(c + 3 * ldc), acc3));
    }
    else
    {
#define MASKED_ADD_STORE4(ROW, ACC)                           \
    do                                                        \
    {                                                         \
        __m256 oldv = _mm256_maskload_ps(c + (ROW) * ldc, m); \
        __m256 sum = _mm256_add_ps(oldv, (ACC));              \
        _mm256_maskstore_ps(c + (ROW) * ldc, m, sum);         \
    } while (0)
        MASKED_ADD_STORE4(0, acc0);
        MASKED_ADD_STORE4(1, acc1);
        MASKED_ADD_STORE4(2, acc2);
        MASKED_ADD_STORE4(3, acc3);
#undef MASKED_ADD_STORE4
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_4x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    PREFETCH_T0(c + 0 * ldc);
    PREFETCH_T0(c + 1 * ldc);
    PREFETCH_T0(c + 2 * ldc);
    PREFETCH_T0(c + 3 * ldc);
    PREFETCH_T1(c + 2 * ldc + 2 * ldc);
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 1), b, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 2), b, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 3), b, acc3);
            bptr += 8;
            aptr += 8;
        }
    }
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 1), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 2), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 3), b, acc3);
    }
    if (jb == 8)
    {
        _mm256_storeu_ps(c + 0 * ldc, acc0);
        _mm256_storeu_ps(c + 1 * ldc, acc1);
        _mm256_storeu_ps(c + 2 * ldc, acc2);
        _mm256_storeu_ps(c + 3 * ldc, acc3);
    }
    else
    {
#define MASKED_STORE4(ROW, ACC)                         \
    do                                                  \
    {                                                   \
        _mm256_maskstore_ps(c + (ROW) * ldc, m, (ACC)); \
    } while (0)
        MASKED_STORE4(0, acc0);
        MASKED_STORE4(1, acc1);
        MASKED_STORE4(2, acc2);
        MASKED_STORE4(3, acc3);
#undef MASKED_STORE4
    }
}
#endif
/**
 * @brief AVX2/FMA micro-kernel: accumulate C(1×jb) += Ap(…)*Bp(…), jb≤8.
 *
 * @details
 * Single-row kernel used for residual rows. Uses aligned loads from Bp and
 * scalar broadcast from Ap; handles jb<8 with mask load/store.
 *
 * @param[in,out] c Pointer to C row start.
 * @param[in] Ap Packed A row (Kblk×8 with lane 0 used).
 * @param[in] Bp Packed B panel (Kblk×8).
 * @param[in] Kblk Shared inner dimension.
 * @param[in] jb Active columns (1..8).
 * @param[in] m Tail mask for jb <8.
 */
#if LINALG_SIMD_ENABLE
static inline void
gemm_1x8_panel_avx2fma_add(float *RESTRICT c,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    PREFETCH_T0(c + 0);
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += 8;
            aptr += 8;
        }
    }
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc);
    }
    if (jb == 8)
    {
        _mm256_storeu_ps(c, _mm256_add_ps(_mm256_loadu_ps(c), acc));
    }
    else
    {
        __m256 oldv = _mm256_maskload_ps(c, m);
        __m256 sum = _mm256_add_ps(oldv, acc);
        _mm256_maskstore_ps(c, m, sum);
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_1x8_panel_avx2fma_store(float *RESTRICT c,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t jb, __m256i m /*<=8*/)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    PREFETCH_T0(c + 0);
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8)
    {
        if (do_pf)
        {
            const size_t kpf = k + 8;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const float *bptr = Bp + k * 8;
        const float *aptr = Ap + k * 8;
        for (int t = 0; t < 8; ++t)
        {
            const __m256 b = _mm256_load_ps(bptr);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(aptr + 0), b, acc);
            bptr += 8;
            aptr += 8;
        }
    }
    for (; k < Kblk; ++k)
    {
        if (do_pf)
        {
            const size_t kpf = k + 1;
            if (kpf < Kblk)
                PREFETCH_T0(Bp + kpf * 8);
        }
        const __m256 b = _mm256_load_ps(Bp + k * 8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + k * 8 + 0), b, acc);
    }
    if (jb == 8)
    {
        _mm256_storeu_ps(c, acc);
    }
    else
    {
        _mm256_maskstore_ps(c, m, acc);
    }
}
#endif

/* ===== helper: AVX2 mask for N-tail (jb < 8) ===== */
#if LINALG_SIMD_ENABLE
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
static inline __m256i avx2_tailmask_fast(int jb)
{
    __m128i b8 = _mm_loadl_epi64((const __m128i *)kMask8x8[jb]);
    return _mm256_cvtepi8_epi32(b8);
}
#endif
enum kernel_shape
{
    K16x6,
    K8x6
};
static inline enum kernel_shape pick_kernel(size_t Mblk, size_t Nblk, size_t Kblk)
{
    if (Nblk <= 6)
    {
        if (Kblk <= 128)
            return K8x6;
        return K16x6;
    }
    if (Nblk <= 12)
    {
        return (Mblk >= 16) ? K16x6 : K8x6;
    }
    return K16x6;
}


/* ======================= Packing ======================= */
/**
 * @brief Pack an A tile into column-major layout for 16-row AVX2 micro-kernels.
 *
 * @details
 * Packs an (ib × Kblk) submatrix of row-major A into column-major order with ld=16,
 * padding rows to 16 with zeros if ib < 16. Elements are stored as:
 *
 * Ap_off = k*16 + r
 * ^row (0..15 padded)
 *
 * where k∈[0,Kblk), r∈[0,15]. This allows contiguous vertical loads of 16 rows per column.
 *
 * @param[out] Ap Packed destination buffer of size Kblk*16 floats, 32B-aligned.
 * @param[in] A Source matrix (row-major), shape (M × K).
 * @param[in] M Rows of A (bounds are guaranteed by caller).
 * @param[in] K Cols of A (leading dimension of row-major A).
 * @param[in] i0 Row offset in A for this tile (start row).
 * @param[in] ib Number of tile rows to pack (tile height).
 * @param[in] kk Column offset in A for this K-slab (start col).
 * @param[in] Kblk Number of columns in this K-slab (tile width).
 *
 * @note The function assumes Ap is 32B-aligned and large enough.
 * @warning No bounds checks; caller guarantees (i0+ib)≤M and (kk+Kblk)≤K.
 */
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
        for (size_t r = 0; r < ib; ++r)
        {
            const size_t i = i0 + r;
            dst[r] = A[i * K + (kk + k)];
        }
    }
}
/**
 * @brief Pack an A tile into column-major layout for 8-row AVX2 micro-kernels.
 *
 * @details
 * Packs an (ib × Kblk) submatrix of row-major A into column-major order with ld=8,
 * padding rows to 8 with zeros if ib < 8. Elements are stored as:
 *
 * Ap_off = k*8 + r
 * ^row (0..7 padded)
 *
 * where k∈[0,Kblk), r∈[0,7]. This allows contiguous vertical loads of 8 rows per column.
 *
 * @param[out] Ap Packed destination buffer of size Kblk*8 floats, 32B-aligned.
 * @param[in] A Source matrix (row-major), shape (M × K).
 * @param[in] M Rows of A (bounds are guaranteed by caller).
 * @param[in] K Cols of A (leading dimension of row-major A).
 * @param[in] i0 Row offset in A for this tile (start row).
 * @param[in] ib Number of tile rows to pack (tile height).
 * @param[in] kk Column offset in A for this K-slab (start col).
 * @param[in] Kblk Number of columns in this K-slab (tile width).
 *
 * @note The function assumes Ap is 32B-aligned and large enough.
 * @warning No bounds checks; caller guarantees (i0+ib)≤M and (kk+Kblk)≤K.
 */
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
        for (size_t r = 0; r < ib; ++r)
        {
            const size_t i = i0 + r;
            dst[r] = A[i * K + (kk + k)];
        }
    }
}
/**
 * @brief Pack a B tile into row-major layout for 6-column AVX2 kernels.
 *
 * @details
 * Packs a (Kblk × jb) submatrix of row-major B into row-major order with ld=6,
 * padding columns to 6 with zeros if jb < 6. Elements are stored as:
 *
 * Bp_off = k*6 + c
 * ^col (0..5 padded)
 *
 * where k∈[0,Kblk), c∈[0,5]. This allows scalar broadcasts from contiguous rows.
 *
 * @param[out] Bp Packed destination buffer, size Kblk*6 floats, 32B-aligned.
 * @param[in] B Source matrix (row-major), shape (K × N).
 * @param[in] K Rows of B (shared inner dimension).
 * @param[in] N Cols of B (leading dimension of row-major B).
 * @param[in] kk Row offset in B for this K-slab.
 * @param[in] Kblk Number of rows in this K-slab.
 * @param[in] j0 Column offset in B for this N-tile.
 * @param[in] jb Number of columns in this N-tile.
 *
 * @note Uses optional prefetch (PREFETCH_T0) guarded by LINALG_GEMM_PREFETCH_ENABLE.
 * @warning Caller ensures Bp is aligned and large enough.
 */
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
        if (jb >= 16 && pf_elts >= 6 && pf_elts <= (N - j0))
            PREFETCH_T0(src + pf_elts);
        /* copy present cols */
        memcpy(dst, src, jb * sizeof(float));
        if (jb < 6)
        {
            memset(dst + jb, 0, (6 - jb) * sizeof(float));
        }
    }
}
/**
 * @brief Pack an A (ib × Kblk) tile into column-major for small blocks.
 *
 * @details
 * Similar to pack_A_block_16row_colmajor but for smaller ib <=16.
 *
 * @param[out] Ap Packed dest buffer (Kblk*16 floats), 32B-aligned.
 * @param[in] A Source (row-major), shape (M × K).
 * @param[in] M Rows of A (unused; bounds guaranteed by caller).
 * @param[in] K Cols of A (leading dimension).
 * @param[in] i0 Start row in A.
 * @param[in] ib Number of rows to pack (≤16).
 * @param[in] kk Start col in A.
 * @param[in] Kblk Number of cols to pack.
 *
 * @note Used for leftovers.
 */
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
        for (size_t r = 0; r < ib; ++r)
        {
            const size_t i = i0 + r;
            dst[r] = A[i * K + (kk + k)];
        }
    }
}
/* ======================= Micro-kernels (AVX2/FMA) ======================= */
/**
 * @brief AVX2/FMA micro-kernel: accumulate C(16×6) += Ap(…)*Bp(…), with tails.
 *
 * @details
 * Computes a 16-row x 6-col block using vertical accumulators. Loads A vertical (2 YMM per column),
 * broadcasts B scalars, FMAs to vertical acc. Handles m<=16, n<=6 via padding (zeros in pack).
 * At end, stores to temp col-major buffer, then copies scalar to row-major C.
 *
 * @param[in,out] c Pointer to C block top-left (row-major), stride ldc.
 * @param[in] ldc Leading dimension of C (N).
 * @param[in] Ap Packed A (col-major ld=16, Kblk*16).
 * @param[in] Bp Packed B (row-major ld=6, Kblk*6).
 * @param[in] Kblk Shared inner dimension.
 * @param[in] m Active rows (1..16).
 * @param[in] n Active columns (1..6).
 */
#if LINALG_SIMD_ENABLE
static inline void
gemm_16x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                            const float *RESTRICT Ap,
                            const float *RESTRICT Bp,
                            size_t Kblk, size_t m, size_t n)
{
    __m256 acc_lo[6];
    __m256 acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    /* prefetch C rows */
    for (size_t r = 0; r < m; r += 4)
    {
        PREFETCH_T0(c + r * ldc);
    }
    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;       // short lookahead
            size_t kpf_l = k + PF_LONG; // long lookahead
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
    /* temp buffer for col-major store */
    alignas(32) float temp[16 * 6];
    for (int j = 0; j < 6; ++j)
    {
        _mm256_store_ps(temp + j * 16, acc_lo[j]);
        _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
    }
    /* add to C row-major, vectorized for n=6 */
    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        if (n == 6)
        {
            __m256 sum = _mm256_set_ps(0, 0, temp[5 * 16 + r], temp[4 * 16 + r], temp[3 * 16 + r], temp[2 * 16 + r], temp[1 * 16 + r], temp[0 * 16 + r]);
            __m256 old = _mm256_loadu_ps(cr);
            sum = _mm256_add_ps(old, sum);
            _mm256_storeu_ps(cr, sum);
        }
        else
        {
            for (size_t jj = 0; jj < n; ++jj)
            {
                cr[jj] += temp[jj * 16 + r];
            }
        }
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_16x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                              const float *RESTRICT Ap,
                              const float *RESTRICT Bp,
                              size_t Kblk, size_t m, size_t n)
{
    __m256 acc_lo[6];
    __m256 acc_hi[6];
    for (int j = 0; j < 6; ++j)
    {
        acc_lo[j] = _mm256_setzero_ps();
        acc_hi[j] = _mm256_setzero_ps();
    }
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;       // short lookahead
            size_t kpf_l = k + PF_LONG; // long lookahead
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
    /* temp buffer for col-major store */
    alignas(32) float temp[16 * 6];
    for (int j = 0; j < 6; ++j)
    {
        _mm256_store_ps(temp + j * 16, acc_lo[j]);
        _mm256_store_ps(temp + j * 16 + 8, acc_hi[j]);
    }
    /* store to C row-major, vectorized for n=6 */
    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        if (n == 6)
        {
            __m256 sum = _mm256_set_ps(0, 0, temp[5 * 16 + r], temp[4 * 16 + r], temp[3 * 16 + r], temp[2 * 16 + r], temp[1 * 16 + r], temp[0 * 16 + r]);
            _mm256_storeu_ps(cr, sum);
        }
        else
        {
            for (size_t jj = 0; jj < n; ++jj)
            {
                cr[jj] = temp[jj * 16 + r];
            }
        }
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_8x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t m, size_t n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    /* prefetch C rows */
    for (size_t r = 0; r < m; r += 4)
    {
        PREFETCH_T0(c + r * ldc);
    }
    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;       // short lookahead
            size_t kpf_l = k + PF_LONG; // long lookahead
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
    /* temp buffer for col-major store */
    alignas(32) float temp[8 * 6];
    _mm256_store_ps(temp + 0 * 8, acc0);
    _mm256_store_ps(temp + 1 * 8, acc1);
    _mm256_store_ps(temp + 2 * 8, acc2);
    _mm256_store_ps(temp + 3 * 8, acc3);
    _mm256_store_ps(temp + 4 * 8, acc4);
    _mm256_store_ps(temp + 5 * 8, acc5);
    /* add to C row-major, vectorized for n=6 */
    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        if (n == 6)
        {
            __m256 sum = _mm256_set_ps(0, 0, temp[5 * 8 + r], temp[4 * 8 + r], temp[3 * 8 + r], temp[2 * 8 + r], temp[1 * 8 + r], temp[0 * 8 + r]);
            __m256 old = _mm256_loadu_ps(cr);
            sum = _mm256_add_ps(old, sum);
            _mm256_storeu_ps(cr, sum);
        }
        else
        {
            for (size_t jj = 0; jj < n; ++jj)
            {
                cr[jj] += temp[jj * 8 + r];
            }
        }
    }
}
#endif
#if LINALG_SIMD_ENABLE
static inline void
gemm_8x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t m, size_t n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8;       // short lookahead
            size_t kpf_l = k + PF_LONG; // long lookahead
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
    /* temp buffer for col-major store */
    alignas(32) float temp[8 * 6];
    _mm256_store_ps(temp + 0 * 8, acc0);
    _mm256_store_ps(temp + 1 * 8, acc1);
    _mm256_store_ps(temp + 2 * 8, acc2);
    _mm256_store_ps(temp + 3 * 8, acc3);
    _mm256_store_ps(temp + 4 * 8, acc4);
    _mm256_store_ps(temp + 5 * 8, acc5);
    /* store to C row-major, vectorized for n=6 */
    for (size_t r = 0; r < m; ++r)
    {
        float *cr = c + r * ldc;
        if (n == 6)
        {
            __m256 sum = _mm256_set_ps(0, 0, temp[5 * 8 + r], temp[4 * 8 + r], temp[3 * 8 + r], temp[2 * 8 + r], temp[1 * 8 + r], temp[0 * 8 + r]);
            _mm256_storeu_ps(cr, sum);
        }
        else
        {
            for (size_t jj = 0; jj < n; ++jj)
            {
                cr[jj] = temp[jj * 8 + r];
            }
        }
    }
}
#endif
/* ======================= Top-level GEMM ======================= */
/**
 * @brief GEMM-lite: compute C = A * B (row-major), AVX2/FMA-optimized.
 *
 * @details
 * Three-level blocking (Nc/Mc/Kc) around AVX2/FMA micro-kernels. B is packed
 * into 6-column panels, A into column-major with ld=16. Kernels:
 * - 16×6: main workhorse with vertical accumulators.
 * - Handles leftovers via padding.
 *
 * Scalar fallback triggers for tiny matrices or when AVX2 is unavailable.
 * Explicit prefetching is controlled by compile-time switch
 * `LINALG_GEMM_PREFETCH_ENABLE` to simplify tuning on modern CPUs.
 *
 * @param[out] C (row_a × column_b), row-major.
 * @param[in] A (row_a × column_a), row-major.
 * @param[in] B (row_b × column_b), row-major.
 * @param[in] row_a M.
 * @param[in] column_a K (and row_b must equal K).
 * @param[in] row_b K (must equal column_a).
 * @param[in] column_b N.
 *
 * @retval 0 Success.
 * @retval -EINVAL Shape mismatch (column_a != row_b).
 * @retval -ENOMEM Temporary packing buffer allocation failed.
 * @retval -ENOTSUP (builds without AVX2 path only) unsupported.
 *
 * @warning C must not alias A or B. Buffers must be valid and sized.
 * @note Tuning knobs: LINALG_BLOCK_{KC,JC,MC}, LINALG_SMALL_N_THRESH,
 * LINALG_GEMM_PF_* and LINALG_GEMM_PREFETCH_ENABLE.
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
    const size_t max_n_panels = (Nc + 5) / 6;
    const size_t max_Bp_elems = Kc * max_n_panels * 6;
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
        const size_t n_panels_tile = (jb_tile + 5) / 6;
        for (size_t kk = 0; kk < K; kk += Kc)
        {
            const size_t Kblk = (kk + Kc <= K) ? Kc : (K - kk);
            /* L2 hints for next B slab within the same j-tile */
            if (kk + Kblk < K && jb_tile >= 64)
            {
                const size_t kk_next = kk + Kblk;
                const size_t step = (size_t)(64 / sizeof(float));
                for (size_t jpf = j0, jpf_end = j0 + jb_tile; jpf < jpf_end; jpf += step)
                {
                    PREFETCH_T1(B + kk_next * N + jpf);
                }
            }
            /* pack B for all panels of this (j0, kk) */
            size_t panel_off = 0;
            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 6, panel_off += Kblk * 6)
            {
                const size_t n_block = (j + 6 <= j0 + jb_tile) ? 6 : (j0 + jb_tile - j);
                pack_B_6col_tile(Bp + panel_off, B, K, N, kk, Kblk, j, n_block);
            }
            for (size_t i0 = 0; i0 < M; i0 += Mc)
            {
                const size_t ib_tile = (i0 + Mc <= M) ? Mc : (M - i0);
                /* L2 hints for A rows of this i-tile & current kk slab */
                if (ib_tile >= 64)
                {
                    for (size_t ipf = i0, ipf_end = i0 + ib_tile; ipf < ipf_end; ipf += 8)
                    {
                        PREFETCH_T1(A + ipf * K + kk);
                    }
                }
                enum kernel_shape shape = pick_kernel(ib_tile, jb_tile, Kblk);
                /* compute using selected kernel with tails */
                size_t i = 0;
                size_t mr = (shape == K16x6) ? 16 : 8;
                size_t Ap_ld = (shape == K16x6) ? 16 : 8;
                void (*pack_A_block)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t) = (shape == K16x6) ? pack_A_block_16row_colmajor : pack_A_block_8row_colmajor;
                void (*pack_A_tile)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t) = (shape == K16x6) ? pack_A_16row_tile : pack_A_block_8row_colmajor; // reuse for 8x6 leftovers
                void (*gemm_add)(float *, size_t, const float *, const float *, size_t, size_t, size_t) = (shape == K16x6) ? gemm_16x6_panel_avx2fma_add : gemm_8x6_panel_avx2fma_add;
                void (*gemm_store)(float *, size_t, const float *, const float *, size_t, size_t, size_t) = (shape == K16x6) ? gemm_16x6_panel_avx2fma_store : gemm_8x6_panel_avx2fma_store;
                for (; i + mr - 1 < ib_tile; i += mr)
                {
                    pack_A_block(Ap, A, M, K, i0 + i, mr, kk, Kblk);
                    size_t panel_off = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 6, panel_off += Kblk * 6)
                    {
                        const size_t n_block = (j + 6 <= j0 + jb_tile) ? 6 : (j0 + jb_tile - j);
                        if (kk == 0)
                        {
                            gemm_store(
                                C + (i0 + i) * N + j, N,
                                Ap,
                                Bp + panel_off,
                                Kblk, mr, n_block);
                        }
                        else
                        {
                            gemm_add(
                                C + (i0 + i) * N + j, N,
                                Ap,
                                Bp + panel_off,
                                Kblk, mr, n_block);
                        }
                    }
                }
                /* leftover <mr rows */
                if (i < ib_tile)
                {
                    const size_t m_block = ib_tile - i;
                    pack_A_tile(Ap, A, M, K, i0 + i, m_block, kk, Kblk);
                    size_t panel_off = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += 6, panel_off += Kblk * 6)
                    {
                        const size_t n_block = (j + 6 <= j0 + jb_tile) ? 6 : (j0 + jb_tile - j);
                        if (kk == 0)
                        {
                            gemm_store(
                                C + (i0 + i) * N + j, N,
                                Ap,
                                Bp + panel_off,
                                Kblk, m_block, n_block);
                        }
                        else
                        {
                            gemm_add(
                                C + (i0 + i) * N + j, N,
                                Ap,
                                Bp + panel_off,
                                Kblk, m_block, n_block);
                        }
                    }
                }
            } /* i0 over Mc */
        } /* kk over Kc */
    } /* j0 over Nc */
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
