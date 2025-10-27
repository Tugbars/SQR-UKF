// SPDX-License-Identifier: MIT
/**
 * @file tran_pack.c
 * @brief Tiled transpose (single-threaded) with optional NT stores + fused pack-transpose helpers.
 *
 * @details
 *  A plain transpose is memory-bound, so the big win is *not writing At at all*.
 *  These helpers let higher-level BLAS-3 kernels request panels already stored
 *  in transposed layout during packing. If you still need a standalone transpose,
 *  @ref tran_tiled provides a cache-friendly 32×32 tiler over your AVX2 8×8 micro-kernel,
 *  with an opt-in non-temporal store path to reduce cache pollution on large outputs.
 *
 *  What you get:
 *   - tran_tiled(): 32×32 macro-tiles → 8×8 AVX + 8×4 SSE tails, optional NT stores.
 *   - pack_T_8xK(): pack a transposed 8×K micro-panel from row-major A (feeds your 8×16 kernel’s A side).
 *   - pack_T_Kx16(): pack a transposed K×16 micro-panel from row-major B (feeds your 8×16 kernel’s B side).
 *
 *  All routines are single-threaded by design. When you add threading, parallelize
 *  the *outer* tiles in tran_tiled() and the caller’s macro-tiling loops around packers.
 *
 * Build-time knobs:
 *   - TRAN_TILE: macro-tile edge (default 32).
 *   - TRAN_USE_NT_STORES: 0/1 to enable _mm256_stream_ps in tran_tiled().
 *
 * ISA:
 *   - AVX2 used for 8×8; SSE for 8×4 tail; scalar cleanup otherwise.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include "linalg_simd.h"

#ifndef TRAN_TILE
#define TRAN_TILE 32
#endif
#ifndef TRAN_USE_NT_STORES
#define TRAN_USE_NT_STORES 1
#endif

/* ---------- your existing micro-kernels (kept verbatim) ---------- */
static inline void transpose8x8_avx(const float *RESTRICT src, float *RESTRICT dst,
                                    size_t src_stride, size_t dst_stride)
{
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    r0 = _mm256_shuffle_ps(t0, t2, 0x44);
    r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    r2 = _mm256_shuffle_ps(t1, t3, 0x44);
    r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    r4 = _mm256_shuffle_ps(t4, t6, 0x44);
    r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    r6 = _mm256_shuffle_ps(t5, t7, 0x44);
    r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    t0 = _mm256_unpacklo_ps(r0, r4);
    t1 = _mm256_unpackhi_ps(r0, r4);
    t2 = _mm256_unpacklo_ps(r1, r5);
    t3 = _mm256_unpackhi_ps(r1, r5);
    t4 = _mm256_unpacklo_ps(r2, r6);
    t5 = _mm256_unpackhi_ps(r2, r6);
    t6 = _mm256_unpacklo_ps(r3, r7);
    t7 = _mm256_unpackhi_ps(r3, r7);

    r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
    r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
    r2 = _mm256_permute2f128_ps(t1, t5, 0x20);
    r3 = _mm256_permute2f128_ps(t1, t5, 0x31);
    r4 = _mm256_permute2f128_ps(t2, t6, 0x20);
    r5 = _mm256_permute2f128_ps(t2, t6, 0x31);
    r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
    r7 = _mm256_permute2f128_ps(t3, t7, 0x31);

#if TRAN_USE_NT_STORES
    _mm256_stream_ps(dst + 0 * dst_stride, r0);
    _mm256_stream_ps(dst + 1 * dst_stride, r1);
    _mm256_stream_ps(dst + 2 * dst_stride, r2);
    _mm256_stream_ps(dst + 3 * dst_stride, r3);
    _mm256_stream_ps(dst + 4 * dst_stride, r4);
    _mm256_stream_ps(dst + 5 * dst_stride, r5);
    _mm256_stream_ps(dst + 6 * dst_stride, r6);
    _mm256_stream_ps(dst + 7 * dst_stride, r7);
#else
    _mm256_storeu_ps(dst + 0 * dst_stride, r0);
    _mm256_storeu_ps(dst + 1 * dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
#endif
}

static inline void transpose8x4_sse(const float *RESTRICT src, float *RESTRICT dst,
                                    size_t src_stride, size_t dst_stride)
{
    __m128 a0 = _mm_loadu_ps(src + 0 * src_stride);
    __m128 a1 = _mm_loadu_ps(src + 1 * src_stride);
    __m128 a2 = _mm_loadu_ps(src + 2 * src_stride);
    __m128 a3 = _mm_loadu_ps(src + 3 * src_stride);
    __m128 b0 = _mm_loadu_ps(src + 4 * src_stride);
    __m128 b1 = _mm_loadu_ps(src + 5 * src_stride);
    __m128 b2 = _mm_loadu_ps(src + 6 * src_stride);
    __m128 b3 = _mm_loadu_ps(src + 7 * src_stride);

    _MM_TRANSPOSE4_PS(a0, a1, a2, a3);
    _MM_TRANSPOSE4_PS(b0, b1, b2, b3);

#if TRAN_USE_NT_STORES
    _mm_stream_ps(dst + 0 * dst_stride + 0, a0);
    _mm_stream_ps(dst + 0 * dst_stride + 4, b0);
    _mm_stream_ps(dst + 1 * dst_stride + 0, a1);
    _mm_stream_ps(dst + 1 * dst_stride + 4, b1);
    _mm_stream_ps(dst + 2 * dst_stride + 0, a2);
    _mm_stream_ps(dst + 2 * dst_stride + 4, b2);
    _mm_stream_ps(dst + 3 * dst_stride + 0, a3);
    _mm_stream_ps(dst + 3 * dst_stride + 4, b3);
#else
    _mm_storeu_ps(dst + 0 * dst_stride + 0, a0);
    _mm_storeu_ps(dst + 0 * dst_stride + 4, b0);
    _mm_storeu_ps(dst + 1 * dst_stride + 0, a1);
    _mm_storeu_ps(dst + 1 * dst_stride + 4, b1);
    _mm_storeu_ps(dst + 2 * dst_stride + 0, a2);
    _mm_storeu_ps(dst + 2 * dst_stride + 4, b2);
    _mm_storeu_ps(dst + 3 * dst_stride + 0, a3);
    _mm_storeu_ps(dst + 3 * dst_stride + 4, b3);
#endif
}

static inline void transpose_scalar_block(const float *RESTRICT src, float *RESTRICT dst,
                                          size_t R, size_t C, size_t i, size_t j,
                                          size_t rb, size_t cb)
{
    for (size_t r = 0; r < rb; ++r)
        for (size_t c = 0; c < cb; ++c)
            dst[(j + c) * R + (i + r)] = src[(i + r) * C + (j + c)];
}

/**
 * @brief Tiled transpose with AVX2 8×8 + SSE 8×4 tails (single-threaded).
 * @param[out] At  Row-major C×R output (may alias A? no; use a temp if you need in-place).
 * @param[in]  A   Row-major R×C input.
 * @param[in]  R   Rows of A.
 * @param[in]  C   Cols of A.
 *
 * @note If TRAN_USE_NT_STORES!=0, uses non-temporal stores; caller should not
 *       immediately read back At. A fence is emitted at the end.
 */
void tran_tiled(float *RESTRICT At, const float *RESTRICT A, uint16_t R, uint16_t C)
{
    if (!R || !C)
        return;

    const size_t TS = TRAN_TILE;
    const size_t Rb = R & ~(size_t)7;
    const size_t Cb = C & ~(size_t)7;
    const size_t C4 = Cb + ((C - Cb) & ~(size_t)3);

    for (size_t i0 = 0; i0 < R; i0 += TS)
    {
        const size_t ib = (i0 + TS <= R) ? TS : (R - i0);
        const size_t ib8 = ib & ~(size_t)7;

        for (size_t j0 = 0; j0 < C; j0 += TS)
        {
            const size_t jb = (j0 + TS <= C) ? TS : (C - j0);
            const size_t jb8 = jb & ~(size_t)7;
            const size_t jb4 = jb8 + ((jb - jb8) & ~(size_t)3);

            /* 8×8 core inside the tile */
            for (size_t i = 0; i < ib8; i += 8)
                for (size_t j = 0; j < jb8; j += 8)
                    transpose8x8_avx(A + (i0 + i) * (size_t)C + (j0 + j),
                                     At + (j0 + j) * (size_t)R + (i0 + i),
                                     C, R);

            /* 8×4 tail in-columns inside the tile */
            for (size_t i = 0; i < ib8; i += 8)
                for (size_t j = jb8; j < jb4; j += 4)
                    transpose8x4_sse(A + (i0 + i) * (size_t)C + (j0 + j),
                                     At + (j0 + j) * (size_t)R + (i0 + i),
                                     C, R);

            /* scalar remainder inside tile (cols tail and bottom rows) */
            const size_t i_tail = ib - ib8;
            const size_t j_tail_c = jb - jb4;
            if (j_tail_c)
                transpose_scalar_block(A, At, R, C, i0, j0 + jb4, ib8, j_tail_c);
            if (i_tail)
                transpose_scalar_block(A, At, R, C, i0 + ib8, j0, i_tail, jb);
        }
    }

#if TRAN_USE_NT_STORES
    _mm_sfence();
#endif
}

/* ======================= Fused pack-transpose helpers ======================= */

/**
 * @brief Pack a transposed 8×K micro-panel from row-major A into contiguous buffer.
 *
 * @param[in]  A     Row-major A (M×Ktot), leading dim lda = Ktot.
 * @param[in]  M     Rows in A.
 * @param[in]  Ktot  Cols in A.
 * @param[in]  i     Row index of the 8-row block in Aᵀ → i..i+7 columns of A.
 * @param[in]  k0    Starting column in Aᵀ → starting row in A.
 * @param[in]  K     Depth (number of rows from A) to pack.
 * @param[out] Ap    Output buffer of size 8×K, laid out row-major by the 8 rows
 *                   (i.e., suitable as the A (mr=8) operand for your 8×16 kernel).
 *
 * Layout result: Ap[r*K + t] = A[(k0+t), (i+r)]  for r=0..7, t=0..K-1
 */
static inline void pack_T_8xK(const float *RESTRICT A, uint16_t M, uint16_t Ktot,
                              uint16_t i, uint16_t k0, uint16_t K, float *RESTRICT Ap)
{
    (void)M;
    for (uint16_t r = 0; r < 8; ++r)
    {
        const float *col = A + (size_t)(i + r); /* column (i+r) of A */
        float *dst = Ap + (size_t)r * K;        /* row r in packed (Aᵀ) */
        for (uint16_t t = 0; t < K; ++t)
            dst[t] = col[(size_t)(k0 + t) * Ktot];
    }
}

/**
 * @brief Pack a transposed K×16 micro-panel from row-major B into contiguous buffer.
 *
 * @param[in]  B     Row-major B (Ktot×N), leading dim ldb = N.
 * @param[in]  Ktot  Rows in B.
 * @param[in]  N     Cols in B.
 * @param[in]  k0    Row start in B (→ column start in Bᵀ).
 * @param[in]  j     Column index of the 16-col block in B.
 * @param[in]  K     Depth (number of rows from B) to pack.
 * @param[out] Bp    Output buffer of size K×16 in column-panels of 16;
 *                   at step t, store two 8-wide vectors contiguous to match your 8×16 kernel.
 *
 * Layout result: Bp[t*16 + c] = B[(k0+t), (j+c)]  for c=0..15
 */
static inline void pack_T_Kx16(const float *RESTRICT B, uint16_t Ktot, uint16_t N,
                               uint16_t k0, uint16_t j, uint16_t K, float *RESTRICT Bp)
{
    for (uint16_t t = 0; t < K; ++t)
    {
        const float *row = B + (size_t)(k0 + t) * N + j;
        memcpy(Bp + (size_t)t * 16, row, 16 * sizeof(float));
    }
}
