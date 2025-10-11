// Fast row-major transpose with AVX2 8x8 kernel + SSE 4x8 tail + scalar cleanup.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include <stdlib.h>

#include "linalg_simd.h"  // linalg_has_avx2()

/* ---------- optional runtime dispatch (CPUID + XGETBV) ---------- */
#ifdef TRAN_DISPATCH
#  include <cpuid.h>
static inline int cpu_has_avx_runtime(void) {
    unsigned a,b,c,d;
    if (!__get_cpuid(1, &a,&b,&c,&d)) return 0;
    const int osxsave = (c & (1u<<27)) != 0;
    const int avx     = (c & (1u<<28)) != 0;
    if (!(osxsave && avx)) return 0;
    /* XCR0[2:1] (YMM/XMM) must be enabled by OS */
    unsigned xcr0_lo = (unsigned)_xgetbv(0);
    return ((xcr0_lo & 0x6) == 0x6);
}
#endif

/* ===================================================================== */
/*                         Micro-kernels                                  */
/* ===================================================================== */

/* 8x8 AVX (row-major src -> row-major dst, with strides) */
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

    /* cross 128-bit lanes to interleave halves */
    r0 = _mm256_permute2f128_ps(t0, t4, 0x20);
    r1 = _mm256_permute2f128_ps(t0, t4, 0x31);
    r2 = _mm256_permute2f128_ps(t1, t5, 0x20);
    r3 = _mm256_permute2f128_ps(t1, t5, 0x31);
    r4 = _mm256_permute2f128_ps(t2, t6, 0x20);
    r5 = _mm256_permute2f128_ps(t2, t6, 0x31);
    r6 = _mm256_permute2f128_ps(t3, t7, 0x20);
    r7 = _mm256_permute2f128_ps(t3, t7, 0x31);

    _mm256_storeu_ps(dst + 0 * dst_stride, r0);
    _mm256_storeu_ps(dst + 1 * dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
}

/* 8-rows x 4-cols tail using SSE (_MM_TRANSPOSE4_PS twice layouted as 8x4) */
static inline void transpose8x4_sse(const float *RESTRICT src, float *RESTRICT dst,
                                    size_t src_stride, size_t dst_stride)
{
    /* Load 8x4 as two 4x4 tiles stacked vertically: rows i..i+3 and i+4..i+7 */
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

    /* Store 4 columns of 8 elements (two 4-element chunks each) */
    _mm_storeu_ps(dst + 0 * dst_stride + 0, a0);
    _mm_storeu_ps(dst + 0 * dst_stride + 4, b0);

    _mm_storeu_ps(dst + 1 * dst_stride + 0, a1);
    _mm_storeu_ps(dst + 1 * dst_stride + 4, b1);

    _mm_storeu_ps(dst + 2 * dst_stride + 0, a2);
    _mm_storeu_ps(dst + 2 * dst_stride + 4, b2);

    _mm_storeu_ps(dst + 3 * dst_stride + 0, a3);
    _mm_storeu_ps(dst + 3 * dst_stride + 4, b3);
}

/* Scalar cleanup / tiny matrices */
static inline void transpose_scalar(const float *RESTRICT src, float *RESTRICT dst,
                                    size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

/* ===================================================================== */
/*                              Top-level                                 */
/* ===================================================================== */

void tran(float *RESTRICT At, const float *RESTRICT A, uint16_t row, uint16_t column)
{
    const size_t R = row, C = column;
    if (R == 0 || C == 0) return;

    /* tiny-matrix fast path */
    if (R < 8 || C < 8) {
        transpose_scalar(A, At, R, C);
        return;
    }

    /* choose SIMD path */
    int use_avx = linalg_has_avx2();
#ifdef TRAN_DISPATCH
    use_avx = use_avx && cpu_has_avx_runtime();
#endif

    /* in-place support: if At == A, use a temporary */
    if (At == A) {
        const size_t nbytes = R * C * sizeof(float);
        float *tmp = (float*)linalg_aligned_alloc(32, nbytes);
        if (!tmp) {                  /* allocation failed: fallback scalar via two-phase */
            float *tmp2 = (float*)malloc(nbytes);
            if (!tmp2) return;       /* give up silently */
            transpose_scalar(A, tmp2, R, C);
            memcpy(At, tmp2, nbytes);
            free(tmp2);
            return;
        }
        /* write into tmp, then copy back to At */
        tran(tmp, A, row, column);   /* recurse once with non-aliased dst */
        memcpy(At, tmp, nbytes);
        linalg_aligned_free(tmp);
        return;
    }

    /* block sizes */
    const size_t rb8 = R & ~7u;              /* max multiple of 8 <= R */
    const size_t cb8 = C & ~7u;              /* max multiple of 8 <= C */
    const size_t cb4 = cb8 + ((C - cb8) & ~3u);  /* then 4-wide tail */

    if (use_avx) {
        /* 8x8 core */
        for (size_t i = 0; i < rb8; i += 8)
            for (size_t j = 0; j < cb8; j += 8)
                transpose8x8_avx(A + i * C + j, At + j * R + i, C, R);

        /* 8x4 tail across columns */
        for (size_t i = 0; i < rb8; i += 8)
            for (size_t j = cb8; j < cb4; j += 4)
                transpose8x4_sse(A + i * C + j, At + j * R + i, C, R);
    }

    /* scalar tails:
       - leftover columns j in [cb4 .. C)
       - leftover rows    i in [rb8 .. R)  (all columns) */
    /* leftover columns for full 8-row blocks */
    for (size_t i = 0; i < rb8; ++i)
        for (size_t j = cb4; j < C; ++j)
            At[j * R + i] = A[i * C + j];

    /* leftover rows (all columns) */
    for (size_t i = rb8; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            At[j * R + i] = A[i * C + j];
}
