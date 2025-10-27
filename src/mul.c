// SPDX-License-Identifier: MIT

/*
Cache-aware 3-level blocking (Mc/Kc/Nc) with reasonable defaults that keep L2/L3 pressure low.
Two NR (6 & 8) and two MR (8 & 16) shapes so the hot inner loops are always 8- or 16-wide on the M dimension and 6/8-wide on N—exactly the sweet spot for AVX2.
AVX2 FMA micro-kernels are unrolled 8× in K (8 FMAs per cycle per accumulator bundle) and use broadcast-loads – you’re hitting the pipeline hard without register spills.
Proper software prefetching (short and long distance, A and B streams, optional L2 bypass) and it’s all compile-time toggles.
Aligned packing buffers (32 B) and optional non-temporal stores for the write-back path – you clearly measured bandwidth.
Tail handling is vectorised with masked loads/stores instead of scalar fall-back.
In-register 8×8 transpose for the 16×8/8×8 kernels avoids extra memory shuffles when C is row-major.
Small-M/N short-circuit to a scalar path avoids the call overhead when the problem is tiny.
Compile-time constants everywhere – the compiler sees loop bounds and can unroll/vectorise the pack routines too.
*/

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
#ifndef LINALG_GEMM_PREFETCH_ENABLE
#define LINALG_GEMM_PREFETCH_ENABLE 1
#endif
#ifndef LINALG_GEMM_PREFETCH_A_LONG
#define LINALG_GEMM_PREFETCH_A_LONG 0
#endif
#ifndef LINALG_NT_STORES
#define LINALG_NT_STORES 1
#endif

#if LINALG_GEMM_PREFETCH_ENABLE
#define PREFETCH_T0(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#define PREFETCH_T1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T1)
#else
#define PREFETCH_T0(ptr) ((void)0)
#define PREFETCH_T1(ptr) ((void)0)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LINALG_ASSUME_ALIGNED(p, n) (p) = (__typeof__(p))__builtin_assume_aligned((p), (n))
#else
#define LINALG_ASSUME_ALIGNED(p, n) (p)
#endif

#if LINALG_SIMD_ENABLE
/* ===== helper: AVX2 mask for N-tail (0..8 lanes) ===== */
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
static inline __m256 load_cols_from_temp(const float *temp, size_t stride, size_t r, size_t n)
{
    alignas(32) float lane[8] = {0};
    for (size_t j = 0; j < n; ++j)
        lane[j] = temp[j * stride + r];
    return _mm256_load_ps(lane);
}

/* ---- Helper for in-register 8x8 transpose ---- */
static inline void transpose_8x8_inreg(__m256 *rows /* in: columns, out: rows */)
{
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    t7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    rows[0] = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    rows[1] = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    rows[2] = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    rows[3] = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    rows[4] = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    rows[5] = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    rows[6] = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    rows[7] = _mm256_permute2f128_ps(tt3, tt7, 0x31);
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
    (void)M;
    if (ib < 16)
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 16;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}

static inline void
pack_A_block_8row_colmajor(float *RESTRICT Ap,
                           const float *RESTRICT A,
                           size_t M, size_t K,
                           size_t i0, size_t ib,
                           size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 8)
        memset(Ap, 0, Kblk * 8 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 8;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
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
    (void)M;
    if (ib < 16)
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k)
    {
        float *dst = Ap + k * 16;
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0)
        {
            const size_t i_pf = i0 + (size_t)LINALG_GEMM_PF_ROWS_AHEAD;
            if (i_pf < i0 + ib)
                PREFETCH_T0(A + i_pf * K + (kk + k));
        }
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
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
            if (k + 1 < Kblk)
                PREFETCH_T0(src + N);
            if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
                PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j);
            if (jb >= 16 && pf_elts >= 8 && w == 8 && pf_elts <= remain)
                PREFETCH_T0(src + pf_elts);
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
        if (k + 1 < Kblk)
            PREFETCH_T0(src + N);
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
            PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j0);
        const size_t remain = N - j0;
        if (jb >= 16 && pf_elts >= 6 && pf_elts <= remain)
            PREFETCH_T0(src + pf_elts);
        memcpy(dst, src, jb * sizeof(float));
        if (jb < 6)
            memset(dst + jb, 0, (6 - jb) * sizeof(float));
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
/* ================== 4x8 micro-kernels (add/store) ================== */
/* Ap layout: ld=8; only first 4 lanes are used, lanes 4..7 are padding. */
static inline void
gemm_4x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb, __m256i m /*mask for jb<=8*/)
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
    PREFETCH_T1(c + 4 * ldc); // light look-ahead

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
        /* Unroll 8 K-steps, each step uses 8 aligned B floats and 4 broadcasts from A */
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
        __m256 old, sum;
        old = _mm256_maskload_ps(c + 0 * ldc, m);
        sum = _mm256_add_ps(old, acc0);
        _mm256_maskstore_ps(c + 0 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 1 * ldc, m);
        sum = _mm256_add_ps(old, acc1);
        _mm256_maskstore_ps(c + 1 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 2 * ldc, m);
        sum = _mm256_add_ps(old, acc2);
        _mm256_maskstore_ps(c + 2 * ldc, m, sum);
        old = _mm256_maskload_ps(c + 3 * ldc, m);
        sum = _mm256_add_ps(old, acc3);
        _mm256_maskstore_ps(c + 3 * ldc, m, sum);
    }
}

static inline void
gemm_4x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t jb, __m256i m /*mask for jb<=8*/)
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
    PREFETCH_T1(c + 4 * ldc);

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
        _mm256_maskstore_ps(c + 0 * ldc, m, acc0);
        _mm256_maskstore_ps(c + 1 * ldc, m, acc1);
        _mm256_maskstore_ps(c + 2 * ldc, m, acc2);
        _mm256_maskstore_ps(c + 3 * ldc, m, acc3);
    }
}

/* ================== 1x8 micro-kernels (add/store) ================== */
/* Ap layout: ld=8; only aptr[0] used. */
static inline void
gemm_1x8_panel_avx2fma_add(float *RESTRICT c,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t jb, __m256i m /*mask for jb<=8*/)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);

    PREFETCH_T0(c);

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

static inline void
gemm_1x8_panel_avx2fma_store(float *RESTRICT c,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t jb, __m256i m /*mask for jb<=8*/)
{
    __m256 acc = _mm256_setzero_ps();
    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);

    PREFETCH_T0(c);

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
#endif /* LINALG_SIMD_ENABLE */

#if LINALG_SIMD_ENABLE

/* ---- 16x8 (add) ---- */
static inline void
gemm_16x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                            const float *RESTRICT Ap,
                            const float *RESTRICT Bp,
                            size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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

    if (m == 16 && n == 8)
    {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        transpose_8x8_inreg(cols_lo);
        transpose_8x8_inreg(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(cr), cols_lo[r]);
            _mm256_storeu_ps(cr, sum);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = _mm256_add_ps(_mm256_loadu_ps(cr), cols_hi[r - 8]);
            _mm256_storeu_ps(cr, sum);
        }
    }
    else
    {
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
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/* ---- 16x8 (store) ---- */
static inline void
gemm_16x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                              const float *RESTRICT Ap,
                              const float *RESTRICT Bp,
                              size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc_lo0 = _mm256_setzero_ps(), acc_lo1 = _mm256_setzero_ps();
    __m256 acc_lo2 = _mm256_setzero_ps(), acc_lo3 = _mm256_setzero_ps();
    __m256 acc_hi0 = _mm256_setzero_ps(), acc_hi1 = _mm256_setzero_ps();
    __m256 acc_hi2 = _mm256_setzero_ps(), acc_hi3 = _mm256_setzero_ps();
    __m256 acc_lo4 = _mm256_setzero_ps(), acc_lo5 = _mm256_setzero_ps();
    __m256 acc_lo6 = _mm256_setzero_ps(), acc_lo7 = _mm256_setzero_ps();
    __m256 acc_hi4 = _mm256_setzero_ps(), acc_hi5 = _mm256_setzero_ps();
    __m256 acc_hi6 = _mm256_setzero_ps(), acc_hi7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 8)
    {
        __m256 cols_lo[8] = {acc_lo0, acc_lo1, acc_lo2, acc_lo3, acc_lo4, acc_lo5, acc_lo6, acc_lo7};
        __m256 cols_hi[8] = {acc_hi0, acc_hi1, acc_hi2, acc_hi3, acc_hi4, acc_hi5, acc_hi6, acc_hi7};
        transpose_8x8_inreg(cols_lo);
        transpose_8x8_inreg(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_lo[r]);
            else
                _mm256_storeu_ps(cr, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_hi[r - 8]);
            else
                _mm256_storeu_ps(cr, cols_hi[r - 8]);
        }
    }
    else
    {
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
            if (use_nt && n == 8)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

/* ---- 8x8 (add) ---- */
static inline void
gemm_8x8_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(b_row + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(b_row + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    if (m == 8 && n == 8)
    {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        transpose_8x8_inreg(cols);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols[r]));
        }
    }
    else
    {
        alignas(32) float temp[8 * 8];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        _mm256_store_ps(temp + 6 * 8, acc6);
        _mm256_store_ps(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            __m256 old = _mm256_maskload_ps(cr, mask);
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/* ---- 8x8 (store) ---- */
static inline void
gemm_8x8_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
            b = _mm256_broadcast_ss(b_row + 6);
            acc6 = _mm256_fmadd_ps(a, b, acc6);
            b = _mm256_broadcast_ss(b_row + 7);
            acc7 = _mm256_fmadd_ps(a, b, acc7);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 8) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 8)
    {
        __m256 cols[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
        transpose_8x8_inreg(cols);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols[r]);
            else
                _mm256_storeu_ps(cr, cols[r]);
        }
    }
    else
    {
        alignas(32) float temp[8 * 8];
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        _mm256_store_ps(temp + 6 * 8, acc6);
        _mm256_store_ps(temp + 7 * 8, acc7);

        for (size_t r = 0; r < m; ++r)
        {
            float *cr = c + r * ldc;
            __m256 sum = load_cols_from_temp(temp, 8, r, n);
            if (use_nt && n == 8)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

/* ---- 16x6 (add) ---- */
static inline void
gemm_16x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                            const float *RESTRICT Ap,
                            const float *RESTRICT Bp,
                            size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

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
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    if (m == 16 && n == 6)
    {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j)
        {
            _mm256_store_ps(temp_lo + j * 8, acc_lo[j]);
            _mm256_store_ps(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j)
        {
            cols_lo[j] = _mm256_load_ps(temp_lo + j * 8);
            cols_hi[j] = _mm256_load_ps(temp_hi + j * 8);
        }
        transpose_8x8_inreg(cols_lo);
        transpose_8x8_inreg(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols_lo[r]));
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols_hi[r - 8]));
        }
    }
    else
    {
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
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/* ---- 16x6 (store) ---- */
static inline void
gemm_16x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                              const float *RESTRICT Ap,
                              const float *RESTRICT Bp,
                              size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

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
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
                __m256 b = _mm256_broadcast_ss(b_row + j);
                acc_lo[j] = _mm256_fmadd_ps(a_lo, b, acc_lo[j]);
                acc_hi[j] = _mm256_fmadd_ps(a_hi, b, acc_hi[j]);
            }
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 6) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 16 && n == 6)
    {
        alignas(32) float temp_lo[8 * 8] = {0}, temp_hi[8 * 8] = {0};
        for (int j = 0; j < 6; ++j)
        {
            _mm256_store_ps(temp_lo + j * 8, acc_lo[j]);
            _mm256_store_ps(temp_hi + j * 8, acc_hi[j]);
        }
        __m256 cols_lo[8], cols_hi[8];
        for (int j = 0; j < 8; ++j)
        {
            cols_lo[j] = _mm256_load_ps(temp_lo + j * 8);
            cols_hi[j] = _mm256_load_ps(temp_hi + j * 8);
        }
        transpose_8x8_inreg(cols_lo);
        transpose_8x8_inreg(cols_hi);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_lo[r]);
            else
                _mm256_storeu_ps(cr, cols_lo[r]);
        }
        for (size_t r = 8; r < 16; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols_hi[r - 8]);
            else
                _mm256_storeu_ps(cr, cols_hi[r - 8]);
        }
    }
    else
    {
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
            if (use_nt && n == 6)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
    }
}

/* ---- 8x6 (add) ---- */
static inline void
gemm_8x6_panel_avx2fma_add(float *RESTRICT c, size_t ldc,
                           const float *RESTRICT Ap,
                           const float *RESTRICT Bp,
                           size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);

    const size_t PF_LONG = 32;
    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    if (m == 8 && n == 6)
    {
        alignas(32) float temp[8 * 8] = {0};
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = _mm256_load_ps(temp + j * 8);
        transpose_8x8_inreg(cols);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            _mm256_storeu_ps(cr, _mm256_add_ps(_mm256_loadu_ps(cr), cols[r]));
        }
    }
    else
    {
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
            _mm256_maskstore_ps(cr, mask, _mm256_add_ps(old, sum));
        }
    }
}

/* ---- 8x6 (store) ---- */
static inline void
gemm_8x6_panel_avx2fma_store(float *RESTRICT c, size_t ldc,
                             const float *RESTRICT Ap,
                             const float *RESTRICT Bp,
                             size_t Kblk, size_t m, size_t n, __m256i mask)
{
    LINALG_ASSUME_ALIGNED(c, 32);
    LINALG_ASSUME_ALIGNED(Ap, 32);
    LINALG_ASSUME_ALIGNED(Bp, 32);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    const int do_pf = (int)(Kblk >= (size_t)LINALG_GEMM_PF_MIN_K);
    for (size_t r = 0; r < m; r += 4)
        PREFETCH_T0(c + r * ldc);
    const size_t PF_LONG = 32;

    for (size_t k = 0; k < Kblk; k += 8)
    {
        if (do_pf)
        {
            size_t kpf_s = k + 8, kpf_l = k + PF_LONG;
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
            __m256 b;
            b = _mm256_broadcast_ss(b_row + 0);
            acc0 = _mm256_fmadd_ps(a, b, acc0);
            b = _mm256_broadcast_ss(b_row + 1);
            acc1 = _mm256_fmadd_ps(a, b, acc1);
            b = _mm256_broadcast_ss(b_row + 2);
            acc2 = _mm256_fmadd_ps(a, b, acc2);
            b = _mm256_broadcast_ss(b_row + 3);
            acc3 = _mm256_fmadd_ps(a, b, acc3);
            b = _mm256_broadcast_ss(b_row + 4);
            acc4 = _mm256_fmadd_ps(a, b, acc4);
            b = _mm256_broadcast_ss(b_row + 5);
            acc5 = _mm256_fmadd_ps(a, b, acc5);
        }
    }

    const int use_nt = LINALG_NT_STORES &&
                       (n == 6) &&
                       (((uintptr_t)(c) & 31u) == 0) &&
                       ((ldc & 7u) == 0);

    if (m == 8 && n == 6)
    {
        alignas(32) float temp[8 * 8] = {0};
        _mm256_store_ps(temp + 0 * 8, acc0);
        _mm256_store_ps(temp + 1 * 8, acc1);
        _mm256_store_ps(temp + 2 * 8, acc2);
        _mm256_store_ps(temp + 3 * 8, acc3);
        _mm256_store_ps(temp + 4 * 8, acc4);
        _mm256_store_ps(temp + 5 * 8, acc5);
        __m256 cols[8];
        for (int j = 0; j < 8; ++j)
            cols[j] = _mm256_load_ps(temp + j * 8);
        transpose_8x8_inreg(cols);
        for (size_t r = 0; r < 8; ++r)
        {
            float *cr = c + r * ldc;
            if (use_nt)
                _mm256_stream_ps(cr, cols[r]);
            else
                _mm256_storeu_ps(cr, cols[r]);
        }
    }
    else
    {
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
            if (use_nt && n == 6)
                _mm256_stream_ps(cr, sum);
            else
                _mm256_maskstore_ps(cr, mask, sum);
        }
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
    (void)Kblk;
    if (Nblk >= 8 && (Nblk % 8 >= 6 || Nblk >= 3 * (size_t)LINALG_SMALL_N_THRESH))
    {
        if (Mblk >= 16)
            return K16x8;
        if (Mblk >= 8)
            return K8x8;
    }
    if (Mblk >= 16)
        return K16x6;
    return K8x6;
}

/* ======================= Top-level GEMM ======================= */
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
        {pack_A_block_16row_colmajor, pack_A_16row_tile, gemm_16x6_panel_avx2fma_add, gemm_16x6_panel_avx2fma_store, 16, 6, 16},
        {pack_A_block_8row_colmajor, pack_A_block_8row_colmajor, gemm_8x6_panel_avx2fma_add, gemm_8x6_panel_avx2fma_store, 8, 6, 8},
        {pack_A_block_16row_colmajor, pack_A_16row_tile, gemm_16x8_panel_avx2fma_add, gemm_16x8_panel_avx2fma_store, 16, 8, 16},
        {pack_A_block_8row_colmajor, pack_A_block_8row_colmajor, gemm_8x8_panel_avx2fma_add, gemm_8x8_panel_avx2fma_store, 8, 8, 8}};

    const size_t max_nr = 8;
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

            if (kk + Kblk < K && jb_tile >= 64)
            {
                const size_t kk_next = kk + Kblk;
                const size_t step = (size_t)(64 / sizeof(float));
                for (size_t jpf = j0, jpf_end = j0 + jb_tile; jpf < jpf_end; jpf += step)
                    PREFETCH_T1(B + kk_next * N + jpf);
            }

            enum kernel_shape shape = pick_kernel(Mc, jb_tile, Kblk);
            const struct ker *ker = &KERS[shape];
            const size_t NR = ker->NR;
            const size_t n_panels_tile = (jb_tile + NR - 1) / NR;

            size_t panel_off = 0;
            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off += Kblk * NR)
            {
                const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                pack_B_tile(Bp + panel_off, B, K, N, kk, Kblk, j, n_block, NR);
            }

            for (size_t i0 = 0; i0 < M; i0 += Mc)
            {
                const size_t ib_tile = (i0 + Mc <= M) ? Mc : (M - i0);

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

                if (i < ib_tile)
                {
                    size_t m_rem = ib_tile - i;

                    /* If this shape is NR==8, use our 4x8 / 1x8 kernels for ragged rows */
                    if (ker->NR == 8)
                    {
                        /* process 4-row chunks */
                        while (m_rem >= 4)
                        {
                            /* pack 4 rows into ld=8 buffer (pads the upper 4 lanes) */
                            pack_A_block_8row_colmajor(Ap, A, M, K, i0 + i, 4, kk, Kblk);

                            size_t panel_off2 = 0;
                            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR)
                            {
                                const size_t n_block = (j + ker->NR <= j0 + jb_tile) ? ker->NR : (j0 + jb_tile - j);
                                __m256i mask = avx2_tailmask_nr(n_block, ker->NR);
                                float *cptr = C + (i0 + i) * N + j;
                                const float *bptr = Bp + panel_off2;

                                if (kk == 0)
                                    gemm_4x8_panel_avx2fma_store(cptr, N, Ap, bptr, Kblk, n_block, mask);
                                else
                                    gemm_4x8_panel_avx2fma_add(cptr, N, Ap, bptr, Kblk, n_block, mask);
                            }

                            i += 4;
                            m_rem -= 4;
                        }

                        /* process final single row if present */
                        if (m_rem >= 1)
                        {
                            pack_A_block_8row_colmajor(Ap, A, M, K, i0 + i, 1, kk, Kblk);

                            size_t panel_off2 = 0;
                            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR)
                            {
                                const size_t n_block = (j + ker->NR <= j0 + jb_tile) ? ker->NR : (j0 + jb_tile - j);
                                __m256i mask = avx2_tailmask_nr(n_block, ker->NR);
                                float *cptr = C + (i0 + i) * N + j;
                                const float *bptr = Bp + panel_off2;

                                if (kk == 0)
                                    gemm_1x8_panel_avx2fma_store(cptr, Ap, bptr, Kblk, n_block, mask);
                                else
                                    gemm_1x8_panel_avx2fma_add(cptr, Ap, bptr, Kblk, n_block, mask);
                            }

                            i += 1;
                            m_rem -= 1;
                        }
                    }

                    /* if NR!=8 (i.e., NR==6), or anything still remains, use the existing generic tail path */
                    if (i < ib_tile)
                    {
                        const size_t m_block = ib_tile - i;
                        ker->packA_tail(Ap, A, M, K, i0 + i, m_block, kk, Kblk);

                        size_t panel_off2 = 0;
                        for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR)
                        {
                            const size_t n_block = (j + ker->NR <= j0 + jb_tile) ? ker->NR : (j0 + jb_tile - j);
                            __m256i mask = avx2_tailmask_nr(n_block, ker->NR);
                            if (kk == 0)
                                ker->gemm_store(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, m_block, n_block, mask);
                            else
                                ker->gemm_add(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, m_block, n_block, mask);
                        }

                        i += m_block;
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
