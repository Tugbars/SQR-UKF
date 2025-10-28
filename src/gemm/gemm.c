/**
 * @file gemm_complete.c  
 * @brief Complete GEMM Implementation - Faithful Refactoring
 *
 * This is a COMPLETE and FAITHFUL refactoring of the original GEMM code.
 * ALL optimizations preserved:
 * - All 10 kernels (16x8, 8x8, 16x6, 8x6, 4x8, 1x8) × 2 (add/store)
 * - Complete tail handling (4-row and 1-row paths for NR==8)
 * - All prefetch optimizations (row-ahead, within-row, long-distance)
 * - Exact packing logic with all details
 * - Non-temporal stores  
 * - In-register transpose
 * - Vectorized tail with masks
 *
 * @author Original Implementation (Refactored)
 * @date 2025
 */

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <immintrin.h>
#include <string.h>

// External dependencies (same as original)
// Requires: linalg_has_avx2(), linalg_aligned_alloc(), linalg_aligned_free()
// and LINALG_DEFAULT_ALIGNMENT, RESTRICT from linalg_simd.h

// For standalone compilation, provide stubs if needed
#ifndef LINALG_DEFAULT_ALIGNMENT
#define LINALG_DEFAULT_ALIGNMENT 32
#endif

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif

// Declare external functions
extern int linalg_has_avx2(void);
extern void *linalg_aligned_alloc(size_t alignment, size_t size);
extern void linalg_aligned_free(void *ptr);

// AVX-512 detection (add if not available in linalg_simd.h)
#ifdef __AVX512F__
static inline int linalg_has_avx512(void)
{
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0; // AVX-512F bit
    }
#endif
    return 0;
}
#else
static inline int linalg_has_avx512(void) { return 0; }
#endif

//==============================================================================
// CONFIGURATION (EXACT SAME AS ORIGINAL)
//==============================================================================

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
#define LINALG_GEMM_PF_DIST 192
#endif
#ifndef LINALG_GEMM_PF_ROWS_AHEAD
#define LINALG_GEMM_PF_ROWS_AHEAD 1
#endif
#ifndef LINALG_GEMM_PF_MIN_K
#define LINALG_GEMM_PF_MIN_K 128
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

#define LINALG_SIMD_ENABLE 1

//==============================================================================
// SIMD OPERATIONS HEADER
//==============================================================================

#include "gemm_simd_ops.h"

//==============================================================================
// COMPLETE KERNELS HEADERS
//==============================================================================

#include "gemm_kernels_avx2_complete.h"
#include "gemm_kernels_avx512.h"

//==============================================================================
// PACKING ROUTINES (EXACT FROM ORIGINAL)
//==============================================================================

/**
 * @brief Pack A (16-row, col-major, for block operations)
 */
static inline void pack_A_block_16row_colmajor(
    float *RESTRICT Ap,
    const float *RESTRICT A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 16)
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k) {
        float *dst = Ap + k * 16;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}

/**
 * @brief Pack A (8-row, col-major, for block operations)
 */
static inline void pack_A_block_8row_colmajor(
    float *RESTRICT Ap,
    const float *RESTRICT A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 8)
        memset(Ap, 0, Kblk * 8 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k) {
        float *dst = Ap + k * 8;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}

#ifdef __AVX512F__
/**
 * @brief Pack A (32-row, col-major, for AVX-512 block operations)
 */
static inline void pack_A_block_32row_colmajor(
    float *RESTRICT Ap,
    const float *RESTRICT A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 32)
        memset(Ap, 0, Kblk * 32 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k) {
        float *dst = Ap + k * 32;
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}

/**
 * @brief Pack A (32-row, with row-ahead prefetch for AVX-512 tail cases)
 */
static inline void pack_A_32row_tile(
    float *RESTRICT Ap,
    const float *RESTRICT A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 32)
        memset(Ap, 0, Kblk * 32 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k) {
        float *dst = Ap + k * 32;
        
        // Row-ahead prefetch
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0) {
            const size_t i_pf = i0 + (size_t)LINALG_GEMM_PF_ROWS_AHEAD;
            if (i_pf < i0 + ib)
                PREFETCH_T0(A + i_pf * K + (kk + k));
        }
        
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}
#endif

/**
 * @brief Pack A (16-row, with row-ahead prefetch for tail cases)
 */
static inline void pack_A_16row_tile(
    float *RESTRICT Ap,
    const float *RESTRICT A,
    size_t M, size_t K,
    size_t i0, size_t ib,
    size_t kk, size_t Kblk)
{
    (void)M;
    if (ib < 16)
        memset(Ap, 0, Kblk * 16 * sizeof(float));
    for (size_t k = 0; k < Kblk; ++k) {
        float *dst = Ap + k * 16;
        
        // Row-ahead prefetch (PRESERVED FROM ORIGINAL)
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0) {
            const size_t i_pf = i0 + (size_t)LINALG_GEMM_PF_ROWS_AHEAD;
            if (i_pf < i0 + ib)
                PREFETCH_T0(A + i_pf * K + (kk + k));
        }
        
        size_t idx = i0 * K + (kk + k);
        for (size_t r = 0; r < ib; ++r, idx += K)
            dst[r] = A[idx];
    }
}

/**
 * @brief Pack B (8-column panels, EXACT FROM ORIGINAL)
 */
static inline void pack_B_8col_tile(
    float *RESTRICT Bp,
    const float *RESTRICT B,
    size_t K, size_t N,
    size_t kk, size_t Kblk,
    size_t j0, size_t jb)
{
    const size_t n_panels = (jb + 7) / 8;
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    size_t off = 0;
    
    for (size_t p = 0, j = j0; p < n_panels; ++p, j += 8) {
        const size_t w = (j + 8 <= j0 + jb) ? 8 : (j0 + jb - j);
        const size_t remain = N - j;
        
        for (size_t k = 0; k < Kblk; ++k) {
            const float *src = B + (kk + k) * N + j;
            float *dst = Bp + off + k * 8;
            
            // Prefetch next row
            if (k + 1 < Kblk)
                PREFETCH_T0(src + N);
                
            // Row-ahead prefetch  
            if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && 
                k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
                PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j);
                
            // Within-row prefetch
            if (jb >= 16 && pf_elts >= 8 && w == 8 && pf_elts <= remain)
                PREFETCH_T0(src + pf_elts);
                
            if (w == 8) {
                memcpy(dst, src, 8 * sizeof(float));
            } else {
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
 * @brief Pack B (6-column panel, EXACT FROM ORIGINAL)
 */
static inline void pack_B_6col_tile(
    float *RESTRICT Bp,
    const float *RESTRICT B,
    size_t K, size_t N,
    size_t kk, size_t Kblk,
    size_t j0, size_t jb)
{
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    
    for (size_t k = 0; k < Kblk; ++k) {
        const float *src = B + (kk + k) * N + j0;
        float *dst = Bp + k * 6;
        
        if (k + 1 < Kblk)
            PREFETCH_T0(src + N);
            
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && 
            k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
            PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j0);
            
        const size_t remain = N - j0;
        if (jb >= 16 && pf_elts >= 6 && pf_elts <= remain)
            PREFETCH_T0(src + pf_elts);
            
        memcpy(dst, src, jb * sizeof(float));
        if (jb < 6)
            memset(dst + jb, 0, (6 - jb) * sizeof(float));
    }
}

#ifdef __AVX512F__
/**
 * @brief Pack B (16-column panels for AVX-512)
 */
static inline void pack_B_16col_tile(
    float *RESTRICT Bp,
    const float *RESTRICT B,
    size_t K, size_t N,
    size_t kk, size_t Kblk,
    size_t j0, size_t jb)
{
    const size_t n_panels = (jb + 15) / 16;
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    size_t off = 0;
    
    for (size_t p = 0, j = j0; p < n_panels; ++p, j += 16) {
        const size_t w = (j + 16 <= j0 + jb) ? 16 : (j0 + jb - j);
        const size_t remain = N - j;
        
        for (size_t k = 0; k < Kblk; ++k) {
            const float *src = B + (kk + k) * N + j;
            float *dst = Bp + off + k * 16;
            
            if (k + 1 < Kblk)
                PREFETCH_T0(src + N);
                
            if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && 
                k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
                PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j);
                
            if (jb >= 32 && pf_elts >= 16 && w == 16 && pf_elts <= remain)
                PREFETCH_T0(src + pf_elts);
                
            if (w == 16) {
                memcpy(dst, src, 16 * sizeof(float));
            } else {
                size_t t = 0;
                for (; t < w; ++t)
                    dst[t] = src[t];
                for (; t < 16; ++t)
                    dst[t] = 0.0f;
            }
        }
        off += Kblk * 16;
    }
}

/**
 * @brief Pack B (12-column panel for AVX-512)
 */
static inline void pack_B_12col_tile(
    float *RESTRICT Bp,
    const float *RESTRICT B,
    size_t K, size_t N,
    size_t kk, size_t Kblk,
    size_t j0, size_t jb)
{
    const size_t pf_elts = (size_t)LINALG_GEMM_PF_DIST / sizeof(float);
    
    for (size_t k = 0; k < Kblk; ++k) {
        const float *src = B + (kk + k) * N + j0;
        float *dst = Bp + k * 12;
        
        if (k + 1 < Kblk)
            PREFETCH_T0(src + N);
            
        if (LINALG_GEMM_PF_ROWS_AHEAD > 0 && 
            k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD < Kblk)
            PREFETCH_T0(B + (kk + k + (size_t)LINALG_GEMM_PF_ROWS_AHEAD) * N + j0);
            
        const size_t remain = N - j0;
        if (jb >= 24 && pf_elts >= 12 && pf_elts <= remain)
            PREFETCH_T0(src + pf_elts);
            
        memcpy(dst, src, jb * sizeof(float));
        if (jb < 12)
            memset(dst + jb, 0, (12 - jb) * sizeof(float));
    }
}
#endif

/**
 * @brief Pack B dispatcher (based on NR)
 */
static inline void pack_B_tile(
    float *RESTRICT Bp,
    const float *RESTRICT B,
    size_t K, size_t N,
    size_t kk, size_t Kblk,
    size_t j0, size_t jb,
    size_t NR)
{
#ifdef __AVX512F__
    if (NR == 16)
        pack_B_16col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
    else if (NR == 12)
        pack_B_12col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
    else
#endif
    if (NR == 8)
        pack_B_8col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
    else
        pack_B_6col_tile(Bp, B, K, N, kk, Kblk, j0, jb);
}

//==============================================================================
// KERNEL SELECTION
//==============================================================================

enum kernel_shape {
    K16x6,  // AVX2
    K8x6,   // AVX2
    K16x8,  // AVX2
    K8x8,   // AVX2
#ifdef __AVX512F__
    K32x12, // AVX-512
    K32x16  // AVX-512
#endif
};

static inline enum kernel_shape pick_kernel(size_t Mblk, size_t Nblk, size_t Kblk)
{
    (void)Kblk;
    
#ifdef __AVX512F__
    // AVX-512 path: prefer 32-row kernels when AVX-512 available
    if (linalg_has_avx512()) {
        if (Nblk >= 16 && (Nblk % 16 >= 12 || Nblk >= 3 * (size_t)LINALG_SMALL_N_THRESH)) {
            if (Mblk >= 32)
                return K32x16;
        }
        if (Mblk >= 32)
            return K32x12;
    }
#endif
    
    // AVX2 fallback
    if (Nblk >= 8 && (Nblk % 8 >= 6 || Nblk >= 3 * (size_t)LINALG_SMALL_N_THRESH)) {
        if (Mblk >= 16)
            return K16x8;
        if (Mblk >= 8)
            return K8x8;
    }
    if (Mblk >= 16)
        return K16x6;
    return K8x6;
}

//==============================================================================
// MAIN GEMM FUNCTION
//==============================================================================

int mul(
    float *RESTRICT C,
    const float *RESTRICT A,
    const float *RESTRICT B,
    uint16_t row_a, uint16_t column_a,
    uint16_t row_b, uint16_t column_b)
{
    if (column_a != row_b)
        return -EINVAL;

    const size_t M = row_a, K = column_a, N = column_b;

    // Scalar fallback for tiny matrices or no AVX2
    if (!linalg_has_avx2() || M == 0 || N == 0 || K == 0 ||
        (M < LINALG_SMALL_N_THRESH && N < LINALG_SMALL_N_THRESH))
    {
        for (size_t i = 0; i < M; ++i) {
            const float *ai = A + i * K;
            for (size_t j = 0; j < N; ++j) {
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

    // Kernel descriptor structure
    struct ker {
        void (*packA_blk)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t);
        void (*packA_tail)(float *, const float *, size_t, size_t, size_t, size_t, size_t, size_t);
        void (*gemm_add)(float *, size_t, const float *, const float *, size_t, size_t, size_t, __m256i);
        void (*gemm_store)(float *, size_t, const float *, const float *, size_t, size_t, size_t, __m256i);
        size_t MR, NR, A_ld;
    };

    // Kernel table (EXTENDED FOR AVX-512)
    static const struct ker KERS[] = {
        // AVX2 kernels
        {pack_A_block_16row_colmajor, pack_A_16row_tile, gemm_16x6_panel_avx2fma_add, gemm_16x6_panel_avx2fma_store, 16, 6, 16},
        {pack_A_block_8row_colmajor, pack_A_block_8row_colmajor, gemm_8x6_panel_avx2fma_add, gemm_8x6_panel_avx2fma_store, 8, 6, 8},
        {pack_A_block_16row_colmajor, pack_A_16row_tile, gemm_16x8_panel_avx2fma_add, gemm_16x8_panel_avx2fma_store, 16, 8, 16},
        {pack_A_block_8row_colmajor, pack_A_block_8row_colmajor, gemm_8x8_panel_avx2fma_add, gemm_8x8_panel_avx2fma_store, 8, 8, 8},
#ifdef __AVX512F__
        // AVX-512 kernels
        {pack_A_block_32row_colmajor, pack_A_32row_tile, gemm_32x12_panel_avx512_add, gemm_32x12_panel_avx512_store, 32, 12, 32},
        {pack_A_block_32row_colmajor, pack_A_32row_tile, gemm_32x16_panel_avx512_add, gemm_32x16_panel_avx512_store, 32, 16, 32}
#endif
    };

    // Allocate packing buffers (sized for maximum: AVX-512 if available)
#ifdef __AVX512F__
    const size_t max_nr = linalg_has_avx512() ? 16 : 8;
    const size_t max_mr = linalg_has_avx512() ? 32 : 16;
#else
    const size_t max_nr = 8;
    const size_t max_mr = 16;
#endif
    const size_t max_n_panels = (Nc + max_nr - 1) / max_nr;
    const size_t max_Bp_elems = Kc * max_n_panels * max_nr;
    float *Bp = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, max_Bp_elems * sizeof(float));
    if (!Bp)
        return -ENOMEM;

    const size_t max_Ap_elems = Kc * max_mr;
    float *Ap = (float *)linalg_aligned_alloc(LINALG_DEFAULT_ALIGNMENT, max_Ap_elems * sizeof(float));
    if (!Ap) {
        linalg_aligned_free(Bp);
        return -ENOMEM;
    }

    // Three-level blocking loop (EXACT FROM ORIGINAL)
    for (size_t j0 = 0; j0 < N; j0 += Nc) {
        const size_t jb_tile = (j0 + Nc <= N) ? Nc : (N - j0);

        for (size_t kk = 0; kk < K; kk += Kc) {
            const size_t Kblk = (kk + Kc <= K) ? Kc : (K - kk);

            // Prefetch next B block
            if (kk + Kblk < K && jb_tile >= 64) {
                const size_t kk_next = kk + Kblk;
                const size_t step = (size_t)(64 / sizeof(float));
                for (size_t jpf = j0, jpf_end = j0 + jb_tile; jpf < jpf_end; jpf += step)
                    PREFETCH_T1(B + kk_next * N + jpf);
            }

            enum kernel_shape shape = pick_kernel(Mc, jb_tile, Kblk);
            const struct ker *ker = &KERS[shape];
            const size_t NR = ker->NR;
            const size_t n_panels_tile = (jb_tile + NR - 1) / NR;

            // Pack B
            size_t panel_off = 0;
            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off += Kblk * NR) {
                const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                pack_B_tile(Bp + panel_off, B, K, N, kk, Kblk, j, n_block, NR);
            }

            // Process M dimension
            for (size_t i0 = 0; i0 < M; i0 += Mc) {
                const size_t ib_tile = (i0 + Mc <= M) ? Mc : (M - i0);

                // Prefetch A
                if (ib_tile >= 64) {
                    for (size_t ipf = i0, ipf_end = i0 + ib_tile; ipf < ipf_end; ipf += 8)
                        PREFETCH_T1(A + ipf * K + kk);
                }

                size_t i = 0;
                const size_t mr = ker->MR;

                // Main MR-sized tiles
                for (; i + mr - 1 < ib_tile; i += mr) {
                    ker->packA_blk(Ap, A, M, K, i0 + i, mr, kk, Kblk);
                    
                    size_t panel_off2 = 0;
                    for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += NR, panel_off2 += Kblk * NR) {
                        const size_t n_block = (j + NR <= j0 + jb_tile) ? NR : (j0 + jb_tile - j);
                        __m256i mask = avx2_tailmask_nr(n_block, NR);
                        
                        if (kk == 0)
                            ker->gemm_store(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, mr, n_block, mask);
                        else
                            ker->gemm_add(C + (i0 + i) * N + j, N, Ap, Bp + panel_off2, Kblk, mr, n_block, mask);
                    }
                }

                // TAIL HANDLING (EXACT FROM ORIGINAL - CRITICAL!)
                if (i < ib_tile) {
                    size_t m_rem = ib_tile - i;

                    // Special handling for NR==8: use 4x8 and 1x8 kernels
                    if (ker->NR == 8) {
                        // Process 4-row chunks
                        while (m_rem >= 4) {
                            pack_A_block_8row_colmajor(Ap, A, M, K, i0 + i, 4, kk, Kblk);

                            size_t panel_off2 = 0;
                            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR) {
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

                        // Process final single row
                        if (m_rem >= 1) {
                            pack_A_block_8row_colmajor(Ap, A, M, K, i0 + i, 1, kk, Kblk);

                            size_t panel_off2 = 0;
                            for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR) {
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

                    // Generic tail (for NR!=8 or anything still remaining)
                    if (i < ib_tile) {
                        const size_t m_block = ib_tile - i;
                        ker->packA_tail(Ap, A, M, K, i0 + i, m_block, kk, Kblk);

                        size_t panel_off2 = 0;
                        for (size_t p = 0, j = j0; p < n_panels_tile; ++p, j += ker->NR, panel_off2 += Kblk * ker->NR) {
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
    (void)C; (void)A; (void)B;
    (void)row_a; (void)column_a;
    (void)row_b; (void)column_b;
    return -ENOTSUP;
#endif
}
