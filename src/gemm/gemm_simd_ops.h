/**
 * @file gemm_simd_ops.h
 * @brief SIMD Operation Abstractions for High-Performance GEMM
 *
 * @details
 * This header provides uniform SIMD abstractions for matrix multiplication
 * operations across multiple instruction sets (AVX2, SSE2, scalar).
 *
 * Design Philosophy:
 * - Consistent naming across all SIMD levels
 * - Zero-overhead abstractions (inline macros)
 * - Easy to add AVX-512 support later
 * - Clear separation between interface and implementation
 *
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 *
 * @section usage USAGE IN IMPLEMENTATION
 * 
 * 1. **Include this header** before kernel implementations
 * 2. **Use GEMM_SIMD_SELECT_* macros** to dispatch to correct SIMD level
 * 3. **Compile with appropriate flags**: -mavx2 -mfma or -msse2
 * 
 * Example:
 * @code
 * #include "gemm_simd_ops.h"
 * 
 * void my_kernel() {
 *     #if defined(__AVX2__)
 *         __m256 acc = GEMM_VEC_ZERO_PS_AVX2();
 *         __m256 a = GEMM_LOAD_PS_AVX2(ptr);
 *         acc = GEMM_FMADD_PS_AVX2(a, b, acc);
 *     #endif
 * }
 * @endcode
 */

#ifndef GEMM_SIMD_OPS_H
#define GEMM_SIMD_OPS_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def GEMM_VECTOR_WIDTH_AVX2
 * @brief Number of single-precision floats in AVX2 vector
 */
#define GEMM_VECTOR_WIDTH_AVX2 8

/**
 * @def GEMM_VECTOR_WIDTH_SSE2
 * @brief Number of single-precision floats in SSE2 vector
 */
#define GEMM_VECTOR_WIDTH_SSE2 4

/**
 * @def GEMM_ALIGNMENT
 * @brief Required alignment for packed buffers (bytes)
 */
#define GEMM_ALIGNMENT 32

//==============================================================================
// AVX2 OPERATIONS (256-bit, 8 x float32)
//==============================================================================

#ifdef __AVX2__

/**
 * @name AVX2 Vector Types
 * @{
 */
typedef __m256  gemm_vec_ps_avx2_t;  ///< AVX2 single-precision vector
typedef __m256i gemm_mask_avx2_t;    ///< AVX2 mask for masked operations
/** @} */

/**
 * @name AVX2 Basic Operations
 * @{
 */

/** @brief Zero vector */
#define GEMM_VEC_ZERO_PS_AVX2() \
    _mm256_setzero_ps()

/** @brief Aligned load (32-byte aligned) */
#define GEMM_LOAD_PS_AVX2(ptr) \
    _mm256_load_ps(ptr)

/** @brief Unaligned load */
#define GEMM_LOADU_PS_AVX2(ptr) \
    _mm256_loadu_ps(ptr)

/** @brief Aligned store (32-byte aligned) */
#define GEMM_STORE_PS_AVX2(ptr, vec) \
    _mm256_store_ps(ptr, vec)

/** @brief Unaligned store */
#define GEMM_STOREU_PS_AVX2(ptr, vec) \
    _mm256_storeu_ps(ptr, vec)

/** @brief Broadcast single float to all lanes */
#define GEMM_BROADCAST_SS_AVX2(ptr) \
    _mm256_broadcast_ss(ptr)

/** @brief Non-temporal store (bypass cache) */
#define GEMM_STREAM_PS_AVX2(ptr, vec) \
    _mm256_stream_ps(ptr, vec)

/** @} */

/**
 * @name AVX2 Arithmetic Operations
 * @{
 */

#if defined(__FMA__)
/** @brief Fused multiply-add: a*b + c */
#define GEMM_FMADD_PS_AVX2(a, b, c) \
    _mm256_fmadd_ps(a, b, c)

/** @brief Fused multiply-subtract: a*b - c */
#define GEMM_FMSUB_PS_AVX2(a, b, c) \
    _mm256_fmsub_ps(a, b, c)
#else
/** @brief Multiply-add fallback without FMA */
#define GEMM_FMADD_PS_AVX2(a, b, c) \
    _mm256_add_ps(_mm256_mul_ps(a, b), c)

/** @brief Multiply-subtract fallback without FMA */
#define GEMM_FMSUB_PS_AVX2(a, b, c) \
    _mm256_sub_ps(_mm256_mul_ps(a, b), c)
#endif

/** @brief Addition */
#define GEMM_ADD_PS_AVX2(a, b) \
    _mm256_add_ps(a, b)

/** @brief Subtraction */
#define GEMM_SUB_PS_AVX2(a, b) \
    _mm256_sub_ps(a, b)

/** @brief Multiplication */
#define GEMM_MUL_PS_AVX2(a, b) \
    _mm256_mul_ps(a, b)

/** @} */

/**
 * @name AVX2 Masked Operations
 * @{
 */

/** @brief Masked load (load only active lanes) */
#define GEMM_MASKLOAD_PS_AVX2(ptr, mask) \
    _mm256_maskload_ps(ptr, mask)

/** @brief Masked store (store only active lanes) */
#define GEMM_MASKSTORE_PS_AVX2(ptr, mask, vec) \
    _mm256_maskstore_ps(ptr, mask, vec)

/** @} */

/**
 * @name AVX2 Transpose Operations
 * @{
 */

/** @brief Unpack low elements */
#define GEMM_UNPACKLO_PS_AVX2(a, b) \
    _mm256_unpacklo_ps(a, b)

/** @brief Unpack high elements */
#define GEMM_UNPACKHI_PS_AVX2(a, b) \
    _mm256_unpackhi_ps(a, b)

/** @brief Shuffle within 128-bit lanes */
#define GEMM_SHUFFLE_PS_AVX2(a, b, imm) \
    _mm256_shuffle_ps(a, b, imm)

/** @brief Permute 128-bit lanes */
#define GEMM_PERMUTE2F128_PS_AVX2(a, b, imm) \
    _mm256_permute2f128_ps(a, b, imm)

/** @} */

/**
 * @name AVX2 Prefetch Operations
 * @{
 */

/** @brief Prefetch to L1 cache */
#define GEMM_PREFETCH_T0_AVX2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

/** @brief Prefetch to L2 cache */
#define GEMM_PREFETCH_T1_AVX2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T1)

/** @brief Prefetch to L3 cache */
#define GEMM_PREFETCH_T2_AVX2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T2)

/** @} */

#endif /* __AVX2__ */

//==============================================================================
// SSE2 OPERATIONS (128-bit, 4 x float32)
//==============================================================================

#ifdef __SSE2__

/**
 * @name SSE2 Vector Types
 * @{
 */
typedef __m128  gemm_vec_ps_sse2_t;  ///< SSE2 single-precision vector
typedef __m128i gemm_mask_sse2_t;    ///< SSE2 mask for masked operations
/** @} */

/**
 * @name SSE2 Basic Operations
 * @{
 */

/** @brief Zero vector */
#define GEMM_VEC_ZERO_PS_SSE2() \
    _mm_setzero_ps()

/** @brief Aligned load (16-byte aligned) */
#define GEMM_LOAD_PS_SSE2(ptr) \
    _mm_load_ps(ptr)

/** @brief Unaligned load */
#define GEMM_LOADU_PS_SSE2(ptr) \
    _mm_loadu_ps(ptr)

/** @brief Aligned store (16-byte aligned) */
#define GEMM_STORE_PS_SSE2(ptr, vec) \
    _mm_store_ps(ptr, vec)

/** @brief Unaligned store */
#define GEMM_STOREU_PS_SSE2(ptr, vec) \
    _mm_storeu_ps(ptr, vec)

/** @brief Broadcast single float to all lanes (SSE3+) */
#ifdef __SSE3__
#define GEMM_BROADCAST_SS_SSE2(ptr) \
    _mm_load_ps1(ptr)
#else
#define GEMM_BROADCAST_SS_SSE2(ptr) \
    _mm_set1_ps(*(ptr))
#endif

/** @brief Non-temporal store (bypass cache) */
#define GEMM_STREAM_PS_SSE2(ptr, vec) \
    _mm_stream_ps(ptr, vec)

/** @} */

/**
 * @name SSE2 Arithmetic Operations
 * @{
 */

/** @brief Fused multiply-add: a*b + c (emulated, no native FMA in SSE2) */
#define GEMM_FMADD_PS_SSE2(a, b, c) \
    _mm_add_ps(_mm_mul_ps(a, b), c)

/** @brief Fused multiply-subtract: a*b - c (emulated) */
#define GEMM_FMSUB_PS_SSE2(a, b, c) \
    _mm_sub_ps(_mm_mul_ps(a, b), c)

/** @brief Addition */
#define GEMM_ADD_PS_SSE2(a, b) \
    _mm_add_ps(a, b)

/** @brief Subtraction */
#define GEMM_SUB_PS_SSE2(a, b) \
    _mm_sub_ps(a, b)

/** @brief Multiplication */
#define GEMM_MUL_PS_SSE2(a, b) \
    _mm_mul_ps(a, b)

/** @} */

/**
 * @name SSE2 Transpose Operations
 * @{
 */

/** @brief Unpack low elements */
#define GEMM_UNPACKLO_PS_SSE2(a, b) \
    _mm_unpacklo_ps(a, b)

/** @brief Unpack high elements */
#define GEMM_UNPACKHI_PS_SSE2(a, b) \
    _mm_unpackhi_ps(a, b)

/** @brief Shuffle elements */
#define GEMM_SHUFFLE_PS_SSE2(a, b, imm) \
    _mm_shuffle_ps(a, b, imm)

/** @brief Move low half to high half */
#define GEMM_MOVEHL_PS_SSE2(a, b) \
    _mm_movehl_ps(a, b)

/** @brief Move high half to low half */
#define GEMM_MOVELH_PS_SSE2(a, b) \
    _mm_movelh_ps(a, b)

/** @} */

/**
 * @name SSE2 Prefetch Operations
 * @{
 */

/** @brief Prefetch to L1 cache */
#define GEMM_PREFETCH_T0_SSE2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

/** @brief Prefetch to L2 cache */
#define GEMM_PREFETCH_T1_SSE2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T1)

/** @brief Prefetch to L3 cache */
#define GEMM_PREFETCH_T2_SSE2(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T2)

/** @} */

/**
 * @note SSE2 does not have native masked load/store
 * Use scalar loops for tail handling with SSE2
 */

#endif /* __SSE2__ */

//==============================================================================
// AVX-512 OPERATIONS (512-bit, 16 x float32)
//==============================================================================

#ifdef __AVX512F__

/**
 * @name AVX-512 Vector Types
 * @{
 */
typedef __m512  gemm_vec_ps_avx512_t;   ///< AVX-512 single-precision vector
typedef __mmask16 gemm_mask_avx512_t;   ///< AVX-512 mask for masked operations
/** @} */

/**
 * @def GEMM_VECTOR_WIDTH_AVX512
 * @brief Number of single-precision floats in AVX-512 vector
 */
#define GEMM_VECTOR_WIDTH_AVX512 16

/**
 * @name AVX-512 Basic Operations
 * @{
 */

/** @brief Zero vector */
#define GEMM_VEC_ZERO_PS_AVX512() \
    _mm512_setzero_ps()

/** @brief Aligned load (64-byte aligned) */
#define GEMM_LOAD_PS_AVX512(ptr) \
    _mm512_load_ps(ptr)

/** @brief Unaligned load */
#define GEMM_LOADU_PS_AVX512(ptr) \
    _mm512_loadu_ps(ptr)

/** @brief Aligned store (64-byte aligned) */
#define GEMM_STORE_PS_AVX512(ptr, vec) \
    _mm512_store_ps(ptr, vec)

/** @brief Unaligned store */
#define GEMM_STOREU_PS_AVX512(ptr, vec) \
    _mm512_storeu_ps(ptr, vec)

/** @brief Broadcast single float to all lanes */
#define GEMM_BROADCAST_SS_AVX512(ptr) \
    _mm512_set1_ps(*(ptr))

/** @brief Non-temporal store (bypass cache) */
#define GEMM_STREAM_PS_AVX512(ptr, vec) \
    _mm512_stream_ps(ptr, vec)

/** @} */

/**
 * @name AVX-512 Arithmetic Operations
 * @{
 */

/** @brief Fused multiply-add: a*b + c */
#define GEMM_FMADD_PS_AVX512(a, b, c) \
    _mm512_fmadd_ps(a, b, c)

/** @brief Fused multiply-subtract: a*b - c */
#define GEMM_FMSUB_PS_AVX512(a, b, c) \
    _mm512_fmsub_ps(a, b, c)

/** @brief Addition */
#define GEMM_ADD_PS_AVX512(a, b) \
    _mm512_add_ps(a, b)

/** @brief Subtraction */
#define GEMM_SUB_PS_AVX512(a, b) \
    _mm512_sub_ps(a, b)

/** @brief Multiplication */
#define GEMM_MUL_PS_AVX512(a, b) \
    _mm512_mul_ps(a, b)

/** @} */

/**
 * @name AVX-512 Masked Operations
 * @{
 */

/** @brief Masked load (load only active lanes) */
#define GEMM_MASKLOAD_PS_AVX512(ptr, mask) \
    _mm512_maskz_loadu_ps(mask, ptr)

/** @brief Masked store (store only active lanes) */
#define GEMM_MASKSTORE_PS_AVX512(ptr, mask, vec) \
    _mm512_mask_storeu_ps(ptr, mask, vec)

/** @brief Build mask from count (0-16 active lanes) */
#define GEMM_BUILD_MASK_AVX512(count) \
    ((__mmask16)((1u << (count)) - 1u))

/** @} */

/**
 * @name AVX-512 Transpose Operations
 * @{
 */

/** @brief Unpack low elements */
#define GEMM_UNPACKLO_PS_AVX512(a, b) \
    _mm512_unpacklo_ps(a, b)

/** @brief Unpack high elements */
#define GEMM_UNPACKHI_PS_AVX512(a, b) \
    _mm512_unpackhi_ps(a, b)

/** @brief Shuffle within 128-bit lanes */
#define GEMM_SHUFFLE_PS_AVX512(a, b, imm) \
    _mm512_shuffle_ps(a, b, imm)

/** @brief Permute 128-bit lanes */
#define GEMM_SHUFFLE_F32X4_AVX512(a, b, imm) \
    _mm512_shuffle_f32x4(a, b, imm)

/** @brief Permutex (full cross-lane permutation) */
#define GEMM_PERMUTEX_PS_AVX512(idx, a) \
    _mm512_permutexvar_ps(idx, a)

/** @} */

/**
 * @name AVX-512 Prefetch Operations
 * @{
 */

/** @brief Prefetch to L1 cache */
#define GEMM_PREFETCH_T0_AVX512(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

/** @brief Prefetch to L2 cache */
#define GEMM_PREFETCH_T1_AVX512(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T1)

/** @brief Prefetch to L3 cache */
#define GEMM_PREFETCH_T2_AVX512(ptr) \
    _mm_prefetch((const char*)(ptr), _MM_HINT_T2)

/** @} */

/**
 * @brief In-register 16×16 transpose for AVX-512
 *
 * @details
 * Transposes 16 vectors of 16 floats each, entirely in registers.
 * Uses AVX-512 shuffle/permute instructions for optimal performance.
 *
 * Algorithm is similar to 8×8 AVX2 transpose but scaled to 16×16.
 * Total: ~48 instructions, no memory traffic
 *
 * @param[in,out] rows Array of 16 vectors (input: columns, output: rows)
 */
static inline void gemm_transpose_16x16_avx512(__m512 *rows)
{
    // Step 1: Unpack pairs within 128-bit lanes (16 operations)
    __m512 t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i]   = GEMM_UNPACKLO_PS_AVX512(rows[i], rows[i+1]);
        t[i+1] = GEMM_UNPACKHI_PS_AVX512(rows[i], rows[i+1]);
    }
    
    // Step 2: Shuffle 4-element blocks (16 operations)
    __m512 tt[16];
    for (int i = 0; i < 16; i += 4) {
        tt[i]   = GEMM_SHUFFLE_PS_AVX512(t[i],   t[i+2], 0x44);
        tt[i+1] = GEMM_SHUFFLE_PS_AVX512(t[i],   t[i+2], 0xEE);
        tt[i+2] = GEMM_SHUFFLE_PS_AVX512(t[i+1], t[i+3], 0x44);
        tt[i+3] = GEMM_SHUFFLE_PS_AVX512(t[i+1], t[i+3], 0xEE);
    }
    
    // Step 3: Permute 128-bit lanes (16 operations)
    for (int i = 0; i < 8; i++) {
        rows[i]   = GEMM_SHUFFLE_F32X4_AVX512(tt[i], tt[i+8], 0x88);
        rows[i+8] = GEMM_SHUFFLE_F32X4_AVX512(tt[i], tt[i+8], 0xDD);
    }
}

/**
 * @brief Build mask for n active lanes (clamped to NR)
 *
 * @param n Number of active columns
 * @param NR Panel width (12 or 16)
 * @return Mask for min(n, NR) lanes
 */
static inline __mmask16 gemm_tailmask_nr_avx512(size_t n, size_t NR)
{
    size_t lanes = (n <= NR) ? n : NR;
    return GEMM_BUILD_MASK_AVX512(lanes);
}

#endif /* __AVX512F__ */

//==============================================================================
// SCALAR OPERATIONS (Portable fallback)
//==============================================================================

/**
 * @name Scalar Operations
 * @brief Portable scalar fallback operations
 * @{
 */

/** @brief Scalar type for single float */
typedef float gemm_scalar_ps_t;

/** @brief Scalar zero */
#define GEMM_SCALAR_ZERO_PS() (0.0f)

/** @brief Scalar load */
#define GEMM_SCALAR_LOAD_PS(ptr) (*(ptr))

/** @brief Scalar store */
#define GEMM_SCALAR_STORE_PS(ptr, val) (*(ptr) = (val))

/** @brief Scalar FMA: a*b + c */
#define GEMM_SCALAR_FMADD_PS(a, b, c) ((a) * (b) + (c))

/** @brief Scalar add */
#define GEMM_SCALAR_ADD_PS(a, b) ((a) + (b))

/** @brief Scalar subtract */
#define GEMM_SCALAR_SUB_PS(a, b) ((a) - (b))

/** @brief Scalar multiply */
#define GEMM_SCALAR_MUL_PS(a, b) ((a) * (b))

/** @brief Scalar prefetch (no-op) */
#define GEMM_SCALAR_PREFETCH_T0(ptr) ((void)0)
#define GEMM_SCALAR_PREFETCH_T1(ptr) ((void)0)

/** @} */

//==============================================================================
// HELPER: BUILD TAIL MASK FOR AVX2
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Build AVX2 mask for n active lanes (0 ≤ n ≤ 8)
 *
 * @details
 * Creates a mask suitable for maskload/maskstore operations.
 * Active lanes have all bits set (-1), inactive lanes are zero.
 *
 * Uses a precomputed lookup table for efficiency.
 *
 * @param lanes Number of active lanes (0-8)
 * @return __m256i mask with first 'lanes' elements active
 *
 * @note This is inline to allow compiler optimizations
 */
static inline __m256i gemm_build_mask_avx2(int lanes)
{
    // Lookup table: 8-bit masks extended to 32-bit lanes
    static const int8_t kMask8x8[9][8] __attribute__((aligned(64))) = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {-1, 0, 0, 0, 0, 0, 0, 0},
        {-1, -1, 0, 0, 0, 0, 0, 0},
        {-1, -1, -1, 0, 0, 0, 0, 0},
        {-1, -1, -1, -1, 0, 0, 0, 0},
        {-1, -1, -1, -1, -1, 0, 0, 0},
        {-1, -1, -1, -1, -1, -1, 0, 0},
        {-1, -1, -1, -1, -1, -1, -1, 0},
        {-1, -1, -1, -1, -1, -1, -1, -1}
    };
    
    // Load 8 bytes and extend to 8 x 32-bit integers
    __m128i b8 = _mm_loadl_epi64((const __m128i *)kMask8x8[lanes]);
    return _mm256_cvtepi8_epi32(b8);
}

/**
 * @brief Build mask based on active columns (clamped to NR)
 *
 * @param n Number of active columns
 * @param NR Panel width (6 or 8)
 * @return Mask for min(n, NR) lanes
 */
static inline __m256i gemm_tailmask_nr(size_t n, size_t NR)
{
    size_t lanes = (n <= NR) ? n : NR;
    return gemm_build_mask_avx2((int)lanes);
}

#endif /* __AVX2__ */

//==============================================================================
// TRANSPOSE HELPERS
//==============================================================================

#ifdef __AVX2__

/**
 * @brief In-register 8×8 transpose for AVX2
 *
 * @details
 * Transposes 8 vectors of 8 floats each, entirely in registers.
 * Uses AVX2 shuffle/permute instructions for optimal performance.
 *
 * Algorithm:
 * 1. Unpack pairs (interleave adjacent elements)
 * 2. Shuffle 4-element blocks
 * 3. Permute 128-bit lanes
 *
 * Total: 24 instructions, no memory traffic
 *
 * @param[in,out] rows Array of 8 vectors (input: columns, output: rows)
 *
 * @note Input are columns, output are rows (transpose operation)
 */
static inline void gemm_transpose_8x8_avx2(__m256 *rows)
{
    // Step 1: Unpack pairs (interleave adjacent columns)
    __m256 t0 = GEMM_UNPACKLO_PS_AVX2(rows[0], rows[1]);
    __m256 t1 = GEMM_UNPACKHI_PS_AVX2(rows[0], rows[1]);
    __m256 t2 = GEMM_UNPACKLO_PS_AVX2(rows[2], rows[3]);
    __m256 t3 = GEMM_UNPACKHI_PS_AVX2(rows[2], rows[3]);
    __m256 t4 = GEMM_UNPACKLO_PS_AVX2(rows[4], rows[5]);
    __m256 t5 = GEMM_UNPACKHI_PS_AVX2(rows[4], rows[5]);
    __m256 t6 = GEMM_UNPACKLO_PS_AVX2(rows[6], rows[7]);
    __m256 t7 = GEMM_UNPACKHI_PS_AVX2(rows[6], rows[7]);

    // Step 2: Shuffle 4-element blocks
    __m256 tt0 = GEMM_SHUFFLE_PS_AVX2(t0, t2, 0x44);
    __m256 tt1 = GEMM_SHUFFLE_PS_AVX2(t0, t2, 0xEE);
    __m256 tt2 = GEMM_SHUFFLE_PS_AVX2(t1, t3, 0x44);
    __m256 tt3 = GEMM_SHUFFLE_PS_AVX2(t1, t3, 0xEE);
    __m256 tt4 = GEMM_SHUFFLE_PS_AVX2(t4, t6, 0x44);
    __m256 tt5 = GEMM_SHUFFLE_PS_AVX2(t4, t6, 0xEE);
    __m256 tt6 = GEMM_SHUFFLE_PS_AVX2(t5, t7, 0x44);
    __m256 tt7 = GEMM_SHUFFLE_PS_AVX2(t5, t7, 0xEE);

    // Step 3: Permute 128-bit lanes (final transpose)
    rows[0] = GEMM_PERMUTE2F128_PS_AVX2(tt0, tt4, 0x20);
    rows[1] = GEMM_PERMUTE2F128_PS_AVX2(tt1, tt5, 0x20);
    rows[2] = GEMM_PERMUTE2F128_PS_AVX2(tt2, tt6, 0x20);
    rows[3] = GEMM_PERMUTE2F128_PS_AVX2(tt3, tt7, 0x20);
    rows[4] = GEMM_PERMUTE2F128_PS_AVX2(tt0, tt4, 0x31);
    rows[5] = GEMM_PERMUTE2F128_PS_AVX2(tt1, tt5, 0x31);
    rows[6] = GEMM_PERMUTE2F128_PS_AVX2(tt2, tt6, 0x31);
    rows[7] = GEMM_PERMUTE2F128_PS_AVX2(tt3, tt7, 0x31);
}

#endif /* __AVX2__ */

//==============================================================================
// ALIGNMENT HINTS (Compiler-specific)
//==============================================================================

/**
 * @brief Tell compiler a pointer is aligned
 *
 * @details
 * Allows compiler to generate more efficient code for aligned loads/stores.
 * Use after verifying alignment at runtime.
 *
 * @param p Pointer to align
 * @param n Alignment in bytes (must be power of 2)
 */
#if defined(__GNUC__) || defined(__clang__)
#define GEMM_ASSUME_ALIGNED(p, n) \
    (p) = (__typeof__(p))__builtin_assume_aligned((p), (n))
#elif defined(_MSC_VER)
#define GEMM_ASSUME_ALIGNED(p, n) \
    __assume((((uintptr_t)(p)) & ((n)-1)) == 0)
#else
#define GEMM_ASSUME_ALIGNED(p, n) ((void)0)
#endif

/**
 * @brief Mark pointer as restricted (no aliasing)
 */
#if defined(__GNUC__) || defined(__clang__)
#define GEMM_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define GEMM_RESTRICT __restrict
#else
#define GEMM_RESTRICT
#endif

//==============================================================================
// CONDITIONAL COMPILATION HELPERS
//==============================================================================

/**
 * @brief Check if prefetching is enabled
 */
#ifndef GEMM_PREFETCH_ENABLE
#define GEMM_PREFETCH_ENABLE 1
#endif

#if GEMM_PREFETCH_ENABLE
#define GEMM_PREFETCH_T0(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#define GEMM_PREFETCH_T1(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
#else
#define GEMM_PREFETCH_T0(ptr) ((void)0)
#define GEMM_PREFETCH_T1(ptr) ((void)0)
#endif

#endif /* GEMM_SIMD_OPS_H */
