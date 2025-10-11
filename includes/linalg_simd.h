/* SPDX-License-Identifier: MIT */
#ifndef CONTROL_LINALG_SIMD_H
#define CONTROL_LINALG_SIMD_H

/**
 * @file linalg_simd.h
 * @brief Vectorized linear algebra kernels (SIMD-accelerated) + tuning knobs.
 *
 * This header exposes the public APIs for the vectorized kernels you implemented:
 *  - tran         : matrix transpose (blocked AVX2 8x8 micro-kernel + scalar tails)
 *  - mul          : GEMV/GEMM-lite (row-major A*B, AVX2 FMA inner loops, scalar/SSE tails)
 *  - lup          : LU with partial pivoting (rank-1 updates accelerated via AVX2)
 *  - inv          : matrix inverse via LUP + triangular solves (in-place safe per API)
 *  - qr           : QR decomposition (Householder) with vectorized reflectors and blocked updates
 *  - cholupdate   : rank-one Cholesky update/downdate (vectorized axpy lanes)
 *
 * All kernels are safe to compile on non-AVX2 targets; they auto-fallback to scalar paths.
 *
 * Memory layout: row-major throughout.
 * Aliasing: unless stated otherwise, output buffers must not alias inputs.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Standard includes (kept minimal for headers) ---------- */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdalign.h>  // alignas (used later in these files)
#include <float.h>
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#  include <malloc.h>   /* _aligned_malloc / __mingw_aligned_malloc */
#endif

static inline int ukf_has_avx2(void) {
#if defined(__AVX2__) && defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

/* ---------- Portability helpers ---------- */
#ifndef RESTRICT
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#    define RESTRICT restrict
#  else
#    define RESTRICT
#  endif
#endif

/* ---------- Feature gates & tuning knobs ---------- */
/**
 * @def LINALG_SIMD_ENABLE
 * Enable SIMD fast paths when the compiler target supports them.
 * Defaults to 1 if __AVX2__ and __FMA__ are defined; otherwise 0.
 */
#ifndef LINALG_SIMD_ENABLE
#  if defined(__AVX2__) && defined(__FMA__)
#    define LINALG_SIMD_ENABLE 1
#  else
#    define LINALG_SIMD_ENABLE 0
#  endif
#endif

/**
 * @def LINALG_SMALL_N_THRESH
 * Matrices with min(m,n) < this threshold use scalar paths to avoid SIMD overhead.
 */
#ifndef LINALG_SMALL_N_THRESH
#  define LINALG_SMALL_N_THRESH 16
#endif

/**
 * @def LINALG_BLOCK_KC
 * Inner-K blocking size (keeps panels in L1). Used in mul/qr where applicable.
 */
#ifndef LINALG_BLOCK_KC
#  define LINALG_BLOCK_KC 192
#endif

/**
 * @def LINALG_BLOCK_JC
 * Outer-N blocking size (column blocking). Used in qr / reflector application.
 */
#ifndef LINALG_BLOCK_JC
#  define LINALG_BLOCK_JC 128
#endif

/**
 * @brief Returns 1 if AVX2+FMA fast path is compiled in; 0 otherwise.
 */
static inline int linalg_has_avx2(void) {
#if LINALG_SIMD_ENABLE
  return 1;
#else
  return 0;
#endif
}

/* =====================================================================
 *                             Public API
 * ===================================================================== */

/**
 * @brief Transpose a row-major matrix A(row x col) into At(col x row).
 *
 * @details
 *  Uses an 8x8 AVX2 micro-kernel for the main body (cache-friendly blocked
 *  loads/unpacks/shuffles) and scalar tails for ragged edges. On non-AVX2 targets,
 *  a portable scalar version is used.
 *
 * @param[out] At   Destination buffer of size col*row (row-major).
 * @param[in]  A    Source buffer of size row*col (row-major).
 * @param[in]  row  Number of rows in A.
 * @param[in]  col  Number of columns in A.
 *
 * @warning At must not alias A.
 */
void tran(float *RESTRICT At, const float *RESTRICT A,
          uint16_t row, uint16_t col);

/**
 * @brief Multiply C = A * B for row-major matrices.
 *
 * @details
 *  Compatibility: A(row_a x column_a) * B(row_b x column_b) with column_a == row_b.
 *  Vectorized path:
 *   - AVX2/FMA dot-product loops with KC blocking and SSE/scalar tails.
 *   - Hoisted broadcasts / register reuse to reduce port-5 pressure.
 *  Returns 0 on success, -EINVAL if shapes are incompatible.
 *
 * @param[out] C         (row_a x column_b), row-major.
 * @param[in]  A         (row_a x column_a), row-major.
 * @param[in]  B         (row_b x column_b), row-major.
 * @param[in]  row_a
 * @param[in]  column_a
 * @param[in]  row_b
 * @param[in]  column_b
 *
 * @retval 0        OK
 * @retval -EINVAL  Shape mismatch (column_a != row_b).
 *
 * @warning C must not alias A or B.
 */
int mul(float *RESTRICT C,
        const float *RESTRICT A,
        const float *RESTRICT B,
        uint16_t row_a, uint16_t column_a,
        uint16_t row_b, uint16_t column_b);

/**
 * @brief LU factorization with partial pivoting: A = Pᵀ L U.
 *
 * @details
 *  Output LU stores L (unit diagonal) and U combined in the same matrix
 *  (row-major). Pivot vector P holds row indices. SIMD path accelerates
 *  the rank-1 trailing updates with AVX2 FMAs and scalar tails.
 *
 * @param[in]  A    Input (row x row); may equal LU (in-place allowed).
 * @param[out] LU   Output factors (row x row).
 * @param[out] P    Pivot indices (length row).
 * @param[in]  row  Dimension.
 *
 * @retval 0         OK
 * @retval -ENOTSUP  Singular within tolerance.
 * @retval -EINVAL   row == 0
 */
int lup(const float *RESTRICT A, float *RESTRICT LU,
        uint8_t *RESTRICT P, uint16_t row);

/**
 * @brief Invert a square matrix via LUP and triangular solves.
 *
 * @details
 *  Uses lup() followed by forward/back substitution on unit vectors.
 *  SIMD improves the triangular updates; results are row-major.
 *
 * @param[out] Ai_out  Inverse (row x row).
 * @param[in]  A       Input (row x row).
 * @param[in]  row     Dimension.
 *
 * @retval 0         OK
 * @retval -ENOTSUP  Singular matrix detected by LUP.
 * @retval -EINVAL   row == 0
 */
int inv(float *RESTRICT Ai_out, const float *RESTRICT A, uint16_t row);

/**
 * @brief QR decomposition (Householder): A = Q * R (or just R).
 *
 * @details
 *  Builds Householder reflectors in a vectorized way and applies them with
 *  blocked AVX2 rank-1 updates. If only_compute_R=true, Q is skipped to
 *  reduce work. Falls back to a small-matrix/scalar path for tiny problems.
 *
 * @param[in]  A               Input matrix (m x n), row-major.
 * @param[out] Q               (m x m) if only_compute_R=false; otherwise ignored.
 * @param[out] R               (m x n) upper-triangular on exit.
 * @param[in]  row_a           m.
 * @param[in]  column_a        n.
 * @param[in]  only_compute_R  If true, compute only R.
 *
 * @retval 0         OK
 * @retval -ENOTSUP  Numerical failure (e.g. near-singular reflectors).
 * @retval -EINVAL   m==0 or n==0
 *
 * @warning Q and R must not alias A.
 */
int qr(const float *RESTRICT A, float *RESTRICT Q, float *RESTRICT R,
       uint16_t row_a, uint16_t column_a, bool only_compute_R);

/**
 * @brief Rank-one Cholesky update/downdate of lower-triangular factor L.
 *
 * @details
 *  Given L (lower triangular of A), compute L̂ such that:
 *    - rank_one_update=true  : Â = A + x xᵀ
 *    - rank_one_update=false : Â = A − x xᵀ
 *  Implementation vectorizes the inner axpy-like updates with AVX2 and
 *  transposes as needed (row-major). Stable for typical SR-UKF sizes.
 *
 * @param[in,out] L              Lower-triangular factor (row x row), row-major.
 * @param[in]     x              Vector (row).
 * @param[in]     row            Dimension.
 * @param[in]     rank_one_update  true=update, false=downdate.
 */

int cholupdate(float *RESTRICT L,
               const float *RESTRICT x,
               uint16_t n,
               bool is_upper,
               bool rank_one_update);

/* =====================================================================
 *                       Notes on usage and aliasing
 * =====================================================================

  - Row-major layout everywhere. Index element (i,j) of an r x c matrix M as:
       M[i*c + j], with 0 <= i < r, 0 <= j < c.

  - Aliasing:
      * tran: At must not alias A.
      * mul : C must not alias A or B.
      * lup : A may equal LU (in-place copy handled); P must not alias LU.
      * inv : Ai_out must not alias A.
      * qr  : Q and R must not alias A; Q is only written when only_compute_R=false.
      * cholupdate: L updated in place; x is read-only.

  - SIMD fallbacks:
      All functions compile and run without AVX2/FMA; scalar paths are selected
      automatically at compile time.

  - Tuning:
      Adjust LINALG_BLOCK_KC/JC to target your cache hierarchy (192–256 typical KC).
      LINALG_SMALL_N_THRESH controls when to skip SIMD for tiny problems.

 */

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---------- Portable aligned allocation (32B by default) ---------- */
#ifndef LINALG_DEFAULT_ALIGNMENT
#  define LINALG_DEFAULT_ALIGNMENT 32
#endif

/* Always prefer these over calling aligned_alloc/_aligned_malloc directly. */
static inline void* linalg_aligned_alloc(size_t alignment, size_t size) {
#if defined(_MSC_VER)
    /* MSVC / also works in MinGW via malloc.h */
    return _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    #include <malloc.h>
    return __mingw_aligned_malloc(size, alignment);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    /* C11 aligned_alloc requires size to be a multiple of alignment */
    size = (size + (alignment - 1)) & ~(alignment - 1);
    return aligned_alloc(alignment, size);
#elif defined(_POSIX_VERSION) || defined(__unix__) || defined(__APPLE__)
    void* p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#else
    /* Fallback: over-allocate and round up */
    void* base = malloc(size + alignment + sizeof(void*));
    if (!base) return NULL;
    uintptr_t raw = (uintptr_t)base + sizeof(void*);
    uintptr_t aligned = (raw + (alignment - 1)) & ~(uintptr_t)(alignment - 1);
    ((void**)aligned)[-1] = base;
    return (void*)aligned;
#endif
}

static inline void linalg_aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    __mingw_aligned_free(p);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    free(p);
#elif defined(_POSIX_VERSION) || defined(__unix__) || defined(__APPLE__)
    free(p);
#else
    if (p) free(((void**)p)[-1]);
#endif
}

#endif /* CONTROL_LINALG_SIMD_H */
