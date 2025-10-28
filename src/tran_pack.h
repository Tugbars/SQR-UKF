// SPDX-License-Identifier: MIT
/**
 * @file tran_pack.h
 * @brief Tiled transpose and fused pack-transpose helpers (single-threaded, AVX2).
 *
 * @details
 * Provides:
 *  - **tran_tiled()**: Cache-blocked transpose with AVX2 8×8 and SSE 8×4 micro-kernels,
 *    plus scalar cleanup.  Optional non-temporal stores for large one-shot writes.
 *  - **pack_T_8xK()**:  Packs an 8×K panel of Aᵀ directly from row-major A, ready for
 *    an 8×16 GEMM kernel’s “A side”.
 *  - **pack_T_Kx16()**: Packs a K×16 panel of Bᵀ directly from row-major B, ready for
 *    an 8×16 GEMM kernel’s “B side”.
 *
 * Intended use:
 *  - **Standalone transpose**: for utilities or testing.
 *  - **Fused pack-transpose**: called inside GEMM/TRSM/SYRK packers so no separate
 *    At buffer is ever written to memory.
 *
 * Performance notes:
 *  - The transpose is memory-bound; its benefit is mostly cache locality
 *    and optional NT-store streaming when writing cold destinations.
 *  - The packers are compute-free and should be inlined by the compiler.
 *
 * Build-time knobs:
 *  - `TRAN_TILE` (default 32): macro-tile size for blocking.
 *  - `TRAN_USE_NT_STORES` (default 1): enable `_mm256_stream_ps` / `_mm_stream_ps`.
 *
 * Safe for single-threaded use.  For parallelization, split the outer tile loops.
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Cache-blocked transpose using AVX2 8×8 + SSE 8×4 kernels.
     *
     * @param[out] At  Row-major output matrix (C×R).  Must not alias A.
     * @param[in]  A   Row-major input matrix (R×C).
     * @param[in]  R   Number of rows in A.
     * @param[in]  C   Number of columns in A.
     *
     * @note If `TRAN_USE_NT_STORES != 0`, non-temporal stores are used;
     *       At should not be reused immediately after the call.
     * @warning In-place transpose (`At == A`) is not supported.
     */
    void tran_tiled(float *RESTRICT At, const float *RESTRICT A,
                    uint16_t R, uint16_t C);

    /**
     * @brief Pack a transposed 8×K micro-panel from row-major A into contiguous buffer.
     *
     * @param[in]  A     Row-major source matrix (M×Ktot).
     * @param[in]  M     Rows of A.
     * @param[in]  Ktot  Columns of A.
     * @param[in]  i     Column start index in A (→ row block in Aᵀ).
     * @param[in]  k0    Row start index in A (→ column start in Aᵀ).
     * @param[in]  K     Depth (number of rows from A) to pack.
     * @param[out] Ap    Output buffer of size 8×K (row-major).
     *
     * Layout: `Ap[r*K + t] = A[(k0+t), (i+r)]`
     */
    static inline void pack_T_8xK(const float *RESTRICT A, uint16_t M, uint16_t Ktot,
                                  uint16_t i, uint16_t k0, uint16_t K,
                                  float *RESTRICT Ap)
    {
        (void)M;
        for (uint16_t r = 0; r < 8; ++r)
        {
            const float *col = A + (size_t)(i + r);
            float *dst = Ap + (size_t)r * K;
            for (uint16_t t = 0; t < K; ++t)
                dst[t] = col[(size_t)(k0 + t) * Ktot];
        }
    }

    /**
     * @brief Pack a transposed K×16 micro-panel from row-major B into contiguous buffer.
     *
     * @param[in]  B     Row-major source matrix (Ktot×N).
     * @param[in]  Ktot  Rows of B.
     * @param[in]  N     Columns of B.
     * @param[in]  k0    Row start index in B (→ column start in Bᵀ).
     * @param[in]  j     Column start index in B (→ row block in Bᵀ).
     * @param[in]  K     Depth (number of rows from B) to pack.
     * @param[out] Bp    Output buffer of size K×16.
     *
     * Layout: `Bp[t*16 + c] = B[(k0+t), (j+c)]`
     */
    static inline void pack_T_Kx16(const float *RESTRICT B, uint16_t Ktot, uint16_t N,
                                   uint16_t k0, uint16_t j, uint16_t K,
                                   float *RESTRICT Bp)
    {
        for (uint16_t t = 0; t < K; ++t)
        {
            const float *row = B + (size_t)(k0 + t) * N + j;
            memcpy(Bp + (size_t)t * 16, row, 16 * sizeof(float));
        }
    }

#ifdef __cplusplus
}
#endif
