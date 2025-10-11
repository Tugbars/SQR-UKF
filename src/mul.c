// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "linalg_simd.h"  // linalg_has_avx2(), LINALG_BLOCK_KC, RESTRICT

/* ---- pack B into 8-column panels, zero-padding last panel columns ----
 * Layout: for each panel j..j+7, store a K×8 block where row k is at panel_base + k*8
 */
static void pack_B_8col(float *RESTRICT Bp, const float *RESTRICT B,
                        size_t K, size_t N)
{
    size_t off = 0;
    for (size_t j = 0; j < N; j += 8) {
        const size_t jb = (j + 8 <= N) ? 8 : (N - j);
        for (size_t k = 0; k < K; ++k) {
            const float *src = B + k*N + j;
            float *dst = Bp + off + k*8;
            size_t t = 0;
            for (; t < jb; ++t) dst[t] = src[t];
            for (; t < 8;  ++t) dst[t] = 0.0f;
        }
        off += K * 8;
    }
}

/* ---- 4x8 micro-kernel over a K-slab (kk..kk+Kblk) of a single 8-col panel ----
 * c: base of output row 0; ldc = 8 (row stride inside the tmp buffer)
 * a0..a3: pointers to A rows (already advanced by kk)
 * panel: pointer to packed B panel at kk*8
 * Produces 4 rows × 8 cols into c with row stride ldc (no store to C here).
 */
static inline void gemm_4x8_panel_avx2fma(float *RESTRICT c, size_t ldc,
                                          const float *RESTRICT a0,
                                          const float *RESTRICT a1,
                                          const float *RESTRICT a2,
                                          const float *RESTRICT a3,
                                          const float *RESTRICT panel,
                                          size_t Kblk)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
#define STEP(T) do { \
        const __m256 b = _mm256_loadu_ps(panel + (k + (T)) * 8); \
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k + (T)), b, acc0); \
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k + (T)), b, acc1); \
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k + (T)), b, acc2); \
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k + (T)), b, acc3); \
    } while (0)
        STEP(0); STEP(1); STEP(2); STEP(3);
        STEP(4); STEP(5); STEP(6); STEP(7);
#undef STEP
    }

#if defined(__SSE4_1__)
    for (; k + 3 < Kblk; k += 4) {
        const __m256 b0 = _mm256_loadu_ps(panel + (k+0)*8);
        const __m256 b1 = _mm256_loadu_ps(panel + (k+1)*8);
        const __m256 b2 = _mm256_loadu_ps(panel + (k+2)*8);
        const __m256 b3 = _mm256_loadu_ps(panel + (k+3)*8);

        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k+0), b0, acc0);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k+1), b1, acc0);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k+2), b2, acc0);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k+3), b3, acc0);

        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k+0), b0, acc1);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k+1), b1, acc1);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k+2), b2, acc1);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k+3), b3, acc1);

        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k+0), b0, acc2);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k+1), b1, acc2);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k+2), b2, acc2);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k+3), b3, acc2);

        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k+0), b0, acc3);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k+1), b1, acc3);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k+2), b2, acc3);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k+3), b3, acc3);
    }
#endif

    for (; k < Kblk; ++k) {
        const __m256 b = _mm256_loadu_ps(panel + k*8);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1 + k), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2 + k), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3 + k), b, acc3);
    }

    _mm256_storeu_ps(c + 0*ldc, acc0);
    _mm256_storeu_ps(c + 1*ldc, acc1);
    _mm256_storeu_ps(c + 2*ldc, acc2);
    _mm256_storeu_ps(c + 3*ldc, acc3);
}

/* 1x8 kernel for leftover rows */
static inline void gemm_1x8_panel_avx2fma(float *RESTRICT c,
                                          const float *RESTRICT a0,
                                          const float *RESTRICT panel,
                                          size_t Kblk)
{
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 7 < Kblk; k += 8) {
#define STEP1(T) do { \
        const __m256 b = _mm256_loadu_ps(panel + (k+(T))*8); \
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k + (T)), b, acc); \
    } while (0)
        STEP1(0); STEP1(1); STEP1(2); STEP1(3);
        STEP1(4); STEP1(5); STEP1(6); STEP1(7);
#undef STEP1
    }
    for (; k < Kblk; ++k) {
        const __m256 b = _mm256_loadu_ps(panel + k*8);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(a0 + k), b, acc);
    }
    _mm256_storeu_ps(c, acc);
}

/* ---- top-level ---- */
int mul(float *RESTRICT C,
        const float *RESTRICT A,
        const float *RESTRICT B,
        uint16_t row_a, uint16_t column_a,
        uint16_t row_b, uint16_t column_b)
{
    if (column_a != row_b) return -EINVAL;

    const size_t M = row_a, K = column_a, N = column_b;

    /* scalar fallback or degenerate sizes */
    if (!linalg_has_avx2() || M == 0 || N == 0 || K == 0) {
        for (size_t i = 0; i < M; ++i) {
            const float *ai = A + i*K;
            for (size_t j = 0; j < N; ++j) {
                const float *bj = B + j;
                float s = 0.f;
                for (size_t k = 0; k < K; ++k) s += ai[k] * bj[k*N];
                C[i*N + j] = s;
            }
        }
        return 0;
    }

    /* Pack B into 8-col panels */
    const size_t n_panels = (N + 7) / 8;
    const size_t Bp_elems = n_panels * K * 8;
    float *Bp = (float*)linalg_aligned_alloc(32, Bp_elems * sizeof(float));
    if (!Bp) return -ENOMEM;
    pack_B_8col(Bp, B, K, N);

    /* Kc cache blocking */
    const size_t Kc = (size_t)LINALG_BLOCK_KC;  /* e.g., 192–256 */

    for (size_t kk = 0; kk < K; kk += Kc) {
        const size_t Kblk = (kk + Kc <= K) ? Kc : (K - kk);

        size_t panel_off = 0;
        for (size_t p = 0, j = 0; p < n_panels; ++p, j += 8, panel_off += K*8) {
            const size_t jb = (j + 8 <= N) ? 8 : (N - j);
            const float *panel = Bp + panel_off + kk*8;

            /* rows in blocks of 4 using 4x8 kernel */
            size_t i = 0;
            for (; i + 3 < M; i += 4) {
                const float *a0 = A + (i+0)*K + kk;
                const float *a1 = A + (i+1)*K + kk;
                const float *a2 = A + (i+2)*K + kk;
                const float *a3 = A + (i+3)*K + kk;

                /* temp buffer for 4 rows × 8 cols */
                alignas(32) float tmp[32];
                gemm_4x8_panel_avx2fma(tmp, 8, a0, a1, a2, a3, panel, Kblk);

                /* accumulate into C: add for kk>0, store for kk==0 */
                float *c0 = C + (i+0)*N + j;
                float *c1 = C + (i+1)*N + j;
                float *c2 = C + (i+2)*N + j;
                float *c3 = C + (i+3)*N + j;

                if (kk == 0) {
                    memcpy(c0, tmp +  0, jb * sizeof(float));
                    memcpy(c1, tmp +  8, jb * sizeof(float));
                    memcpy(c2, tmp + 16, jb * sizeof(float));
                    memcpy(c3, tmp + 24, jb * sizeof(float));
                } else {
                    for (size_t t = 0; t < jb; ++t) c0[t] += tmp[ 0 + t];
                    for (size_t t = 0; t < jb; ++t) c1[t] += tmp[ 8 + t];
                    for (size_t t = 0; t < jb; ++t) c2[t] += tmp[16 + t];
                    for (size_t t = 0; t < jb; ++t) c3[t] += tmp[24 + t];
                }
            }

            /* leftover rows (≤3): 1×8 kernel */
            for (; i < M; ++i) {
                const float *ai = A + i*K + kk;
                alignas(32) float tmp1[8];
                gemm_1x8_panel_avx2fma(tmp1, ai, panel, Kblk);
                float *ci = C + i*N + j;
                if (kk == 0) {
                    memcpy(ci, tmp1, jb * sizeof(float));
                } else {
                    for (size_t t = 0; t < jb; ++t) ci[t] += tmp1[t];
                }
            }
        }
    }

    linalg_aligned_free(Bp);
    return 0;
}
