// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linalg_simd.h"

int main(void)
{
    printf("Has AVX2: %d\n", linalg_has_avx2());

    const uint16_t n = 4;
    float A[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 6, 4, 8,
        3, 1, 1, 2
    };

    float At[16], Ai[16], C[16];
    uint8_t P[4];

    // --- transpose test ---
    tran(At, A, n, n);
    printf("\nA? =\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) printf("%6.2f ", At[i*n + j]);
        putchar('\n');
    }

    // --- multiply test (At * A) ---
    mul(C, At, A, n, n, n, n);
    printf("\nA?A =\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) printf("%8.2f ", C[i*n + j]);
        putchar('\n');
    }

    // --- LUP + inv test ---
    if (lup(A, Ai, P, n) == 0 && inv(Ai, A, n) == 0) {
        printf("\nA?¹ =\n");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) printf("%8.3f ", Ai[i*n + j]);
            putchar('\n');
        }
    } else {
        printf("LUP/inv failed.\n");
    }

    // --- QR test ---
    float Q[16], R[16];
    if (qr(A, Q, R, n, n, false) == 0) {
        printf("\nQ?Q (should be I) =\n");
        float QQT[16];
        tran(At, Q, n, n);
        mul(QQT, At, Q, n, n, n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) printf("%8.3f ", QQT[i*n + j]);
            putchar('\n');
        }
    }

    return 0;
}