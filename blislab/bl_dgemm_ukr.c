#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[(i) * (ld) + (j)]
#define b(i, j, ld) b[(i) * (ld) + (j)]
#define c(i, j, ld) c[(i) * (ld) + (j)]

//
// C-based micorkernel
//
void bl_dgemm_ukr(int k,
                  int m,
                  int n,
                  double *a,
                  double *b,
                  double *c,
                  unsigned long long ldc,
                  aux_t *data)
{
    int l, j, i;

    for (l = 0; l < k; ++l)
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
            {
                // ldc is used here because a[] and b[] are not packed by the
                // starter code
                // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate

                // 3H DEBUGGING: micro-kernel also need to change after packing !!!
                // because packed A is now column-major
                // leading dim also needs to change.
                c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
            }
        }
    }
}

// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

void bl_dgemm_444(int k,
                  int m,
                  int n,
                  double *restrict a,
                  double *restrict b,
                  double *restrict c,
                  unsigned long long ldc,
                  aux_t *data)

{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    
    c0x = svld1_f64(npred, c);
    c1x = svld1_f64(npred, c + ldc);
    c2x = svld1_f64(npred, c + 2 * ldc);
    c3x = svld1_f64(npred, c + 3 * ldc);
    
    for (int kk = 0; kk < k; kk++)
    {
        double aval;

        aval = *(a + 0 + kk*m);
        ax = svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), b + kk * n);
        c0x = svmla_f64_m(npred, c0x, bx, ax);


        aval = *(a + 1 + kk* m );
        ax = svdup_f64(aval);
        c1x = svmla_f64_m(npred, c1x, bx, ax);

        aval = *(a + 2 + kk* m );
        ax = svdup_f64(aval);
        c2x = svmla_f64_m(npred, c2x, bx, ax);

        aval = *(a + 3  + kk* m);
        ax = svdup_f64(aval);
        c3x = svmla_f64_m(npred, c3x, bx, ax);
    }
    
   
    svst1_f64(npred, c, c0x);
    svst1_f64(npred, c + ldc, c1x);
    svst1_f64(npred, c + 2 * ldc, c2x);
    svst1_f64(npred, c + 3 * ldc, c3x);
}


void bl_dgemm_844(int k,
                  int m,
                  int n,
                  double *restrict a,
                  double *restrict b,
                  double *restrict c,
                  unsigned long long ldc,
                  aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    
    // Load the existing values from the C matrix for 8 rows
    c0x = svld1_f64(npred, c + 0 * ldc);
    c1x = svld1_f64(npred, c + 1 * ldc);
    c2x = svld1_f64(npred, c + 2 * ldc);
    c3x = svld1_f64(npred, c + 3 * ldc);
    c4x = svld1_f64(npred, c + 4 * ldc);
    c5x = svld1_f64(npred, c + 5 * ldc);
    c6x = svld1_f64(npred, c + 6 * ldc);
    c7x = svld1_f64(npred, c + 7 * ldc);

    for (int kk = 0; kk < k; ++kk)
    {
        // Load a vector from B
        bx = svld1_f64(svptrue_b64(), b + kk * n);

        // Process each row of matrix A and update corresponding row of C
        ax = svdup_f64(*(a + 0 + kk * m));
        c0x = svmla_f64_m(npred, c0x, bx, ax);

        ax = svdup_f64(*(a + 1 + kk * m));
        c1x = svmla_f64_m(npred, c1x, bx, ax);

        ax = svdup_f64(*(a + 2 + kk * m));
        c2x = svmla_f64_m(npred, c2x, bx, ax);

        ax = svdup_f64(*(a + 3 + kk * m));
        c3x = svmla_f64_m(npred, c3x, bx, ax);

        ax = svdup_f64(*(a + 4 + kk * m));
        c4x = svmla_f64_m(npred, c4x, bx, ax);

        ax = svdup_f64(*(a + 5 + kk * m));
        c5x = svmla_f64_m(npred, c5x, bx, ax);

        ax = svdup_f64(*(a + 6 + kk * m));
        c6x = svmla_f64_m(npred, c6x, bx, ax);

        ax = svdup_f64(*(a + 7 + kk * m));
        c7x = svmla_f64_m(npred, c7x, bx, ax);
    }
    
    // Store the computed values back to the C matrix for 8 rows
    svst1_f64(npred, c + 0 * ldc, c0x);
    svst1_f64(npred, c + 1 * ldc, c1x);
    svst1_f64(npred, c + 2 * ldc, c2x);
    svst1_f64(npred, c + 3 * ldc, c3x);
    svst1_f64(npred, c + 4 * ldc, c4x);
    svst1_f64(npred, c + 5 * ldc, c5x);
    svst1_f64(npred, c + 6 * ldc, c6x);
    svst1_f64(npred, c + 7 * ldc, c7x);
}


void bl_dgemm_1644(int k,
                   int m,
                   int n,
                   double *restrict a,
                   double *restrict b,
                   double *restrict c,
                   unsigned long long ldc,
                   aux_t *data)
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    register svfloat64_t c8x, c9x, c10x, c11x, c12x, c13x, c14x, c15x;
    svbool_t npred = svwhilelt_b64_u64(0, n);

    c0x = svld1_f64(npred, c + 0 * ldc);
    c1x = svld1_f64(npred, c + 1 * ldc);
    c2x = svld1_f64(npred, c + 2 * ldc);
    c3x = svld1_f64(npred, c + 3 * ldc);
    c4x = svld1_f64(npred, c + 4 * ldc);
    c5x = svld1_f64(npred, c + 5 * ldc);
    c6x = svld1_f64(npred, c + 6 * ldc);
    c7x = svld1_f64(npred, c + 7 * ldc);
    c8x = svld1_f64(npred, c + 8 * ldc);
    c9x = svld1_f64(npred, c + 9 * ldc);
    c10x = svld1_f64(npred, c + 10 * ldc);
    c11x = svld1_f64(npred, c + 11 * ldc);
    c12x = svld1_f64(npred, c + 12 * ldc);
    c13x = svld1_f64(npred, c + 13 * ldc);
    c14x = svld1_f64(npred, c + 14 * ldc);
    c15x = svld1_f64(npred, c + 15 * ldc);

    for (int kk = 0; kk < k; ++kk)
    {
        bx = svld1_f64(svptrue_b64(), b + kk * n);

        ax = svdup_f64(*(a + 0 + kk * m));
        c0x = svmla_f64_m(npred, c0x, bx, ax);
        ax = svdup_f64(*(a + 1 + kk * m));
        c1x = svmla_f64_m(npred, c1x, bx, ax);
        ax = svdup_f64(*(a + 2 + kk * m));
        c2x = svmla_f64_m(npred, c2x, bx, ax);
        ax = svdup_f64(*(a + 3 + kk * m));
        c3x = svmla_f64_m(npred, c3x, bx, ax);
        ax = svdup_f64(*(a + 4 + kk * m));
        c4x = svmla_f64_m(npred, c4x, bx, ax);
        ax = svdup_f64(*(a + 5 + kk * m));
        c5x = svmla_f64_m(npred, c5x, bx, ax);
        ax = svdup_f64(*(a + 6 + kk * m));
        c6x = svmla_f64_m(npred, c6x, bx, ax);
        ax = svdup_f64(*(a + 7 + kk * m));
        c7x = svmla_f64_m(npred, c7x, bx, ax);
        ax = svdup_f64(*(a + 8 + kk * m));
        c8x = svmla_f64_m(npred, c8x, bx, ax);
        ax = svdup_f64(*(a + 9 + kk * m));
        c9x = svmla_f64_m(npred, c9x, bx, ax);
        ax = svdup_f64(*(a + 10 + kk * m));
        c10x = svmla_f64_m(npred, c10x, bx, ax);
        ax = svdup_f64(*(a + 11 + kk * m));
        c11x = svmla_f64_m(npred, c11x, bx, ax);
        ax = svdup_f64(*(a + 12 + kk * m));
        c12x = svmla_f64_m(npred, c12x, bx, ax);
        ax = svdup_f64(*(a + 13 + kk * m));
        c13x = svmla_f64_m(npred, c13x, bx, ax);
        ax = svdup_f64(*(a + 14 + kk * m));
        c14x = svmla_f64_m(npred, c14x, bx, ax);
        ax = svdup_f64(*(a + 15 + kk * m));
        c15x = svmla_f64_m(npred, c15x, bx, ax);
    }

    svst1_f64(npred, c + 0 * ldc, c0x);
    svst1_f64(npred, c + 1 * ldc, c1x);
    svst1_f64(npred, c + 2 * ldc, c2x);
    svst1_f64(npred, c + 3 * ldc, c3x);
    svst1_f64(npred, c + 4 * ldc, c4x);
    svst1_f64(npred, c + 5 * ldc, c5x);
    svst1_f64(npred, c + 6 * ldc, c6x);
    svst1_f64(npred, c + 7 * ldc, c7x);
    svst1_f64(npred, c + 8 * ldc, c8x);
    svst1_f64(npred, c + 9 * ldc, c9x);
    svst1_f64(npred, c + 10 * ldc, c10x);
    svst1_f64(npred, c + 11 * ldc, c11x);
    svst1_f64(npred, c + 12 * ldc, c12x);
    svst1_f64(npred, c + 13 * ldc, c13x);
    svst1_f64(npred, c + 14 * ldc, c14x);
    svst1_f64(npred, c + 15 * ldc, c15x);
}
