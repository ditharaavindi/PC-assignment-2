#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 2048;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double start = omp_get_wtime();

#pragma omp parallel for schedule(static, 32) collapse(2)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }

    double end = omp_get_wtime();
    printf("Matrix Size = %d × %d\n", N, N);
    printf("OpenMP Optimized Time = %.6f s\n", end - start);

    free(A);
    free(B);
    free(C);
    return 0;
}
