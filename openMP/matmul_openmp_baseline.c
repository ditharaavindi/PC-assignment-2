#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 2048;
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    printf("Allocating matrices for %dx%d...\n", N, N);
    fflush(stdout);

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double start = omp_get_wtime();
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

    printf("Initialization complete.\n");
    fflush(stdout);

    printf("Matrix Size = %d × %d\n", N, N);
    printf("OpenMP Baseline Time = %.6f s\n", end - start);

    free(A);
    free(B);
    free(C);
    return 0;
}
