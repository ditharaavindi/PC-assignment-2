#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4096;

    int iterations = 10;
    int chunk = N / size;

    double *A = malloc(chunk * N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    double *y = malloc(chunk * sizeof(double));

    if (rank == 0)
        for (int i = 0; i < N; i++)
            x[i] = 1.0;

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < chunk * N; i++)
        A[i] = 2.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double total = 0.0;

    for (int it = 0; it < iterations; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int i = 0; i < chunk; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < N; j++)
                sum += A[i * N + j] * x[j];
            y[i] = sum;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        total += (end - start);
    }

    if (rank == 0)
        printf("MPI Baseline Average Time = %.6f s (over %d runs)\n",
               total / iterations, iterations);

    free(A);
    free(x);
    free(y);
    MPI_Finalize();
}
