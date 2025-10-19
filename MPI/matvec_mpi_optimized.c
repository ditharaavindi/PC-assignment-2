#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8192;
    int chunk = (N + size - 1) / size;

    float *A = malloc(chunk * N * sizeof(float));
    float *x = malloc(N * sizeof(float));
    float *y = malloc(chunk * sizeof(float));

    if (rank == 0)
        for (int i=0; i<N; i++) x[i] = 1.0;

    MPI_Bcast(x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i=0; i<chunk*N; i++) A[i] = 2.0;

    double start = MPI_Wtime();
    #pragma omp parallel for
    for (int i=0; i<chunk; i++) {
        float sum = 0.0;
        for (int j=0; j<N; j++)
            sum += A[i*N+j]*x[j];
        y[i] = sum;
    }
    double end = MPI_Wtime();

    if (rank == 0)
        printf("MPI Optimized Time = %.6f s\n", end - start);

    free(A); free(x); free(y);
    MPI_Finalize();
}
