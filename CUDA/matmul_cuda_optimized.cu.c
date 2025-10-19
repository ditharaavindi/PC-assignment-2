#include <stdio.h>
#define N 2048
#define TILE 32

__global__ void matMulTiled(float *A,float *B,float *C,int n){
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float sum = 0.0f;

    for(int m=0;m<n/TILE;m++){
        sA[threadIdx.y][threadIdx.x] = A[row*n + m*TILE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(m*TILE + threadIdx.y)*n + col];
        __syncthreads();

        for(int k=0;k<TILE;k++)
            sum += sA[threadIdx.y][k]*sB[k][threadIdx.x];
        __syncthreads();
    }
    C[row*n + col] = sum;
}

int main(){
    float *A, *B, *C;
    size_t size = N*N*sizeof(float);
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for(int i=0;i<N*N;i++){ A[i]=1.0f; B[i]=2.0f; }

    dim3 block(TILE, TILE);
    dim3 grid(N/TILE, N/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulTiled<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("CUDA Optimized Time = %.3f ms\n", ms);

    cudaFree(A); cudaFree(B); cudaFree(C);
}
