#include <stdio.h>

#define N 2048

__global__ void matMul(float *A, float *B, float *C, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if(row < n && col < n){
        for(int k=0;k<n;k++)
            sum += A[row*n+k]*B[k*n+col];
        C[row*n+col] = sum;
    }
}

int main(){
    float *A, *B, *C;
    size_t size = N*N*sizeof(float);
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for(int i=0;i<N*N;i++){ A[i]=1.0f; B[i]=2.0f; }

    dim3 block(16,16);
    dim3 grid((N+15)/16,(N+15)/16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMul<<<grid,block>>>(A,B,C,N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("CUDA Baseline Time = %.3f ms\n", ms);

    cudaFree(A); cudaFree(B); cudaFree(C);
}
