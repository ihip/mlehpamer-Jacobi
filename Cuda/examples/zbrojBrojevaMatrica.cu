#include <stdio.h>
#include <stdlib.h>
#include "../Matrix_helper/matrix_generate_random.h"
#include "../Time/time.h"

#define MATRIX_ORDER 1024

__global__ void zbrojiBrojeveMatrice(int **ulaz, int *izlaz)
{

    __shared__ int temp[MATRIX_ORDER];

    int index = threadIdx.x + blockIdx.x * blockDim.x; // index of the element in the block

    temp[threadIdx.x] = ulaz[threadIdx.x][blockIdx.x]; // Na poziju dretve unutar bloka se zapiše vrijednost

    __syncthreads(); //Čeka se sinkronizacija dretva
    atomicAdd(izlaz, ulaz[threadIdx.x][blockIdx.x]);
}

/**
 *Metdod that prints matrix
 * */
void printMatrix(int **matrix)
{
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        for (int j = 0; j < MATRIX_ORDER; j++)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void printArray(int *ulaz)
{
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        printf("%d", ulaz[i]);
    }
    printf("\n");
}

/**
 * Alocate memory for 2D matrix
 *
 */
void alocateMemory(int **matrix)
{
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        matrix[i] = (int *)malloc(MATRIX_ORDER * sizeof(int));
    }
}

int main(void)
{
    int **host_ulaz, *host_izlaz;
    int **device_in, *device_out;

    printf("Generiram matricu\n");
    host_ulaz = symetric_matrix_integer(MATRIX_ORDER);

    host_izlaz = (int *)malloc(sizeof(int) * MATRIX_ORDER);

    printf("Alociram memoriju\n");
    cudaMalloc((void **)&device_out, sizeof(int));

    /* allocate device "matrix" */
    int **tmp = (int **)malloc(MATRIX_ORDER * sizeof(tmp[0]));
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        cudaMalloc((void **)&tmp[i], MATRIX_ORDER * sizeof(tmp[0][0]));
    }
    cudaMalloc((void **)&device_in, MATRIX_ORDER * sizeof(device_in[0]));

    /* copy "matrix" from host to device */
    cudaMemcpy(device_in, tmp, MATRIX_ORDER * sizeof(device_in[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        cudaMemcpy(tmp[i], host_ulaz[i], MATRIX_ORDER * sizeof(device_in[0][0]), cudaMemcpyHostToDevice);
    }
    free(tmp);

    printf("Pokrećem zbrajanje brojeva\n");
    startTimer();

    zbrojiBrojeveMatrice<<<32769, MATRIX_ORDER>>>(device_in, device_out);

    cudaMemcpy(host_izlaz, device_out, sizeof(int), cudaMemcpyDeviceToHost);

    endTimer();

    double time = getTime();

    printf("Time: %f\n", time);

    printf("%d\n", *host_izlaz);

    cudaFree(device_in);
    cudaFree(device_out);

    free(host_izlaz);

    /* free host "matrix" */
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        free(host_ulaz[i]);
    }
    free(host_ulaz);

    /* free device "matrix" */
    tmp = (int **)malloc(MATRIX_ORDER * sizeof(tmp[0]));
    cudaMemcpy(tmp, device_in, MATRIX_ORDER * sizeof(device_in[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        cudaFree(tmp[i]);
    }
    free(tmp);
    cudaFree(device_in);

    return 0;
}