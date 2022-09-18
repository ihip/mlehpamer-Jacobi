#include <stdio.h>
#include <stdlib.h>

#define N (2048 * 2048)
#define THREAD_PER_BLOCK 512

/**
 * Add two vectors together
 */
__global__ void add(int *a, int *b, int *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

/**
 * Generate random numbers
 */
void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 100;
    }
}

int main(void)
{
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    int size = N * sizeof(int);

    // Alociranje(rezerviranje) memorije na uređaju (GPU)
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Alociranje(rezerviranje) memorije na CPU-u
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    random_ints(a, N);
    random_ints(b, N);

    // Kopiranje podataka iz CPU-a u GPU
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Poziv kernel-a
    add<<<N / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(dev_a, dev_b, dev_c);

    // Kopiranje podataka iz GPU-a u CPU
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Ispis rezultata
    for (int i = 0; i < N; i++)
    {
        printf("%d\n", c[i]);
    }

    // Oslobadjanje memorije
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    return 0;
}