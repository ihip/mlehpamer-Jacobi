#include <stdio.h>
#include <stdlib.h>

#define N 512

__global__ void dot(int *a, int *b, int *c)
{
    __shared__ int temp[N]; // Djeljiva memroja za produkt množenja, inace je produkt privatan pa se ne može pristupiti
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

    __syncthreads();

    if (0 == threadIdx.x) // Prva dretva radi zbrajanje "pairwise products"
    {
        int sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += temp[i];
            *c = sum;
        }
    }
}

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

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, sizeof(int));

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(sizeof(int));

    random_ints(a, N);
    random_ints(b, N);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dot<<<1, N>>>(dev_a, dev_b, dev_c); // Jedan blok sa N dretvi

    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d\n", *c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
