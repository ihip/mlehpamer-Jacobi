#include <stdio.h>
#include <stdlib.h>

#define N (2048 * 2048)
#define Threads_Per_Block 512

__global__ void dotMultipleBlocks(int *a, int *b, int *c)
{
    __shared__ int temp[Threads_Per_Block];

    int index = threadIdx.x + blockIdx.x * blockDim.x; // index of the element in the block

    temp[threadIdx.x] = a[index] * b[index]; // Na poziju dretve unutar bloka se zapiše vrijednost

    __syncthreads(); //Čeka se sinkronizacija dretva

    if (0 == threadIdx.x)
    {
        int sum = 0;
        for (int i = 0; i < Threads_Per_Block; i++)
        {
            sum += temp[i];
            atomicAdd(c, sum); // Potrebna je napisana operacija jer korisitmo više blokova pa možda više blokova pokušava pristupiti istoj memoriji i zapisati vrijednost
            // Radi se zapravo zapisivanje u memoriju varijable c, bolje nego sum += c tj. ovo je ispravna zamjena
        }
    }
}

void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 100 +15;
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

    dotMultipleBlocks<<<(N / Threads_Per_Block), Threads_Per_Block>>>(dev_a, dev_b, dev_c);

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
