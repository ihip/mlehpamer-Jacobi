#include <stdio.h>

#define N 512 // Broj paralelnih niti

// global oznacava funkciju koja se pokreće na uređaju(device - GPU), a zove se preko domaćina (HOST - CPU)
__global__ void kernel(void)
{ // kernel je samo naziv, može staviti bilo koji drugi naziv
}

/**
Upravljanje memorijom GPU-a
— cudaMalloc() vs malloc() -- alociranje memorije
— cudaFree() vs free()
— cudaMemcpy() vs memcpy()
*/

__global__ void add(int *a, int *b, int *c)
{ // Koriste se pokazivaci jer se zeli memorija sa uređaja (tj. Device -GPU)
    *c = *a + *b;
}

__global__ void addWithBlock(int *a, int *b, int *c) // Koriste se paralelni blokovi
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void addWithParalelThreads(int *a, int *b, int *c) // Koriste se paralelne niti
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 100;
    }
}

/**
 * Method that prints out c values
 * /
 */
void print_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int main(void)
{
    int *a, *b, *c;             // host copies of a, b, c
    int *dev_a, *dev_b, *dev_c; // device copies of a, b, c

    int size = sizeof(int) * N; // Ako radimo sa N paralelnih niti, tj N integerea pa trebamo toliko memorije

    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    random_ints(a, N);
    random_ints(b, N);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    addWithParalelThreads<<<N, N>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    print_ints(c, N);

    free(a);
    free(b);
    free(c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}