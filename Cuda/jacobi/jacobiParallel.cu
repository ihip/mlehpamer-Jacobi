#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../../utils/Time/time.h"
#include "../../utils/Matrix_helper/matrix_generate_random.h"
#include "../../utils/Matrix_helper/printResults.h"

/**
 * Initialize matrix_Host pairs for chess tournament ordering
 */
void initializeMatrixPairs(int *array1, int *array2, int matrix_order)
{
    int j = 0;

    if (matrix_order % 2 != 0)
    {
        matrix_order++;
    }

    for (int i = 0; i < matrix_order / 2; i++)
    {
        array1[i] = i;
    }
    for (int i = matrix_order / 2; i < matrix_order; i++)
    {
        array2[j] = i;
        j++;
    }

    if (matrix_order % 2 != 0)
    {
        array2[matrix_order / 2 - 1] = -1;
    }
}

/*
 * Method that alocates space for matrix_Host
 */
void allocateMatrix(double **matrix_Host, int matrix_order)
{
    for (int i = 0; i < matrix_order; i++)
    {
        matrix_Host[i] = (double *)malloc(matrix_order * sizeof(double));
    }
}

/**
 * Compute Jacobi matrix_Host of transformations
 * @param inputMatrix - matrix_Host from which we will create jacobi matrix_Host
 * @param jacobiMatrix - matrix_Host of transformations - input matrix_Host is unitary matrix_Host
 * @param pair1 - array of matrix_Host pair non overlaping
 * @param pair2 - array of matrix_Host pair non overlaping
 */
__global__ void computeJacobiMatrix(double **inputMatrix, double **jacobiMatrix, int *pair1, int *pair2, int matrix_order, int *pair1_temp, int *pair2_temp)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = matrix_order;
    if (matrix_order % 2 != 0)
    {
        size++;
    }
    if (index < size / 2)
    {
        int i = pair1[index];
        int j = pair2[index];

        pair1_temp[index] = pair1[index];
        pair2_temp[index] = pair2[index];

        if (i > j)
        {
            int temp = i;
            i = j;
            j = temp;
        }

        if (i != -1 && inputMatrix[i][j] != 0)
        {
            double b = (inputMatrix[i][i] - inputMatrix[j][j]) / (2.0 * inputMatrix[i][j]);
            double t = ((b > 0) ? 1.0 : ((b < 0) ? -1.0 : 0.0)) / ((fabs(b)) + sqrt(pow(b, 2) + 1.0));
            double c = 1.0 / (sqrt(pow(t, 2) + 1.0));
            double s = c * t;

            jacobiMatrix[i][j] = s;
            jacobiMatrix[j][i] = -s;
            jacobiMatrix[j][j] = c;
            jacobiMatrix[i][i] = c;
        }
    }
}

/**
 * Kernel for shifting arrays for Jacobi parallel pairs
 * Also calculate matrix convergance
 */
__global__ void shiftJacobiPairs(int *pair1, int *pair2, int *temp1, int *temp2, double **matrix_Host, int matrix_order)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = matrix_order;

    if (matrix_order % 2 != 0)
    {
        size++;
    }

    if (index < size / 2)
    {
        if (index == 0)
        { // If it is first element
            pair2[index] = temp2[index + 1];
        }
        else if (index == 1)
        { // If it is second element
            pair1[index] = temp2[0];
            pair2[index] = temp2[index + 1];
        }
        else if (index == (size / 2 - 1))
        { // If it is last element
            pair1[size / 2 - 1] = temp1[size / 2 - 2];
            pair2[size / 2 - 1] = temp1[size / 2 - 1];
        }
        else
        { // Every other element
            pair1[index] = temp1[index - 1];
            pair2[index] = temp2[index + 1];
        }
    }
}

/**
 * Update rows after jacobi computation
 */
__global__ void updateRows(double **matrix_A, double **jacobi_matrix, int *pair_1, int *pair_2, int matrix_order)
{
    int index = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.x + blockDim.x * blockIdx.x;

    int size = matrix_order;
    if (matrix_order % 2 != 0)
    {
        size++;
    }
    if (index < size / 2)
    {
        int i = pair_1[index];
        int j = pair_2[index];

        if (i > j)
        {
            int temp = i;
            i = j;
            j = temp;
        }
        if (i < matrix_order && j < matrix_order && i != -1 && matrix_A[i][j] != 0)
        {

            double s = jacobi_matrix[j][i];
            double c = jacobi_matrix[j][j];

            if (k < matrix_order)
            {
                double mik = matrix_A[i][k];
                double mjk = matrix_A[j][k];

                matrix_A[i][k] = (c * mik) - (s * mjk);
                matrix_A[j][k] = (s * mik) + (c * mjk);
            }
        }
    }
}

/**
 * Update columns after jacobi computation
 */
__global__ void updateColumns(double **matrix_A, double **jacobi_matrix, int *pair_1, int *pair_2, int matrix_order)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.y + blockDim.y * blockIdx.y;

    int size = matrix_order;
    if (matrix_order % 2 != 0)
    {
        size++;
    }
    if (index < size / 2)
    {
        int i = pair_1[index];
        int j = pair_2[index];

        if (i > j)
        {
            int temp = i;
            i = j;
            j = temp;
        }
        if (i < matrix_order && j < matrix_order && i != -1 && matrix_A[i][j] != 0)
        {
            double s = jacobi_matrix[j][i];
            double c = jacobi_matrix[j][j];

            if (k < matrix_order)
            {
                double mki = matrix_A[k][i];
                double mkj = matrix_A[k][j];

                matrix_A[k][i] = (c * mki) - (s * mkj);
                matrix_A[k][j] = (s * mki) + (c * mkj);
            }
        }
    }
}

/*
 * Method that sums upper elements of matrix
 */
__global__ void sumOffset(double **matrix, double *precision, int matrix_order)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < matrix_order && row < col)
    {
        double mij = matrix[row][col];
        atomicAdd(precision, pow(mij, 2));
    }
}

/**
 * Main method of CUDA program
 */
extern "C" int runJacobiParallel()
{
    printf("Data initialization.\n");
    int _MATRIX_ORDER = getMatrixOrder();
    double **matrix_Host = symetric_matrix_double();
    double **matrixOut_Host;
    double **jacobiMatrix_Host = identityMatrixDouble();

    double **matrix_Device;
    double **jacobiMatrix_Device;
    double *precision_Device;

    double doubleSize = sizeof(double);
    double *precision_Host = (double *)malloc(doubleSize);

    /*Allocate double pointer host memory */
    matrixOut_Host = (double **)malloc(_MATRIX_ORDER * sizeof(double *));
    allocateMatrix(matrixOut_Host, _MATRIX_ORDER);

    //================ MANAGE PAIRS ==============================//
    int size;
    if (_MATRIX_ORDER % 2 != 0)
    {
        size = ((_MATRIX_ORDER + 1) / 2) * sizeof(int);
    }
    else
    {
        size = (_MATRIX_ORDER / 2) * sizeof(int);
    }
    int *pair1 = (int *)malloc(size);
    int *pair2 = (int *)malloc(size);

    cudaMalloc((void **)&precision_Device, sizeof(double));

    int *device_pair1, *device_temp1;
    int *device_pair2, *device_temp2;

    cudaMalloc((void **)&device_pair1, size);
    cudaMalloc((void **)&device_pair2, size);
    cudaMalloc((void **)&device_temp1, size);
    cudaMalloc((void **)&device_temp2, size);

    initializeMatrixPairs(pair1, pair2, _MATRIX_ORDER);

    cudaMemcpy(device_pair1, pair1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_pair2, pair2, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_temp1, pair1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_temp2, pair2, size, cudaMemcpyHostToDevice);

    //==============================================================================================================================//
    if (_MATRIX_ORDER < 12)
    {
        char msg[] = "Generated matrix:";
        print_matrix_double(matrix_Host, _MATRIX_ORDER, msg);
    }
    else
    {
        printf("Matrix generated.\n");
    }
    printf("\n%s\n", "Calculating...");
    printf("\nMatrix order: %d\n", _MATRIX_ORDER);

    //==============================================================================================================================//
    /* Allocate memory on device for every input/output and copy data HOST --> DEVICE */
    double **tmp = (double **)malloc(_MATRIX_ORDER * sizeof(tmp[0]));
    double **tmp_1 = (double **)malloc(_MATRIX_ORDER * sizeof(tmp_1[0]));

    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        cudaMalloc((void **)&tmp[i], _MATRIX_ORDER * sizeof(tmp[0][0]));
        cudaMalloc((void **)&tmp_1[i], _MATRIX_ORDER * sizeof(tmp_1[0][0]));
    }
    cudaMalloc((void **)&matrix_Device, _MATRIX_ORDER * sizeof(matrix_Device[0]));
    cudaMalloc((void **)&jacobiMatrix_Device, _MATRIX_ORDER * sizeof(jacobiMatrix_Device[0]));

    cudaMemcpy(matrix_Device, tmp, _MATRIX_ORDER * sizeof(matrix_Device[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(jacobiMatrix_Device, tmp_1, _MATRIX_ORDER * sizeof(jacobiMatrix_Device[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        cudaMemcpy(tmp[i], matrix_Host[i], _MATRIX_ORDER * sizeof(matrix_Device[0][0]), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_1[i], jacobiMatrix_Host[i], _MATRIX_ORDER * sizeof(jacobiMatrix_Device[0][0]), cudaMemcpyHostToDevice);
    }
    //==============================================================================================================================//
    /* Define dimension for kernels*/
    int threads = 32;
    int blocks = (_MATRIX_ORDER + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS_COLS(blocks / 2 + 1, blocks);
    dim3 BLOCKS_ROWS(blocks, blocks / 2 + 1);
    dim3 BLOCKS(blocks, blocks);
    //==============================================================================================================================//

    int iterations = 0;
    *precision_Host = 1.0;

    /* Start timer to measure time*/
    startTimer();
    do
    {
        //===================================== COMPUTE JACOBI MATRIX ===================================================//
        computeJacobiMatrix<<<blocks, threads>>>(matrix_Device, jacobiMatrix_Device, device_pair1, device_pair2, _MATRIX_ORDER, device_temp1, device_temp2);
        cudaDeviceSynchronize();

        //================================= TRANSFORM MATRIX ROWS =======================================================//
        updateRows<<<BLOCKS_ROWS, THREADS>>>(matrix_Device, jacobiMatrix_Device, device_pair1, device_pair2, _MATRIX_ORDER);
        cudaDeviceSynchronize();

        //================================= TRANSFORM MATRIX COLUMNS ====================================================//
        updateColumns<<<BLOCKS_COLS, THREADS>>>(matrix_Device, jacobiMatrix_Device, device_pair1, device_pair2, _MATRIX_ORDER);
        cudaDeviceSynchronize();

        //==================================== PERFORM SHIFT OF PARAMETERS =============================================//
        shiftJacobiPairs<<<blocks, threads>>>(device_pair1, device_pair2, device_temp1, device_temp2, matrix_Device, _MATRIX_ORDER);
        cudaDeviceSynchronize();

        //==================================== CALLCULATE OFFSET / CONVERGANCE ==========================================//
        if (iterations % (_MATRIX_ORDER - 1) == 0)
        {
            double zero = 0.0;
            cudaMemcpy(precision_Device, &zero, doubleSize, cudaMemcpyHostToDevice);
            sumOffset<<<BLOCKS, THREADS>>>(matrix_Device, precision_Device, _MATRIX_ORDER);
            cudaDeviceSynchronize();
            cudaMemcpy(precision_Host, precision_Device, doubleSize, cudaMemcpyDeviceToHost);
            printf("Offset(A) : %6.20f\n", sqrt(*precision_Host));
        }
        iterations++;
    } while (sqrt(*precision_Host) > 1e-20);
    /* End timer */
    endTimer();

    //============================= COPY RESULTS DEVICE ---> HOST ====================================================//
    double **tmp_2 = (double **)malloc(_MATRIX_ORDER * sizeof(tmp_2[0]));
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        cudaMalloc((void **)&tmp_2[i], _MATRIX_ORDER * sizeof(tmp_2[0][0]));
    }

    cudaMemcpy(tmp_2, matrix_Device, _MATRIX_ORDER * sizeof(matrix_Device[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        cudaMemcpy(matrixOut_Host[i], tmp_2[i], _MATRIX_ORDER * sizeof(matrix_Device[0][0]), cudaMemcpyDeviceToHost);
    }

    //=================================== PRINT OUT RESULT ===========================================================//

    printf("\n");
    char msg_1[] = "Eigenvalues from parallel Jacobi algorithm:";
    print_eigenvalues_double_from_matrix(matrixOut_Host, _MATRIX_ORDER, msg_1);
    printf("\n");
    printf("CPU time : %f\n", getTime());
    printf("Real time: %f\n", getwallClockTime());

    printf("%s%d", "NUMBER OF ITTERATIONS: ", iterations);
    printf("\n");

    //=================================== FREE ALL USED MEMORY =======================================================//

    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        cudaFree(tmp[i]);
        cudaFree(tmp_1[i]);
        cudaFree(tmp_2[i]);
    }

    cudaFree(tmp);
    cudaFree(tmp_1);
    cudaFree(tmp_2);

    free(*matrix_Host);
    free(*matrixOut_Host);
    free(*jacobiMatrix_Host);

    free(pair1);
    free(pair2);

    free(matrix_Host);
    free(matrixOut_Host);
    free(jacobiMatrix_Host);
    free(precision_Host);

    cudaFree(precision_Device);
    cudaFree(device_pair1);
    cudaFree(device_pair2);
    cudaFree(device_temp1);
    cudaFree(device_temp2);

    cudaFree(matrix_Device);
    cudaFree(jacobiMatrix_Device);

    return 1;
}
