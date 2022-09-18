#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../../utils/Time/time.h"
#include "../../utils/Matrix_helper/matrix_generate_random.h"
#include "../../utils/Matrix_helper/printResults.h"
#include <unistd.h>
/**
 * Kernel for multiplying 2D array / double pointer array
 * Optimized method for mulitplying matrix_Host
 */
__global__ void multiplyMatrix_(double **A_d, double **B_d, double **C_d, int matrix_order)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < matrix_order && row < matrix_order)
    {
        double tmp = 0.0;
        for (int z = 0; z < matrix_order; z++)
        {
            tmp += A_d[col][z] * B_d[z][row];
        }
        C_d[row][col] = tmp;
    }
}

__global__ void QRFactorization(double **symetricMatrix, double **identityMatrix, double **device_out, double *deviceInColumn, int column, int matrix_order)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = matrix_order - 1;
    if (i < (matrix_order))
    {

        double u_ = sqrt(pow(deviceInColumn[j - 1], 2) + pow(deviceInColumn[j], 2));
        double c = deviceInColumn[j - 1] / u_;
        double s = deviceInColumn[j] / u_;

        while (j > column)
        {
            if (u_ != 0)
            {
                double u = symetricMatrix[j - 1][i];
                double v = symetricMatrix[j][i];
                double alfa = identityMatrix[i][j - 1];
                double beta = identityMatrix[i][j];

                device_out[j - 1][i] = (c * u) + (s * v);
                device_out[j][i] = -(s * u) + (c * v);

                symetricMatrix[j - 1][i] = (c * u) + (s * v);
                symetricMatrix[j][i] = -(s * u) + (c * v);

                identityMatrix[i][j] = -(s * alfa) + (c * beta);
                identityMatrix[i][j - 1] = (c * alfa) + (s * beta);

                j--;
                double a = u_;
                u_ = sqrt(pow(deviceInColumn[j - 1], 2) + pow(a, 2));
                c = deviceInColumn[j - 1] / u_;
                s = a / u_;
            }
            else
            {
                j--;
                u_ = sqrt(pow(deviceInColumn[j - 1], 2) + pow(deviceInColumn[j], 2));
                c = deviceInColumn[j - 1] / u_;
                s = deviceInColumn[j] / u_;

                device_out[j - 1][i] = 0;
                device_out[j][i] = 0;
            }
        }
    }
}

__global__ void getColumn(double **A, double *columnOutput, int c, int matrix_order)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < matrix_order)
    {
        columnOutput[index] = A[index][c];
    }
}

/*
 * Method that alocates space for matrix_Host
 */
void allocateMatrix_(double **matrix_Host, int matrix_order)
{
    for (int i = 0; i < matrix_order; i++)
    {
        matrix_Host[i] = (double *)malloc(matrix_order * sizeof(double));
    }
}

__global__ void resetMatrix(int matrix_order, double **unitary_matrix)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < matrix_order)
    {
        for (int i = 0; i < matrix_order; i++)
        {
            if (index == i)
            {
                unitary_matrix[i][index] = 1;
            }
            else
            {
                unitary_matrix[i][index] = 0;
            }
        }
    }
}

__global__ void prec(double **matrix, double *precision, int matrix_order)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < matrix_order && row < col && row != col)
    {
        double mij = matrix[row][col];
        atomicAdd(precision, pow(mij, 2));
    }
}

extern "C" int runQRParallel()
{
    int _MATRIX_ORDER_ = getMatrixOrder();
    double **host_ulaz, **host_izlaz;
    double **device_in, **device_out;
    double **host_identity_matrix, **device_identity_matrix_transpose;
    double **device_identity_matrix;

    double *deviceInColumn;
    int *columnIdentifier;
    cudaMalloc((void **)&deviceInColumn, sizeof(double));
    cudaMalloc((void **)&columnIdentifier, sizeof(int));

    printf("Generiram matricu\n");
    host_ulaz = symetric_matrix_double();
    host_identity_matrix = identityMatrixDouble();

    if (_MATRIX_ORDER_ < 12)
    {
        char msg[] = "Generated matrix:";
        print_matrix_double(host_ulaz, _MATRIX_ORDER_, msg);
    }
    else
    {
        printf("Matrix generated.\n");
    }
    printf("\n%s\n", "Calculating...");
    printf("\nMatrix order: %d\n", _MATRIX_ORDER_);

    /*Allocate double pointer host memory */
    host_izlaz = (double **)malloc(_MATRIX_ORDER_ * sizeof(double *));
    allocateMatrix_(host_izlaz, _MATRIX_ORDER_);

    //==============================================================================================================================//
    /* Allocate memory on device for every input/output and copy data HOST --> DEVICE */
    double **tmp = (double **)malloc(_MATRIX_ORDER_ * sizeof(tmp[0]));
    double **tmp_1 = (double **)malloc(_MATRIX_ORDER_ * sizeof(tmp_1[0]));
    double **tmp_2 = (double **)malloc(_MATRIX_ORDER_ * sizeof(tmp_1[0]));
    double **tmp_3 = (double **)malloc(_MATRIX_ORDER_ * sizeof(tmp_3[0]));
    for (int i = 0; i < _MATRIX_ORDER_; i++)
    {
        cudaMalloc((void **)&tmp[i], _MATRIX_ORDER_ * sizeof(tmp[0][0]));
        cudaMalloc((void **)&tmp_1[i], _MATRIX_ORDER_ * sizeof(tmp_1[0][0]));
        cudaMalloc((void **)&tmp_2[i], _MATRIX_ORDER_ * sizeof(tmp_1[0][0]));
        cudaMalloc((void **)&tmp_3[i], _MATRIX_ORDER_ * sizeof(tmp_3[0][0]));
    }
    cudaMalloc((void **)&device_in, _MATRIX_ORDER_ * sizeof(device_in[0]));
    cudaMalloc((void **)&device_identity_matrix, _MATRIX_ORDER_ * sizeof(device_identity_matrix[0]));
    cudaMalloc((void **)&device_identity_matrix_transpose, _MATRIX_ORDER_ * sizeof(device_identity_matrix_transpose[0]));
    cudaMalloc((void **)&device_out, _MATRIX_ORDER_ * sizeof(device_out[0]));

    cudaMemcpy(device_in, tmp, _MATRIX_ORDER_ * sizeof(device_in[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(device_identity_matrix, tmp_1, _MATRIX_ORDER_ * sizeof(device_identity_matrix[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(device_identity_matrix_transpose, tmp_2, _MATRIX_ORDER_ * sizeof(device_identity_matrix_transpose[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(device_out, tmp_3, _MATRIX_ORDER_ * sizeof(device_out[0]), cudaMemcpyHostToDevice);
    for (int i = 0; i < _MATRIX_ORDER_; i++)
    {
        cudaMemcpy(tmp[i], host_ulaz[i], _MATRIX_ORDER_ * sizeof(device_in[0][0]), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_1[i], host_identity_matrix[i], _MATRIX_ORDER_ * sizeof(device_identity_matrix[0][0]), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_2[i], host_identity_matrix[i], _MATRIX_ORDER_ * sizeof(device_identity_matrix_transpose[0][0]), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_3[i], host_ulaz[i], _MATRIX_ORDER_ * sizeof(device_out[0][0]), cudaMemcpyHostToDevice);
    }
    //==============================================================================================================================//

    double *precision_Device;
    double doubleSize = sizeof(double);
    cudaMalloc((void **)&precision_Device, sizeof(double));
    double *precision_Host = (double *)malloc(doubleSize);

    startTimer();
    int j = 0;

    int threads = 32;
    int blocks = (_MATRIX_ORDER_ + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    do
    {
        resetMatrix<<<blocks, threads>>>(_MATRIX_ORDER_, device_identity_matrix);
        for (int i = 0; i < _MATRIX_ORDER_ - 1; ++i)
        {
            getColumn<<<blocks, threads>>>(device_in, deviceInColumn, i, _MATRIX_ORDER_);

            QRFactorization<<<blocks, threads>>>(device_in, device_identity_matrix, device_out, deviceInColumn, i, _MATRIX_ORDER_);
        }

        multiplyMatrix_<<<BLOCKS, THREADS>>>(device_out, device_identity_matrix, device_in, _MATRIX_ORDER_);

        if (j % (_MATRIX_ORDER_ * 10 - 1) == 0)
        {
            double zero = 0;
            cudaMemcpy(precision_Device, &zero, doubleSize, cudaMemcpyHostToDevice);
            prec<<<BLOCKS, THREADS>>>(device_in, precision_Device, _MATRIX_ORDER_);
            cudaMemcpy(precision_Host, precision_Device, doubleSize, cudaMemcpyDeviceToHost);
            printf("%s%2.30f%s", "Precision: ", sqrt(*precision_Host), "\n");
        }
        j++;

    } while (sqrt(*precision_Host) > 1e-1);
    /* COPY DATA FROM DEVICE TO HOST */

    cudaDeviceSynchronize();
    double **tmp_4 = (double **)malloc(_MATRIX_ORDER_ * sizeof(tmp_4[0]));
    for (int i = 0; i < _MATRIX_ORDER_; i++)
    {
        cudaMalloc((void **)&tmp_4[i], _MATRIX_ORDER_ * sizeof(tmp_4[0][0]));
    }
    cudaDeviceSynchronize();
    /* copy "matrix" from host to device */
    cudaMemcpy(tmp_4, device_in, _MATRIX_ORDER_ * sizeof(device_in[0]), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < _MATRIX_ORDER_; i++)
    {
        cudaMemcpy(host_izlaz[i], tmp_4[i], _MATRIX_ORDER_ * sizeof(device_in[0][0]), cudaMemcpyDeviceToHost);
    }

    //=================================== PRINT OUT RESULT ===========================================================//

    printf("\n");
    char msg_1[] = "Eigenvalues from parallel QR algorithm:";
    print_eigenvalues_double_from_matrix(host_izlaz, _MATRIX_ORDER_, msg_1);
    printf("\n");
    endTimer();
    printf("Total time: %f\n", getTime());
    printf("Iterations: %d\n", j);

    for (int i = 0; i < _MATRIX_ORDER_; i++)
    {
        cudaFree(tmp[i]);
        cudaFree(tmp_1[i]);
        cudaFree(tmp_3[i]);
        cudaFree(tmp_4[i]);
    }

    free(tmp);
    free(tmp_1);
    free(tmp_3);
    free(tmp_4);

    free(*host_identity_matrix);
    free(*host_izlaz);
    free(*host_ulaz);

    free(host_identity_matrix);
    free(host_izlaz);
    free(host_ulaz);

    cudaFree(device_identity_matrix);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(deviceInColumn);

    return 1;
}