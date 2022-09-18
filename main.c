#include <stdlib.h>
#include <stdio.h>
#include "./Cuda/jacobi/jacobiSerial.h"
#include "./Cuda/jacobi/jacobiParallel.h"
#include "./Cuda/QR/QRSymetricParallel.h"
#include "./utils/Matrix_helper/matrix_generate_random.h"

/**
 * Handle user input choices.
 */
int main(void)
{
    int matrix_order;
    int choice = 0;

    printf("===============================================================================\n");
    printf("============ Welcome to matrix eigenvalues computation program ================\n");
    printf("===============================================================================\n");

    do
    {
        printf("Please insert matrix order: ");
        scanf("%d", &matrix_order);
        if (matrix_order < 1)
        {
            printf("Matrix order must be greater than 0.\n");
        }
    } while (matrix_order < 1);

    setMatrixOrder(matrix_order);
    int MATRIX_ORDER = getMatrixOrder();
    do
    {
        printf("\n\n\n===============================================================================\n");
        printf("================================ MENU =========================================\n");
        printf("===============================================================================\n");
        printf("%s%d%s", "Current matrix order: ", MATRIX_ORDER, "\n");
        printf("===============================================================================\n");
        printf("Please choose one of the following options:\n");
        printf("0. Change order of the matrix.\n");
        printf("1. Compute matrix eigenvalues by serial Jacobi algorithm.\n");
        printf("2. Compute matrix eigenvalues by parallel Jacobi algorithm.\n");
        printf("3. Compute matrix eigenvalues by LAPACK routine DSYEV.\n");
        printf("9. Exit.\n");

        printf("Your choice: ");

        scanf("%d", &choice);
        switch (choice)
        {
        case 0:
        {
            printf("Please insert new matrix order: ");
            scanf("%d", &matrix_order);
            setMatrixOrder(matrix_order);
            MATRIX_ORDER = getMatrixOrder();
            break;
        }
        case 1:
        {
            runJacobiSerial();
            break;
        }
        case 2:
        {
            runJacobiParallel();
            break;
        }
        case 3:
        {
            dsyev();
            break;
        }
        }

    } while (choice != 9);
}