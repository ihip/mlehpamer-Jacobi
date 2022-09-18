#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../../utils/Time/time.h"
#include "../../utils/Matrix_helper/matrix_generate_random.h"
#include "../../utils/Matrix_helper/printResults.h"

/**
 * Method that returns the sum of the matrix elements off diagonal
 */
double precision(double **matrix, int matrix_order)
{
    double suma = 0;
    for (int i = 0; i < matrix_order - 1; i++)
    {
        for (int j = i + 1; j < matrix_order; j++)
        {
            suma += pow(matrix[i][j], 2);
        }
    }
    return sqrt(suma);
}

/**
 * Run main method for serial computing of Jacobi method
 */
int runJacobiSerial()
{
    int MATRIX_ORDER = getMatrixOrder();
    double **matrix = symetric_matrix_double();

    if (MATRIX_ORDER < 12)
    {
        char msg[] = "Generated matrix:";
        print_matrix_double(matrix, MATRIX_ORDER, msg);
    }
    else
    {
        printf("Matrix generated.\n");
    }
    int z = 0;
    startTimer();
    while (precision(matrix, MATRIX_ORDER) > 1e-20)
    {
        for (int i = 0; i < MATRIX_ORDER - 1; i++)
        {
            for (int j = i + 1; j < MATRIX_ORDER; j++)
            {
                if (matrix[i][j] != 0)
                {
                    double b = (matrix[i][i] - matrix[j][j]) / (2.0 * matrix[i][j]);
                    double t = ((b > 0) ? 1.0 : ((b < 0) ? -1.0 : 0.0)) / ((fabs(b)) + sqrt(pow(b, 2) + 1.0));
                    double c = 1.0 / (sqrt(pow(t, 2) + 1.0));
                    double s = c * t;

                    double mii = matrix[i][i];
                    double mij = matrix[i][j];
                    double mjj = matrix[j][j];

                    matrix[j][j] = (pow(s, 2) * mii) - (2.0 * c * s * mij) + (pow(c, 2) * mjj);
                    matrix[i][i] = (pow(c, 2) * mii) + (2.0 * c * s * mij) + (pow(s, 2) * mjj);

                    matrix[i][j] = matrix[j][i] = 0;

                    for (int k = 0; k < MATRIX_ORDER; k++)
                    {
                        if (k != i && k != j)
                        {
                            double mik = matrix[i][k];
                            double mjk = matrix[j][k];
                            double mki = matrix[k][i];

                            matrix[i][k] = matrix[k][i] = c * mik + s * mjk;
                            matrix[j][k] = matrix[k][j] = c * mjk - s * mki;
                        }
                    }
                }
            }
            z++;
        }
    }
    endTimer();
    printf("\n");
    print_eigenvalues_double_from_matrix(matrix, MATRIX_ORDER, "Eigenvalues from serial Jacobi algorithm:");

    free(*matrix);
    free(matrix);

    printf("\n");
    printf("CPU time: %f\n", getTime());
    printf("Real time: %f\n", getwallClockTime());

    printf("%s%d", "Iterations: ", z);
}