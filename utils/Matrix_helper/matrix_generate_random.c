#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define SEED 112333121

int MATRIX_ORDER = 0;

//============================================= START OF Initializing matrix order ===========================================================//

/**
 * Set order of the matrix
 */
void setMatrixOrder(int matrix_order)
{
    MATRIX_ORDER = matrix_order;
}

/**
 * Get order of the matrix
 */
int getMatrixOrder()
{
    return MATRIX_ORDER;
}

//=============================================== END OF Initializing matrix order ===========================================================//

/**
 * Generate symetric matrix with random double numbers
 */
double **symetric_matrix_double()
{
    double *values = calloc(MATRIX_ORDER * MATRIX_ORDER, sizeof(double));
    double **rows = malloc(MATRIX_ORDER * sizeof(double *));

    srand48(SEED);

    for (int i = 0; i < MATRIX_ORDER; ++i)
    {
        rows[i] = values + i * MATRIX_ORDER;
    }

    for (int i = 0; i < MATRIX_ORDER; ++i)
    {
        for (int j = i; j < MATRIX_ORDER; ++j)
        {
            rows[i][j] = drand48();
            rows[j][i] = rows[i][j];
        }
    }
    return rows;
}

//========================================= END OF Generating symetric matrix ========================================================//

//========================================= START OF Generating identity matrix =====================================================//

double **identityMatrixDouble()
{
    double **matrix = malloc(MATRIX_ORDER * sizeof(double *));
    for (int i = 0; i < MATRIX_ORDER; i++)
    {
        matrix[i] = (double *)malloc(MATRIX_ORDER * sizeof(double));
    }

    for (int i = 0; i < MATRIX_ORDER; ++i)
    {
        for (int j = i; j < MATRIX_ORDER; ++j)
        {

            if (i == j)
            {
                matrix[i][j] = 1;
            }
            else
            {
                matrix[i][j] = 0;
                matrix[j][i] = 0;
            }
        }
    }

    return matrix;
}
//========================================= END OF Generating identity matrix =======================================================//
