#include <stdio.h>
#include <stdlib.h>

/**
 * Sort function
 */
int cmpfunc(const void *a, const void *b)
{
    if (*(double *)a > *(double *)b)
        return 1;
    else if (*(double *)a < *(double *)b)
        return -1;
    else
        return 0;
}

/**
 * Print eigenvalues of matrix (double)
 */
void print_eigenvalues_double(char *desc, int _MATRIX_ORDER, double *mat)
{
    printf("\n%s\n", desc);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        printf("%s%d%s %6.15f\n", "λ[", i + 1, "]≈", mat[i]);
    }
}

/**
 * Method that prints matrix_Host
 */
void print_matrix_double(double **matrix, int _MATRIX_ORDER, char *desc)
{
    printf("\n%s\n", desc);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        for (int j = 0; j < _MATRIX_ORDER; j++)
        {
            printf("%f%s", matrix[i][j], " ");
        }
        printf("\n");
    }
}

/**
 * Method that prints matrix_Host
 */
void print_matrix_integer(int **matrix, int _MATRIX_ORDER, char *desc)
{
    printf("\n%s\n", desc);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        for (int j = 0; j < _MATRIX_ORDER; j++)
        {
            printf("%d%s", matrix[i][j], " ");
        }
        printf("\n");
    }
}

/**
 * Method that prints eigenvalues extracted from matrix diagonal
 */
void print_eigenvalues_double_from_matrix(double **matrix, int _MATRIX_ORDER, char *desc)
{
    double array[_MATRIX_ORDER];
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        array[i] = matrix[i][i];
    }
    qsort(array, _MATRIX_ORDER, sizeof(double), cmpfunc);
    print_eigenvalues_double(desc, _MATRIX_ORDER, array);
    printf("\n");
}

/**
 *Print array of double type
 */
void print_array_double(double *array, int _MATRIX_ORDER, char *desc)
{
    printf("\n%s\n", desc);
    for (int i = 0; i < _MATRIX_ORDER; i++)
    {
        printf("%f%s", array[i], " ");
    }
    printf("\n");
}