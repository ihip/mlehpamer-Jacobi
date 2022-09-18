#ifndef _PRINT_RESULT_
#define _PRINT_RESULT_

#ifdef __cplusplus
// when included in C++ file, let compiler know these are C functions
extern "C"
{
#endif
    void print_matrix_double(double **matrix, int _MATRIX_ORDER, char *desc);
    void print_eigenvalues_double(char *desc, int _MATRIX_ORDER, double *mat);
    void print_eigenvalues_double_from_matrix(double **matrix, int _MATRIX_ORDER, char *desc);
    void print_array_double(double *array, int _MATRIX_ORDER, char *desc);
    void print_matrix_integer(int **matrix, int _MATRIX_ORDER, char *desc);

#ifdef __cplusplus
}
#endif

#endif /* _PRINT_RESULT_*/