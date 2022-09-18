#ifndef _MATRIX_GENERATE_RANDOM_
#define _MATRIX_GENERATE_RANDOM_

#ifdef __cplusplus
// when included in C++ file, let compiler know these are C functions
extern "C"
{
#endif
    void setMatrixOrder(int matrix_order);
    int getMatrixOrder();

    double **symetric_matrix_double();
    double **identityMatrixDouble();

#ifdef __cplusplus
}
#endif

#endif /* _MATRIX_GENERATE_RANDOM_*/