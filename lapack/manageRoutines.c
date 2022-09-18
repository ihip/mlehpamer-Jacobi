#include <stdio.h>
#include <lapacke.h>
#include "../utils/Matrix_helper/matrix_generate_random.h"
#include "../utils/Matrix_helper/printResults.h"
#include "../utils/Time/time.h"

//================================== LAPACK ROUTINES ============================================================//

/**
 * Call lapack routine dsyev
 */
void dsyev()
{
    int MATRIX_ORDER = getMatrixOrder();
    double **a = symetric_matrix_double();
    lapack_int info, n, lda;

    char jobz = 'N';        // -- Eigenvalues only
    char uplo = 'U';        // -- Upper triangle is stored
    n = MATRIX_ORDER;       // -- Order of the matrix
    lda = MATRIX_ORDER;     // -- The leading dimension of the array
    double w[MATRIX_ORDER]; // -- Used to store eigenvalues results

    if (MATRIX_ORDER < 12)
    {
        char msg[] = "Generated matrix:";
        print_matrix_double(a, MATRIX_ORDER, msg);
    }
    else
    {
        printf("Matrix generated.\n");
    }
    startTimer();
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, n, *a, lda, w);
    endTimer();

    /* Print Solution */
    print_eigenvalues_double("Eigenvalues for LAPACK routine DSYEV: ", MATRIX_ORDER, w);
    printf("\n");

    free(*a);
    free(a);

    printf("\n");
    printf("CPU time: %f\n", getTime());
    printf("Real time: %f\n", getwallClockTime());
}