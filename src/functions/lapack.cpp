#include "Hatrix/classes/Matrix.h"

#include "mkl_cblas.h"
#include "mkl_lapacke.h"

#include <algorithm>
#include <vector>


namespace Hatrix {

void lu(Matrix& A, Matrix& L, Matrix& U) {
  std::vector<int> ipiv(std::min(A.rows, A.cols));

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.rows, ipiv.data());

  // copy out U and L
  int n_diag = U.min_dim();
  for (int j = 0; j < n_diag; ++j) {
    cblas_dcopy(j + 1, &A(0, j), 1, &U(0, j), 1);
    cblas_dcopy(A.rows - j, &A(j, j), 1, &L(j, j), 1);
  }

  // copy out the rest of U if trapezoidal
  if (U.cols > U.rows){
    cblas_dcopy((U.cols - U.rows) * U.rows, &A(0, n_diag), 1, &U(0, n_diag), 1);
  }

  // L: set diagonal to 1 and upper triangular matrix to 0
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'U', L.rows, L.cols, 0, 1, &L, L.rows);

  // U: set lower triangular to 0?
}

void qr(const Matrix& A, Matrix& Q, Matrix& R) {
  int k = std::min(A.rows, A.cols);
  std::vector<double> tau(k);
  Matrix A_copy(A);
  LAPACKE_dgeqrf(
    LAPACK_COL_MAJOR, A_copy.rows, A_copy.cols, &A_copy, A_copy.rows, tau.data()
  );
  //Copy upper triangular (or trapezoidal) part of A_copy to R
  for (int j = 0; j < R.cols; j++) {
    cblas_dcopy(std::min(j+1, R.rows), &A_copy(0, j), 1, &R(0, j), 1);
  }
  //Copy lower triangular of A_copy to Q
  for (int j = 0; j < Q.cols; j++) {
    Q(j,j) = 1.0;
    cblas_dcopy(Q.rows-j-1, &A_copy(j+1, j), 1, &Q(j+1, j), 1);
  }
  LAPACKE_dorgqr(
    LAPACK_COL_MAJOR, Q.rows, Q.cols, Q.cols, &Q, Q.rows, tau.data()
  );
}

} // namespace Hatrix
