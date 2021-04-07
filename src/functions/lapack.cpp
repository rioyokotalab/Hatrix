#include "Hatrix/classes/Matrix.h"

#include "lapacke.h"


namespace Hatrix {

void getrf(Matrix& A) {
  int ipiv[std::min(A.rows, A.cols)];
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.rows, A.cols, &A, A.cols, ipiv);
}

void qr(Matrix& A, Matrix& Q, Matrix& R) {
  int k = std::min(A.rows, A.cols);
  double tau[k];
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.rows, A.cols, &A, A.cols, tau);
  for(int i=0; i<std::min(Q.rows, Q.cols); i++) Q(i, i) = 1.0;
  for(int i=0; i<A.rows; i++) {
    for(int j=0; j<A.cols; j++) {
      if(j >= i)
        R(i, j) = A(i, j);
      else
        Q(i, j) = A(i, j);
    }
  }
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q.rows, Q.cols, k, &Q, Q.cols, tau);
}

} // namespace Hatrix
