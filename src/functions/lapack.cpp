#include "Hatrix/classes/Matrix.h"

#include "mkl_cblas.h"
#include "mkl_lapacke.h"

#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdint>
using std::uint64_t;


namespace Hatrix {

void lu(Matrix& A, Matrix& L, Matrix& U) {
  // check dimensions
  assert(L.rows == A.rows);
  assert(L.cols == U.rows && L.cols == A.min_dim());
  assert(U.cols == A.cols);

  std::vector<int> ipiv(std::min(A.rows, A.cols));

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.rows, ipiv.data());

  // U: set lower triangular matrix to 0
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', U.rows, U.cols, 0, 0, &L, U.rows);

  // copy out U and L
  uint64_t n_diag = U.min_dim();
  for (uint64_t j=0; j<n_diag; ++j) {
    cblas_dcopy(j + 1, &A(0, j), 1, &U(0, j), 1);
    cblas_dcopy(A.rows - j, &A(j, j), 1, &L(j, j), 1);
  }

  // copy out the rest of U if trapezoidal
  if (U.cols > U.rows){
    cblas_dcopy((U.cols - U.rows) * U.rows, &A(0, n_diag), 1, &U(0, n_diag), 1);
  }

  // L: set diagonal to 1 and upper triangular matrix to 0
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'U', L.rows, L.cols, 0, 1, &L, L.rows);
}

void qr(Matrix& A, Matrix& Q, Matrix& R) {
  // check dimensions
  assert(Q.rows == A.rows);
  assert(Q.cols == R.rows);
  assert(R.cols == A.cols);

  uint64_t k = std::min(A.rows, A.cols);
  std::vector<double> tau(k);
  LAPACKE_dgeqrf(
    LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.rows, tau.data()
  );
  //Copy upper triangular (or trapezoidal) part of A to R
  for (uint64_t j=0; j<R.cols; j++) {
    cblas_dcopy(std::min(j+1, R.rows), &A(0, j), 1, &R(0, j), 1);
  }
  //Copy lower triangular of A_copy to Q
  for (uint64_t j=0; j<Q.cols; j++) {
    Q(j,j) = 1.0;
    cblas_dcopy(Q.rows-j-1, &A(j+1, j), 1, &Q(j+1, j), 1);
  }
  LAPACKE_dorgqr(
    LAPACK_COL_MAJOR, Q.rows, Q.cols, k, &Q, Q.rows, tau.data()
  );
}

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V){
    // check dimensions
  assert(U.rows == A.rows);
  assert(S.cols == S.rows && S.cols == A.min_dim());
  assert(U.cols == S.cols && V.rows == S.rows);
  assert(V.cols == A.cols);

  std::vector<double> Sdiag(S.rows);
  std::vector<double> work(S.rows-1);
  LAPACKE_dgesvd(
    LAPACK_COL_MAJOR, 'S', 'S',
    A.rows, A.cols, &A, A.rows,
    Sdiag.data(),
    &U, U.rows,
    &V, V.rows,
    work.data()
  );
  S = 0;
  for(uint64_t i=0; i<S.rows; i++){
    S(i, i) = Sdiag[i];
  }
}

double truncated_svd(
  Matrix& A, Matrix& U, Matrix& S, Matrix& V, uint64_t rank
) {
  assert(rank < A.min_dim());
  svd(A, U, S, V);
  double expected_err = 0;
  for (uint64_t k=rank; k<A.min_dim(); ++k) expected_err += S(k, k) * S(k, k);
  U.shrink(U.rows, rank);
  S.shrink(rank, rank);
  V.shrink(rank, V.cols);
  return expected_err;
}

} // namespace Hatrix
