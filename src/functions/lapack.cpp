#include "Hatrix/classes/Matrix.h"

#include "mkl.h"

#include <algorithm>
#include <vector>


namespace Hatrix {

void getrf(Matrix& A, Matrix& L, Matrix& U) {
  std::vector<int> ipiv(std::min(A.rows, A.cols));
  int info, n = A.rows * A.cols, one = 1;
  Matrix B(A.rows, A.cols);
  dcopy(&n, A.data_, &one, B.data_, &one);
  
  dgetrf(&B.rows, &B.cols, B.data_, &B.rows, ipiv.data(), &info);

  // copy out U and L
  int n_diag = U.min_dim();
  for (int i = 0; i < n_diag; ++i) {
    int nr = i + 1;
    dcopy(&nr, B.data_ + i * B.rows, &one, U.data_ + i * U.rows, &one);
    nr = B.rows - i;
    dcopy(&nr, B.data_ + i * B.rows + i, &one, L.data_ + i * L.rows + i, &one);
  }

  // copy out the rest of U if trapezoidal
  if (U.cols > U.rows){
    int nr = (U.cols - U.rows) * U.rows;
    dcopy(&nr, B.data_ + n_diag * B.rows, &one, U.data_ + n_diag * U.rows, &one);
  }
  
  // L: set diagonal to 1 and upper triangular matrix to 0
  double dl_one = 1;
  double zero = 0;
  char fill = 'U';
  dlaset(&fill, &L.rows, &L.cols, &zero, &dl_one, L.data_, &L.rows);

  // U: set lower triangular to 0?
}

void qr(const Matrix& A, Matrix& Q, Matrix& R) {
  int k = std::min(A.rows, A.cols);
  std::vector<double> tau(k);

  int l = A.rows * A.cols, one = 1;
  int lwork = A.cols * 64, info;
  std::vector<double> work(lwork);

  dcopy(&l, A.data_, &one, Q.data_, &one);
  dgeqrf(
    &Q.rows, &Q.cols, Q.data_, &Q.rows,
    tau.data(), work.data(), &lwork, &info
  );

  for (int i = 0; i < R.cols; i++) {
    int nr = i + 1;
    dcopy(&nr, Q.data_ + i * Q.rows, &one, R.data_ + i * R.rows, &one);
  }
  dorgqr(
    &Q.rows, &Q.cols, &Q.cols,
    Q.data_, &Q.rows,
    tau.data(), work.data(), &lwork, &info
  );
}

} // namespace Hatrix
