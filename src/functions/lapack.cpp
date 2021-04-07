#include "Hatrix/classes/Matrix.h"

#include "mkl.h"

#include <algorithm>
#include <vector>


namespace Hatrix {

void getrf(Matrix& A) {
  std::vector<int> ipiv(std::min(A.rows, A.cols));
  int info;
  dgetrf(&A.rows, &A.cols, A.data_, &A.rows, ipiv.data(), &info);
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
