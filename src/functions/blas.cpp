#include "Hatrix/classes/Matrix.h"

#include "mkl.h"


namespace Hatrix {

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  char transa, char transb, double alpha, double beta
) {
  dgemm(
    &transa, &transb,
    &C.rows, &C.cols, &A.cols,
    &alpha, A.data_, &A.rows, B.data_, &B.rows,
    &beta, C.data_, &C.rows
  );
};

void trsm(
  const Matrix& A, Matrix& B,
  char side, char uplo, char transa, char diag, double alpha
) {
  dtrsm(
    &side, &uplo, &transa, &diag,
    &B.rows, &B.cols, &alpha,
    A.data_, &A.rows, B.data_, &B.rows
  );
}

} // namespace Hatrix
