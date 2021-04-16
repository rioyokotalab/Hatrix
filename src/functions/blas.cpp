#include "Hatrix/functions/blas.h"

#include "Hatrix/classes/Matrix.h"

#include "mkl_cblas.h"


namespace Hatrix {

void matmul(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool transA, bool transB, double alpha, double beta
) {
  cblas_dgemm(
    CblasColMajor,
    transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
    C.rows, C.cols, transA ? A.rows : A.cols,
    alpha, &A, A.rows, &B, B.rows,
    beta, &C, C.rows
  );
};

void solve_triangular(
  const Matrix& A, Matrix& B,
  int side, int uplo, bool diag, bool transA, double alpha
) {
  cblas_dtrsm(
    CblasColMajor,
    side == TRSMLeft ? CblasLeft :  CblasRight,
    uplo == TRSMUpper ? CblasUpper :  CblasLower,
    transA ? CblasTrans : CblasNoTrans, diag ? CblasUnit : CblasNonUnit,
    B.rows, B.cols,
    alpha, &A, A.rows, &B, B.rows
  );
}

} // namespace Hatrix
