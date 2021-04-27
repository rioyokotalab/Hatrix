#include "Hatrix/functions/blas.h"

#include "Hatrix/classes/Matrix.h"

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include <cassert>


namespace Hatrix {

void matmul(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool transA, bool transB, double alpha, double beta
) {
  assert(transA ? A.cols : A.rows == C.rows);
  assert(transB ? B.rows : B.cols == C.cols);
  assert(transA ? A.rows : A.cols == transB ? B.cols : B.rows);
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
    side == Left ? CblasLeft :  CblasRight,
    uplo == Upper ? CblasUpper :  CblasLower,
    transA ? CblasTrans : CblasNoTrans, diag ? CblasUnit : CblasNonUnit,
    B.rows, B.cols,
    alpha, &A, A.rows, &B, B.rows
  );
}

} // namespace Hatrix
