#include "Hatrix/functions/blas.h"

#include <cassert>

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

void matmul(const Matrix& A, const Matrix& B, Matrix& C, bool transA,
            bool transB, double alpha, double beta) {
  assert(transA ? A.cols : A.rows == C.rows);
  assert(transB ? B.rows : B.cols == C.cols);
  assert(transA ? A.rows : A.cols == transB ? B.cols : B.rows);
  cblas_dgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, C.rows, C.cols,
              transA ? A.rows : A.cols, alpha, &A, A.stride, &B, B.stride, beta,
              &C, C.stride);
};

Matrix matmul(const Matrix& A, const Matrix& B, bool transA, bool transB,
              double alpha) {
  if (transA) {
    if (transB) { assert(A.rows == B.cols); }
    else        { assert(A.rows == B.rows); }
  }
  else {
    if (transB) { assert(A.cols == B.cols); }
    else        { assert(A.cols == B.rows); }
  }

  Matrix C(transA ? A.cols : A.rows, transB ? B.rows : B.cols);
  matmul(A, B, C, transA, transB, alpha, 0);
  return C;
}

void triangular_matmul(const Matrix& A, Matrix& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cblas_dtrmm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

Matrix triangular_matmul(const Matrix& A, const Matrix& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  Matrix C(B);
  triangular_matmul(A, C, side, uplo, transA, diag, alpha);
  return C;
}

void solve_triangular(const Matrix& A, Matrix& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cblas_dtrsm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

void scale(Matrix& A, double alpha) {
  for (int64_t j=0; j<A.cols; ++j) {
    cblas_dscal(A.rows, alpha, &A(0, j), 1);
  }
}

}  // namespace Hatrix
