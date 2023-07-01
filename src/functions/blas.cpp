#include "Hatrix/functions/blas.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

void swap_row(Matrix& A, int64_t r1, int64_t r2) {
  cblas_dswap(A.rows, &A(r1, 0), A.stride, &A(r2, 0), A.stride);
}

void swap_col(Matrix& A, int64_t c1, int64_t c2) {
  cblas_dswap(A.cols, &A(0, c1), 1, &A(0, c2), 1);
}

void
array_copy(const double* from, double* to, int64_t size) {
  cblas_dcopy(size, from, 1, to, 1);
}

void
matmul(const Matrix& A, const Matrix& B, Matrix& C, bool transA,
       bool transB, double alpha, double beta) {
  assert((transA ? A.cols : A.rows) == C.rows);
  assert((transB ? B.rows : B.cols) == C.cols);
  assert((transA ? A.rows : A.cols) == (transB ? B.cols : B.rows));
  cblas_dgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, C.rows, C.cols,
              transA ? A.rows : A.cols, alpha, &A, A.stride, &B, B.stride, beta,
              &C, C.stride);
};

Matrix
matmul(const Matrix& A, const Matrix& B, bool transA, bool transB,
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

void
syrk(const Matrix& A, Matrix& C, Mode uplo, bool transA, double alpha,
     double beta) {
  assert(C.rows == C.cols);
  cblas_dsyrk(CblasColMajor,
              uplo == Lower ? CblasLower : CblasUpper,
              transA ? CblasTrans : CblasNoTrans,
              C.rows,
              transA ? A.rows : A.cols,
              alpha,
              &A,
              A.stride,
              beta,
              &C,
              C.stride);
}

void
triangular_matmul(const Matrix& A, Matrix& B, Side side, Mode uplo,
                  bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cblas_dtrmm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

Matrix
triangular_matmul_out(const Matrix& A, const Matrix& B, Side side, Mode uplo,
                      bool transA, bool diag, double alpha) {
  Matrix C(B);
  triangular_matmul(A, C, side, uplo, transA, diag, alpha);
  return C;
}

void
solve_triangular(const Matrix& A, Matrix& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cblas_dtrsm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

void solve_diagonal(const Matrix& D, Matrix& B, Side side, double alpha) {
  assert(side == Left ? D.cols == B.rows : B.cols == D.rows);

  for(int64_t j = 0; j < B.cols; j++) {
    for(int64_t i = 0; i < B.rows; i++) {
      B(i, j) = alpha * B(i, j) / (side == Left ? D(i, i) : D(j, j));
    }
  }
}

void scale(Matrix& A, double alpha) {
  if (A.numel() == 0) return;
  for (int64_t j=0; j<A.cols; ++j) {
    cblas_dscal(A.rows, alpha, &A(0, j), 1);
  }
}

void row_scale(Matrix& A, const Matrix& D) {
  assert(D.rows == D.cols);
  assert(D.cols == A.rows);

  if (A.numel() == 0) return;
  for(int i = 0; i < A.rows; i++) {
    cblas_dscal(A.cols, D(i, i), &A(i, 0), A.stride);
  }
}

void column_scale(Matrix& A, const Matrix& D) {
  assert(D.rows == D.cols);
  assert(D.rows == A.cols);

  if (A.numel() == 0) return;
  for(int j = 0; j < A.cols; j++) {
    cblas_dscal(A.rows, D(j, j), &A(0, j), 1);
  }
}

}  // namespace Hatrix
