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

template void array_copy(const double* from, double* to, int64_t size);

template void matmul(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, bool transA = false,
            bool transB = false, double alpha = 1.0, double beta = 1.0);

template Matrix<double> matmul(const Matrix<double>& A, const Matrix<double>& B, bool transA = false,
              bool transB = false, double alpha = 1.0);

template void syrk(const Matrix<double>& A, Matrix<double>& C, Mode uplo, bool transA, double alpha,
            double beta);


template void triangular_matmul(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha = 1.0);

template Matrix<double> triangular_matmul_out(const Matrix<double>& A, const Matrix<double>& B, Side side, Mode uplo,
			     bool transA, bool diag, double alpha = 1.0);

template void solve_triangular(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                      bool unit_diag, bool transA = false, double alpha = 1.0);

template void solve_diagonal(const Matrix<double>& D, Matrix<double>& B, Side side, double alpha = 1.0);

template void scale(Matrix<double>& A, double alpha);

template void row_scale(Matrix<double>& A, const Matrix<double>& D);

template void column_scale(Matrix<double>& A, const Matrix<double>& D);


template <typename DT>
void array_copy(const DT* from, DT* to, int64_t size) {
  cblas_dcopy(size, from, 1, to, 1);
}

template <typename DT>
void matmul(const Matrix<DT>& A, const Matrix<DT>& B, Matrix<DT>& C, bool transA,
            bool transB, double alpha, double beta) {
  assert(transA ? A.cols : A.rows == C.rows);
  assert(transB ? B.rows : B.cols == C.cols);
  assert(transA ? A.rows : A.cols == transB ? B.cols : B.rows);
  cblas_dgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, C.rows, C.cols,
              transA ? A.rows : A.cols, alpha, &A, A.stride, &B, B.stride, beta,
              &C, C.stride);
};

template <typename DT>
Matrix<DT> matmul(const Matrix<DT>& A, const Matrix<DT>& B, bool transA, bool transB,
              double alpha) {
  if (transA) {
    if (transB) { assert(A.rows == B.cols); }
    else        { assert(A.rows == B.rows); }
  }
  else {
    if (transB) { assert(A.cols == B.cols); }
    else        { assert(A.cols == B.rows); }
  }

  Matrix<DT> C(transA ? A.cols : A.rows, transB ? B.rows : B.cols);
  matmul(A, B, C, transA, transB, alpha, 0);
  return C;
}

template <typename DT>
void syrk(const Matrix<DT>& A, Matrix<DT>& C, Mode uplo, bool transA, double alpha,
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

template <typename DT>
void triangular_matmul(const Matrix<DT>& A, Matrix<DT>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cblas_dtrmm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

template <typename DT>
Matrix<DT> triangular_matmul_out(const Matrix<DT>& A, const Matrix<DT>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  Matrix<DT> C(B);
  triangular_matmul(A, C, side, uplo, transA, diag, alpha);
  return C;
}

template <typename DT>
void solve_triangular(const Matrix<DT>& A, Matrix<DT>& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cblas_dtrsm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols, alpha, &A,
              A.stride, &B, B.stride);
}

template <typename DT>
void solve_diagonal(const Matrix<DT>& D, Matrix<DT>& B, Side side, double alpha) {
  assert(side == Left ? D.cols == B.rows : B.cols == D.rows);

  for(int64_t j = 0; j < B.cols; j++) {
    for(int64_t i = 0; i < B.rows; i++) {
      B(i, j) = alpha * B(i, j) / (side == Left ? D(i, i) : D(j, j));
    }
  }
}

template <typename DT>
void scale(Matrix<DT>& A, double alpha) {
  for (int64_t j=0; j<A.cols; ++j) {
    cblas_dscal(A.rows, alpha, &A(0, j), 1);
  }
}

template <typename DT>
void row_scale(Matrix<DT>& A, const Matrix<DT>& D) {
  assert(D.rows == D.cols);
  assert(D.cols == A.rows);

  for(int i = 0; i < A.rows; i++) {
    cblas_dscal(A.cols, D(i, i), &A(i, 0), A.stride);
  }
}

template <typename DT>
void column_scale(Matrix<DT>& A, const Matrix<DT>& D) {
  assert(D.rows == D.cols);
  assert(D.rows == A.cols);

  for(int j = 0; j < A.cols; j++) {
    cblas_dscal(A.rows, D(j, j), &A(0, j), 1);
  }
}

}  // namespace Hatrix
