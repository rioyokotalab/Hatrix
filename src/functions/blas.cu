#include <cassert>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/util/context.h"
#include "cublas_v2.h"

namespace Hatrix {

void matmul(const Matrix& A, const Matrix& B, Matrix& C, bool transA,
            bool transB, double alpha, double beta) {
  cublasHandle_t handle = Context::cublasH[Context::sid];
  Context::iterate();
  cublasDgemm(handle, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              transB ? CUBLAS_OP_T : CUBLAS_OP_N, C.rows, C.cols,
              transA ? A.rows : A.cols, &alpha, &A, A.stride, &B, B.sride,
              &beta, &C, C.stride);
};

Matrix matmul(const Matrix& A, const Matrix& B, bool transA, bool transB,
              double alpha) {
  Matrix C(transA ? A.cols : A.rows, transB ? B.rows : B.cols);
  matmul(A, B, C, transA, transB, alpha, 0);
  return C;
}

void triangular_matmul(const Matrix& A, Matrix& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cublasHandle_t handle = Context::cublasH[Context::sid];
  Context::iterate();
  cublasDtrmm(handle, side == Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
              uplo == Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
              transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, B.rows, B.cols,
              &alpha, &A, A.strie, &B, B.strie, &B, B.stride);
}

void solve_triangular(const Matrix& A, Matrix& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cublasHandle_t handle = Context::cublasH[Context::sid];
  Context::iterate();
  cublasDtrsm(handle, side == Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
              uplo == Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
              transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, B.rows, B.cols,
              &alpha, &A, A.stride, &B, B.stride);
}

void scale(Matrix& A, double alpha) {
  cublasHandle_t handle = Context::cublasH[Context::sid];
  Context::iterate();
  cublasDscal(handle, A.rows * A.cols, &alpha, &A, 1);
}

}  // namespace Hatrix
