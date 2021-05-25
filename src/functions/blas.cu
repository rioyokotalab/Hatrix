#include "Hatrix/functions/blas.h"

#include "cublas_v2.h"
#include <cassert>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

void matmul(const Matrix &A, const Matrix &B, Matrix &C, bool transA,
            bool transB, double alpha, double beta) {
  void* args[1];
  runtime_args(args, arg_t::BLAS);
  cublasHandle_t blasH = reinterpret_cast<cublasHandle_t>(args[0]);
  cublasDgemm(blasH, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              transB ? CUBLAS_OP_T : CUBLAS_OP_N, C.rows, C.cols,
              transA ? A.rows : A.cols, &alpha, &A, A.rows, &B, B.rows, &beta,
              &C, C.rows);

};

void triangular_matmul(const Matrix& A, Matrix& B, int side, int uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  void* args[1];
  runtime_args(args, arg_t::BLAS);
  cublasHandle_t blasH = reinterpret_cast<cublasHandle_t>(args[0]);
  cublasDtrmm(blasH, side == Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
              uplo == Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
	      transA ? CUBLAS_OP_T : CUBLAS_OP_N, 
	      diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, B.rows, B.cols, &alpha, 
	      &A, A.rows, &B, B.rows, &B, B.rows);
}

void solve_triangular(const Matrix &A, Matrix &B, int side, int uplo, bool diag,
                      bool transA, double alpha) {
  void* args[1];
  runtime_args(args, arg_t::BLAS);
  cublasHandle_t blasH = reinterpret_cast<cublasHandle_t>(args[0]);
  cublasDtrsm(blasH, side == Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
              uplo == Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
              transA ? CUBLAS_OP_T : CUBLAS_OP_N,
              diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, B.rows, B.cols,
              &alpha, &A, A.rows, &B, B.rows);
}

void scale(Matrix& A, double alpha) {
  void* args[1];
  runtime_args(args, arg_t::BLAS);
  cublasHandle_t blasH = reinterpret_cast<cublasHandle_t>(args[0]);
  cublasDscal(blasH, A.rows * A.cols, &alpha, &A, 1);
}

} // namespace Hatrix
