#include "Hatrix/functions/blas.h"

#include "Hatrix/classes/Matrix.h"

#include "Hatrix/handle.h"
#include "cublas_v2.h"
#include "cusolverDn.h"

namespace Hatrix {

  cublasHandle_t blasH = nullptr;
  cusolverDnHandle_t solvH = nullptr;

  void init() {
    cublasCreate(&blasH);
    cusolverDnCreate(&solvH);
  }

  void terminate() {
    cublasDestroy(blasH); blasH = 0;
    cusolverDnDestroy(solvH); blasH = 0;
  }

void matmul(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool transA, bool transB, double alpha, double beta
) {
  cublasDgemm(
    blasH,
    transA ? CUBLAS_OP_T : CUBLAS_OP_N, transB ? CUBLAS_OP_T : CUBLAS_OP_N,
    C.rows, C.cols, transA ? A.rows : A.cols,
    &alpha, &A, A.rows, &B, B.rows,
    &beta, &C, C.rows
  );

  cudaDeviceSynchronize();
};

void solve_triangular(
  const Matrix& A, Matrix& B,
  int side, int uplo, bool diag, bool transA, double alpha
) {
  cublasDtrsm(
    blasH,
    side == Left ? CUBLAS_SIDE_LEFT :  CUBLAS_SIDE_RIGHT,
    uplo == Upper ? CUBLAS_FILL_MODE_UPPER :  CUBLAS_FILL_MODE_LOWER,
    transA ? CUBLAS_OP_T : CUBLAS_OP_N, diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
    B.rows, B.cols,
    &alpha, &A, A.rows, &B, B.rows
  );

  cudaDeviceSynchronize();
}

} // namespace Hatrix
