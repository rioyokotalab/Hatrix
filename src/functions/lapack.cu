#include "Hatrix/classes/Matrix.h"

#include "Hatrix/handle.h"
#include "cusolverDn.h"

namespace Hatrix {

  extern cublasHandle_t blasH;
  extern cusolverDnHandle_t solvH;

void lu(Matrix& A, Matrix& L, Matrix& U) {

  int Lwork;
  cusolverDnDgetrf_bufferSize(solvH, A.rows, A.cols, &A, A.rows, &Lwork);

  double* work;
  cudaMalloc(reinterpret_cast<void**>(&work), Lwork);

  cusolverDnDgetrf(solvH, A.rows, A.cols, &A, A.rows, work, nullptr, nullptr);

  // copy out U and L

  // copy out the rest of U if trapezoidal

  // L: set diagonal to 1 and upper triangular matrix to 0

  // U: set lower triangular to 0?

  cudaDeviceSynchronize();
  cudaFree(work);

}

void qr(const Matrix& A, Matrix& Q, Matrix& R) {

  int Lwork, k = std::min(A.rows, A.cols);
  double* tau, * work;

  cublasDcopy(blasH, A.rows * A.cols, &A, 1, &Q, 1);
  cusolverDnDgeqrf_bufferSize(solvH, Q.rows, Q.cols, &Q, Q.rows, &Lwork);

  cudaMalloc(reinterpret_cast<void**>(&work), Lwork);
  cudaMalloc(reinterpret_cast<void**>(&tau), k * sizeof(double));
  cusolverDnDgeqrf(solvH, Q.rows, Q.cols, &Q, Q.rows, tau, work, Lwork, nullptr);

  double one = 1, zero = 0;
  cublasDgeam(blasH, CUBLAS_OP_N, CUBLAS_OP_N, Q.rows, Q.cols, &one, &Q, Q.rows, &zero, &R, R.rows, &R, R.rows);
  cusolverDnDorgqr(solvH, Q.rows, Q.cols, k, &Q, Q.rows, tau, work, Lwork, nullptr);

  cudaDeviceSynchronize();
  cudaFree(tau);
  cudaFree(work);

}

} // namespace Hatrix
