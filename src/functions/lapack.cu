#include "Hatrix/classes/Matrix.h"

#include "Hatrix/util/handle.h"
#include "cusolverDn.h"
#include <algorithm>

namespace Hatrix {

  extern cublasHandle_t blasH;
  extern cusolverDnHandle_t solvH;

void lu(Matrix& A, Matrix& L, Matrix& U) {

  int Lwork;
  cusolverDnDgetrf_bufferSize(solvH, A.rows, A.cols, &A, A.rows, &Lwork);

  double* work;
  cudaMalloc(reinterpret_cast<void**>(&work), Lwork);

  cusolverDnDgetrf(solvH, A.rows, A.cols, &A, A.rows, work, nullptr, nullptr);

  cudaDeviceSynchronize();
  cudaFree(work);

  for (int i = 0; i < L.cols && i < A.cols; i++) {
    double one = 1;
    cudaMemcpy(&L + i * L.rows + i, &one, sizeof(double), cudaMemcpyHostToDevice);
    if (i + 1 < A.rows)
      cudaMemcpy(&L + i * L.rows + i + 1, &A + i * A.rows + i + 1, (A.rows - i - 1) * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  for (int i = 0; i < A.cols; i++) {
    cudaMemcpy(&U + i * U.rows, &A + i * A.rows, std::min(i + 1, (int)A.rows) * sizeof(double), cudaMemcpyDeviceToDevice);
  }

}

void qr(Matrix& A, Matrix& Q, Matrix& R) {

  int Lwork;
  double* tau, * work;

  cusolverDnDgeqrf_bufferSize(solvH, A.rows, A.cols, &A, A.rows, &Lwork);

  cudaMalloc(reinterpret_cast<void**>(&work), Lwork);
  cudaMalloc(reinterpret_cast<void**>(&tau), std::min(A.rows, A.cols) * sizeof(double));
  cusolverDnDgeqrf(solvH, A.rows, A.cols, &A, A.rows, tau, work, Lwork, nullptr);

  cudaDeviceSynchronize();
  cudaMemcpy(&Q, &A, Q.rows * std::min(A.cols, Q.cols) * sizeof(double), cudaMemcpyDeviceToDevice);

  for (int i = 0; i < A.cols; i++) {
    cudaMemcpy(&R + i * R.rows, &A + i * A.rows, std::min(i + 1, (int)A.rows) * sizeof(double), cudaMemcpyDeviceToDevice);
  }
  cusolverDnDorgqr(solvH, Q.rows, Q.cols, Q.cols, &Q, Q.rows, tau, work, Lwork, nullptr);

  cudaDeviceSynchronize();
  cudaFree(tau);
  cudaFree(work);

}

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V) {
  
  int Lwork;
  cusolverDnDgesvd_bufferSize(solvH, A.rows, A.cols, &Lwork);
  double* work;

  cudaMalloc(reinterpret_cast<void**>(&work), Lwork);
  cusolverDnDgesvd(solvH, 'A', 'A', A.rows, A.cols, &A, A.rows, &S, &U, U.rows, &V, V.rows, work, Lwork, nullptr, nullptr);

  cudaDeviceSynchronize();
  for (int i = std::min(S.rows, S.cols); i > 0; i--) {
    double zero = 0;
    cudaMemcpy(&S + i * S.rows + i, &S + i, sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&S + i, &zero, sizeof(double), cudaMemcpyHostToDevice);
  }

  cudaFree(work);
}

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank) {

  return 0;
}

} // namespace Hatrix
