#include "Hatrix/functions/lapack.h"

#include <algorithm>
#include <cassert>

#include "cusolverDn.h"

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

extern cublasHandle_t blasH;
extern cusolverDnHandle_t solvH;

void lu(Matrix &A, Matrix &L, Matrix &U) {
  int Lwork;
  cusolverDnDgetrf_bufferSize(solvH, A.rows, A.cols, &A, A.rows, &Lwork);

  double *work;
  cudaMalloc(reinterpret_cast<void **>(&work), Lwork);

  cusolverDnDgetrf(solvH, A.rows, A.cols, &A, A.rows, work, nullptr, nullptr);

  cudaDeviceSynchronize();
  cudaFree(work);

  for (int i = 0; i < L.cols && i < A.cols; i++) {
    double one = 1;
    cudaMemcpy(&L + i * L.rows + i, &one, sizeof(double),
               cudaMemcpyHostToDevice);
    if (i + 1 < A.rows)
      cudaMemcpy(&L + i * L.rows + i + 1, &A + i * A.rows + i + 1,
                 (A.rows - i - 1) * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  for (int i = 0; i < A.cols; i++) {
    cudaMemcpy(&U + i * U.rows, &A + i * A.rows,
               std::min(i + 1, (int)A.rows) * sizeof(double),
               cudaMemcpyDeviceToDevice);
  }
}

void qr(Matrix &A, Matrix &Q, Matrix &R) {
  int Lwork;
  double *tau, *work;

  cusolverDnDgeqrf_bufferSize(solvH, A.rows, A.cols, &A, A.rows, &Lwork);

  cudaMalloc(reinterpret_cast<void **>(&work), Lwork);
  cudaMalloc(reinterpret_cast<void **>(&tau),
             std::min(A.rows, A.cols) * sizeof(double));
  cusolverDnDgeqrf(solvH, A.rows, A.cols, &A, A.rows, tau, work, Lwork,
                   nullptr);

  cudaDeviceSynchronize();
  cudaMemcpy(&Q, &A, Q.rows * std::min(A.cols, Q.cols) * sizeof(double),
             cudaMemcpyDeviceToDevice);

  for (int i = 0; i < A.cols; i++) {
    cudaMemcpy(&R + i * R.rows, &A + i * A.rows,
               std::min(i + 1, (int)A.rows) * sizeof(double),
               cudaMemcpyDeviceToDevice);
  }
  cusolverDnDorgqr(solvH, Q.rows, Q.cols, Q.cols, &Q, Q.rows, tau, work, Lwork,
                   nullptr);

  cudaDeviceSynchronize();
  cudaFree(tau);
  cudaFree(work);
}

void svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V) {
  double *work, *s;
  cudaMallocManaged(reinterpret_cast<void **>(&s),
                    std::min(S.rows, S.cols) * sizeof(double));

  int Lwork;

  cusolverDnDgesvd_bufferSize(solvH, A.rows, A.cols, &Lwork);
  cudaMalloc(reinterpret_cast<void **>(&work), Lwork);

  cusolverDnDgesvd(solvH, 'S', 'S', A.rows, A.cols, &A, A.rows, s, &U, U.rows,
                   &V, V.rows, work, Lwork, nullptr, nullptr);

  cudaDeviceSynchronize();

  for (int i = 0; i < std::min(S.rows, S.cols); i++) {
    S(i, i) = s[i];
  }

  cudaFree(s);
  cudaFree(work);
}

double truncated_svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V, int64_t rank) {
  assert(rank < A.min_dim());
  svd(A, U, S, V);
  double expected_err = 0;
  for (int64_t k = rank; k < A.min_dim(); ++k)
    expected_err += S(k, k) * S(k, k);
  U.shrink(U.rows, rank);
  S.shrink(rank, rank);
  V.shrink(rank, V.cols);
  return expected_err;
}

double norm(const Matrix& A) {
  double result;
  cublasDnrm2(blasH, A.rows * A.cols, &A, 1, &result);
  cudaDeviceSynchronize();
  return result;
}

}  // namespace Hatrix
