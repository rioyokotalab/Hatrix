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
  char jobu = 'S', jobv = 'S';
  int64_t m = A.rows, n = A.cols, iters = 2;
  int64_t lda = A.rows, ldu = U.rows, ldv = V.rows;

  rank = rank > m ? m : rank;
  rank = rank > n ? n : rank;
  p = p < 0 ? 0 : p;
  p = p + rank > m ? m - rank : p;
  p = p + rank > n ? n - rank : p;

  cusolverDnParams_t params_gesvdr;
  cusolverDnCreateParams(&params_gesvdr);

  size_t workspaceInBytesOnDevice_gesvdr, workspaceInBytesOnHost_gesvdr;
  cusolverDnXgesvdr_bufferSize(solvH, params_gesvdr, jobu, jobv, m, n, rank, p, iters, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S,
    CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, &workspaceInBytesOnDevice_gesvdr, &workspaceInBytesOnHost_gesvdr);
  
  double* Work_host = (double*)malloc(workspaceInBytesOnHost_gesvdr), *Work_dev;
  cudaMalloc(reinterpret_cast<void **>(&Work_dev), Lwork);

  cusolverDnXgesvdr(solvH, params_gesvdr, jobu, jobv, m, n, rank, p, iters, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S, 
    CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, Work_dev, workspaceInBytesOnDevice_gesvdr, Work_host, workspaceInBytesOnHost_gesvdr, nullptr);

  free(Work_host);
  cudaFree(Work_dev);
  cusolverDnDestroyParams(params_gesvdr);
  return 0.;
}

double norm(const Matrix& A) {
  double result;
  cublasDnrm2(blasH, A.rows * A.cols, &A, 1, &result);
  cudaDeviceSynchronize();
  return result;
}

}  // namespace Hatrix
