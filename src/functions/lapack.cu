#include "Hatrix/functions/lapack.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime_api.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/util/context.h"

namespace Hatrix {

void dgetrf(double* a, int64_t m, int64_t n, int64_t lda, int64_t* ipiv) {
  void* args[7];
  runtime_args(args, arg_t::SOLV);
  cusolverDnHandle_t handle = reinterpret_cast<cusolverDnHandle_t>(args[0]);
  cusolverDnParams_t params = reinterpret_cast<cusolverDnParams_t>(args[1]);
  void* work = args[2], * work_host = args[4];
  size_t Lwork = *reinterpret_cast<size_t*>(args[3]), Lwork_host = *reinterpret_cast<size_t*>(args[5]);
  int* dev_info = reinterpret_cast<int*>(args[6]);

  size_t workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf;
  cusolverDnXgetrf_bufferSize(handle, params, m, n, CUDA_R_64F, a, lda, CUDA_R_64F, &workspaceInBytesOnDevice_getrf, &workspaceInBytesOnHost_getrf);
  if (workspaceInBytesOnDevice_getrf <= Lwork && workspaceInBytesOnHost_getrf <= Lwork_host)
    cusolverDnXgetrf(handle, params, m, n, CUDA_R_64F, a, lda, ipiv, CUDA_R_64F, work, Lwork, work_host, Lwork_host, dev_info);
  else
    fprintf(stderr, "Insufficient work for DGETRF. %zu, %zu\n", workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf);
}

void dtricpy(int kind, int uplo, int diag, int64_t m, int64_t n, double* dst, int64_t ldd, const double* src, int64_t lds) {
  void* args[1];
  runtime_args(args, arg_t::STREAM);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(args[0]);

  lds = lds == 0 ? ldd : lds;
  bool diag_unit = static_cast<cublasDiagType_t>(diag) == CUBLAS_DIAG_UNIT;

  if (static_cast<cublasFillMode_t>(uplo) == CUBLAS_FILL_MODE_LOWER) {
    int64_t n_col = m - diag_unit;
    for (int64_t i = 0; i < n && n_col > 0; i++) {
      int offset = m - n_col;
      cudaMemcpyAsync(dst + i * ldd + offset, src + i * lds + offset, n_col * sizeof(double), static_cast<cudaMemcpyKind>(kind), stream);
      n_col--;
    }
  }
  else if (static_cast<cublasFillMode_t>(uplo) == CUBLAS_FILL_MODE_UPPER) {
    int64_t n_col = 1;
    for (int64_t i = diag_unit; i < n; i++) {
      cudaMemcpyAsync(dst + i * ldd, src + i * lds, n_col * sizeof(double), static_cast<cudaMemcpyKind>(kind), stream);
      n_col = n_col == m ? m : n_col + 1;
    }
  }

  if (diag_unit) {
    double one = 1;
    for (int i = 0; i < m && i < n; i++)
      cudaMemcpyAsync(dst + i * ldd + i, &one, sizeof(double), cudaMemcpyHostToDevice, stream);
  }
}

void lu(Matrix &A, Matrix &L, Matrix &U) {
  mode_t old = parallel_mode(mode_t::SERIAL);
  dgetrf(&A, A.rows, A.cols, A.rows, nullptr);
  dtricpy(cudaMemcpyDefault, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, A.rows, A.cols, &L, L.rows, &A, A.rows);
  parallel_mode(old);
  dtricpy(cudaMemcpyDefault, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, A.rows, A.cols, &U, U.rows, &A, A.rows);
}

void qr(Matrix &A, Matrix &Q, Matrix &R) {

}

void svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V) {
  
}

double truncated_svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V, int64_t rank) {
  
}

double norm(const Matrix& A) {
  void* args[1];
  runtime_args(args, arg_t::BLAS);
  cublasHandle_t blasH = reinterpret_cast<cublasHandle_t>(args[0]);
  
  double result;
  cublasDnrm2(blasH, A.rows * A.cols, &A, 1, &result);
  cudaDeviceSynchronize();
  return result;
}

}  // namespace Hatrix
