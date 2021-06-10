#include "Hatrix/functions/lapack.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <tuple>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "cuda_runtime_api.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/util/context.h"

namespace Hatrix {

void dgetrf(double* a, int64_t m, int64_t n, int64_t lda, int64_t* ipiv) {
  cusolverDnHandle_t handle = Context::cusolverH[Context::sid];
  cusolverDnParams_t params = Context::cusolverParams[Context::sid];
  void* work = Context::bufferOnDevice[Context::sid], * work_host = Context::bufferOnHost[Context::sid];
  size_t Lwork = Context::workspaceInBytesOnDevice, Lwork_host = Context::workspaceInBytesOnHost;
  int* dev_info = Context::info + Context::sid;
  Context::iterate();

  size_t workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf;
  cusolverDnXgetrf_bufferSize(handle, params, m, n, CUDA_R_64F, a, lda, CUDA_R_64F, &workspaceInBytesOnDevice_getrf, &workspaceInBytesOnHost_getrf);
  if (workspaceInBytesOnDevice_getrf <= Lwork && workspaceInBytesOnHost_getrf <= Lwork_host)
    cusolverDnXgetrf(handle, params, m, n, CUDA_R_64F, a, lda, ipiv, CUDA_R_64F, work, Lwork, work_host, Lwork_host, dev_info);
  else
    fprintf(stderr, "Insufficient work for DGETRF. %zu, %zu\n", workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf);
}

void dtricpy(int kind, int uplo, int diag, int64_t m, int64_t n, double* dst, int64_t ldd, const double* src, int64_t lds) {
  cudaStream_t stream = Context::stream[Context::sid];
  Context::iterate();

  lds = lds == 0 ? ldd : lds;
  bool diag_unit = static_cast<cublasDiagType_t>(diag) == CUBLAS_DIAG_UNIT;

  if (static_cast<cublasFillMode_t>(uplo) == CUBLAS_FILL_MODE_LOWER) {
    int64_t n_col = m - diag_unit;
    for (int64_t i = 0; i < n && n_col > 0; i++) {
      int64_t offset = m - n_col;
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
  bool forking = Context::forking;
  Context::forking = false;
  dgetrf(&A, A.rows, A.cols, A.rows, nullptr);
  dtricpy(cudaMemcpyDefault, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_UNIT, A.rows, A.cols, &L, L.rows, &A, A.rows);
  Context::forking = forking;
  dtricpy(cudaMemcpyDefault, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, A.rows, A.cols, &U, U.rows, &A, A.rows);
}

void qr(Matrix &A, Matrix &Q, Matrix &R) {

}

void dgesvd(int64_t m, int64_t n, double* A, int64_t lda, double* S, double* U, int64_t ldu, double* V, int64_t ldv) {
  cusolverDnHandle_t handle = Context::cusolverH[Context::sid];
  cusolverDnParams_t params = Context::cusolverParams[Context::sid];
  void* work = Context::bufferOnDevice[Context::sid], * work_host = Context::bufferOnHost[Context::sid];
  size_t Lwork = Context::workspaceInBytesOnDevice, Lwork_host = Context::workspaceInBytesOnHost;
  int* dev_info = Context::info + Context::sid;
  Context::iterate();

  size_t workspaceInBytesOnDevice_gesvd, workspaceInBytesOnHost_gesvd;
  cusolverDnXgesvd_bufferSize(handle, params, 'S', 'S', m, n, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S,
    CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, &workspaceInBytesOnDevice_gesvd, &workspaceInBytesOnHost_gesvd);

  if (workspaceInBytesOnDevice_gesvd <= Lwork && workspaceInBytesOnHost_gesvd <= Lwork_host)
    cusolverDnXgesvd(handle, params, 'S', 'S', m, n, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S, 
      CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, work, Lwork, work_host, Lwork_host, dev_info);
  else
    fprintf(stderr, "Insufficient work for DGESVD. %zu, %zu\n", workspaceInBytesOnDevice_gesvd, workspaceInBytesOnHost_gesvd);
}

void dsv2m(double* s, int64_t m, int64_t n, int64_t lds) {
  cudaStream_t stream = Context::stream[Context::sid];
  double* work = (double*)Context::bufferOnDevice[Context::sid];
  size_t Lwork = Context::workspaceInBytesOnDevice;
  Context::iterate();

  int64_t r = m > n ? n : m;
  if (Lwork >= r * sizeof(double)) {
    cudaMemcpyAsync(work, s, r * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(s, 0, sizeof(double) * lds * n, stream);
    for (int64_t i = 0; i < r; i++)
      cudaMemcpyAsync(s + i * lds + i, work + i, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  }
  else
    fprintf(stderr, "Insufficient work for extending singular vector to matrix. %zu\n", r * sizeof(double));
}

void dvt2v(double* vt, int64_t m, int64_t n, int64_t ldvt, int64_t ldv) {
  cudaStream_t stream = Context::stream[Context::sid];
  double* work = (double*)Context::bufferOnDevice[Context::sid];
  size_t Lwork = Context::workspaceInBytesOnDevice;
  Context::iterate();

  ldvt = ldvt < m ? m : ldvt;
  ldv = ldv < n ? n : ldv;

  if (Lwork >= sizeof(double) * m * n) {
    for (int64_t i = 0; i < n; i++)
      for (int64_t j = 0; j < m; j++)
        cudaMemcpyAsync(work + j * n + i, vt + i * ldvt + j, sizeof(double), cudaMemcpyDeviceToDevice, stream);

    cudaMemcpy2DAsync(vt, sizeof(double) * ldv, work, sizeof(double) * n, sizeof(double) * n, m, cudaMemcpyDeviceToDevice, stream);
  }
  else
    fprintf(stderr, "Insufficient work for tranposing vt to v. %zu\n", sizeof(double) * m * n);
}


void svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V) {
  bool forking = Context::forking;
  Context::forking = false;
  if (A.rows >= A.cols) {
    dgesvd(A.rows, A.cols, &A, A.rows, &S, &U, U.rows, &V, V.rows);
  }
  else {
    dvt2v(&A, A.rows, A.cols, A.rows, A.cols);
    dgesvd(A.cols, A.rows, &A, A.cols, &S, &V, V.cols, &U, U.cols);
    dvt2v(&U, U.cols, U.rows, U.cols, U.rows);
    dvt2v(&V, V.cols, V.rows, V.cols, V.rows);
  }
  Context::forking = forking;
  dsv2m(&S, S.rows, S.cols, S.rows);
}

double truncated_svd(Matrix &A, Matrix &U, Matrix &S, Matrix &V, int64_t rank) {
  assert(rank < A.min_dim());
  svd(A, U, S, V);
  Context::join();
  double expected_err = 0;
  for (int64_t k = rank; k < A.min_dim(); ++k)
    expected_err += S(k, k) * S(k, k);
  U.shrink(U.rows, rank);
  S.shrink(rank, rank);
  V.shrink(rank, V.cols);
  return std::sqrt(expected_err);
}

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank) {
  Matrix U(A.rows, A.min_dim());
  Matrix S(A.min_dim(), A.min_dim());
  Matrix V(A.min_dim(), A.cols);
  double expected_err = truncated_svd(A, U, S, V, rank);
  return {std::move(U), std::move(S), std::move(V), expected_err};
}

double norm(const Matrix& A) {
  cublasHandle_t handle = Context::cublasH[Context::sid];

  double result;
  cublasDnrm2(handle, A.rows * A.cols, &A, 1, &result);
  cudaDeviceSynchronize();
  return result;
}

}  // namespace Hatrix
