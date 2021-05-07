
#include <funcs.h>

#include <cuda_runtime_api.h>
#include <cstdio>

using namespace Hatrix::gpu;

void Hatrix::gpu::dalloc(Stream& s, double** a_ptr, int m, int n, int lda, int value) {
  cudaMalloc(reinterpret_cast<void**>(a_ptr), sizeof(double) * lda * n);
  cudaMemsetAsync(reinterpret_cast<void*>(*a_ptr), value, sizeof(double) * lda * n, s);
}

void Hatrix::gpu::dcopy2D(Stream& s, cudaMemcpyKind kind, int m, int n, double* dst, int ldd, const double* src, int lds) {
  lds = lds == 0 ? ldd : lds;
  cudaMemcpy2DAsync(dst, sizeof(double) * ldd, src, sizeof(double) * lds, sizeof(double) * n, m, kind, s);
}

void Hatrix::gpu::dpotrf(Stream& s, cublasFillMode_t uplo, double* a, int64_t n, int64_t lda) {
  size_t workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf;
  cusolverDnXpotrf_bufferSize(s, s, uplo, n, CUDA_R_64F, a, lda, CUDA_R_64F, &workspaceInBytesOnDevice_getrf, &workspaceInBytesOnHost_getrf);
  if (workspaceInBytesOnDevice_getrf <= (size_t)s && workspaceInBytesOnHost_getrf <= s.Lwork_host)
    cusolverDnXpotrf(s, s, uplo, n, CUDA_R_64F, a, lda, CUDA_R_64F, s, (size_t)s, s.Work_host, s.Lwork_host, s);
  else
    fprintf(stderr, "Insufficient work for DPOTRF.\n");
}

void Hatrix::gpu::dgetrf(Stream& s, double* a, int64_t m, int64_t n, int64_t lda, int64_t* ipiv) {
  size_t workspaceInBytesOnDevice_getrf, workspaceInBytesOnHost_getrf;
  cusolverDnXgetrf_bufferSize(s, s, m, n, CUDA_R_64F, a, lda, CUDA_R_64F, &workspaceInBytesOnDevice_getrf, &workspaceInBytesOnHost_getrf);
  if (workspaceInBytesOnDevice_getrf <= (size_t)s && workspaceInBytesOnHost_getrf <= s.Lwork_host)
    cusolverDnXgetrf(s, s, m, n, CUDA_R_64F, a, lda, ipiv, CUDA_R_64F, s, (size_t)s, s.Work_host, s.Lwork_host, s);
  else
    fprintf(stderr, "Insufficient work for DGETRF.\n");
}

void Hatrix::gpu::dorth(Stream& s, double* q, double* a, int64_t m, int64_t nq, int64_t na, int64_t ldq, int64_t lda) {
  size_t workspaceInBytesOnDevice_geqrf, workspaceInBytesOnHost_geqrf;
  double* tau = (double*)(void*)s;
  size_t Ltau = m > na ? na : m, Lwork = (size_t)s - Ltau * sizeof(double);
  cusolverDnXgeqrf_bufferSize(s, s, m, na, CUDA_R_64F, a, lda, CUDA_R_64F, tau, CUDA_R_64F, &workspaceInBytesOnDevice_geqrf, &workspaceInBytesOnHost_geqrf);
  if (workspaceInBytesOnDevice_geqrf <= Lwork && workspaceInBytesOnHost_geqrf <= s.Lwork_host)
    cusolverDnXgeqrf(s, s, m, na, CUDA_R_64F, a, lda, CUDA_R_64F, tau, CUDA_R_64F, tau + Ltau, Lwork, s.Work_host, s.Lwork_host, s);
  else
    fprintf(stderr, "Insufficient work for DGEQRF.\n");

  double alpha = 1., beta = 0.;
  int Lwork_i;
  cublasDgeam(s, CUBLAS_OP_N, CUBLAS_OP_N, (int)m, (int)Ltau, &alpha, a, (int)lda, &beta, q, (int)ldq, q, (int)ldq);
  cusolverDnDorgqr_bufferSize(s, (int)m, (int)nq, (int)Ltau, q, (int)ldq, tau, &Lwork_i);
  if ((size_t)Lwork_i <= Lwork)
    cusolverDnDorgqr(s, (int)m, (int)nq, (int)Ltau, q, (int)ldq, tau, tau + Ltau, (int)Lwork, s);
  else
    fprintf(stderr, "Insufficient work for DORGQR.\n");
}

void Hatrix::gpu::dorth(Stream& s, double* q, int64_t m, int64_t n, int64_t ldq) {
  size_t workspaceInBytesOnDevice_geqrf, workspaceInBytesOnHost_geqrf;
  double* tau = (double*)(void*)s;
  size_t Ltau = m > n ? n : m, Lwork = (size_t)s - Ltau * sizeof(double);
  cusolverDnXgeqrf_bufferSize(s, s, m, n, CUDA_R_64F, q, ldq, CUDA_R_64F, tau, CUDA_R_64F, &workspaceInBytesOnDevice_geqrf, &workspaceInBytesOnHost_geqrf);
  if (workspaceInBytesOnDevice_geqrf <= Lwork && workspaceInBytesOnHost_geqrf <= s.Lwork_host)
    cusolverDnXgeqrf(s, s, m, n, CUDA_R_64F, q, ldq, CUDA_R_64F, tau, CUDA_R_64F, tau + Ltau, Lwork, s.Work_host, s.Lwork_host, s);
  else
    fprintf(stderr, "Insufficient work for DGEQRF.\n");

  int Lwork_i;
  cusolverDnDorgqr_bufferSize(s, (int)m, (int)n, (int)Ltau, q, (int)ldq, tau, &Lwork_i);
  if ((size_t)Lwork_i <= Lwork)
    cusolverDnDorgqr(s, (int)m, (int)n, (int)Ltau, q, (int)ldq, tau, tau + Ltau, (int)Lwork, s);
  else
    fprintf(stderr, "Insufficient work for DORGQR.\n");
}

void Hatrix::gpu::dgesvdr(Stream& s, char jobu, char jobv, int64_t m, int64_t n, int64_t rank, int64_t p, int64_t iters, double* A, int64_t lda,
  double* S, double* U, int64_t ldu, double* V, int64_t ldv) {
  jobu = jobu == 'S' ? 'S' : 'N';
  jobv = jobv == 'S' ? 'S' : 'N';

  rank = rank > m ? m : rank;
  rank = rank > n ? n : rank;
  p = p < 0 ? 0 : p;
  p = p + rank > m ? m - rank : p;
  p = p + rank > n ? n - rank : p;

  size_t workspaceInBytesOnDevice_gesvdr, workspaceInBytesOnHost_gesvdr;
  cusolverDnXgesvdr_bufferSize(s, s, jobu, jobv, m, n, rank, p, iters, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S,
    CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, &workspaceInBytesOnDevice_gesvdr, &workspaceInBytesOnHost_gesvdr);

  if (workspaceInBytesOnDevice_gesvdr <= (size_t)s && workspaceInBytesOnHost_gesvdr <= s.Lwork_host)
    cusolverDnXgesvdr(s, s, jobu, jobv, m, n, rank, p, iters, CUDA_R_64F, (void*)A, lda, CUDA_R_64F, S, 
      CUDA_R_64F, U, ldu, CUDA_R_64F, V, ldv, CUDA_R_64F, s, (size_t)s, s.Work_host, s.Lwork_host, s);
  else
    fprintf(stderr, "Insufficient work for DGESVDR.\n");
}
