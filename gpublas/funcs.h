#pragma once
#ifndef _Func
#define _Func

#include <handle.h>

namespace shimo {
  namespace gpu {

    void dalloc(Stream& s, double** a_ptr, int m, int n, int lda, int value = 0);

    void dcopy2D(Stream& s, cudaMemcpyKind kind, int m, int n, double* dst, int ldd, const double* src, int lds = 0);

    void dpotrf(Stream& s, cublasFillMode_t uplo, double* a, int64_t n, int64_t lda);

    void dgetrf(Stream& s, double* a, int64_t m, int64_t n, int64_t lda, int64_t* ipiv = nullptr);

    void dorth(Stream& s, double* q, double* a, int64_t m, int64_t nq, int64_t na, int64_t ldq, int64_t lda);

    void dorth(Stream& s, double* q, int64_t m, int64_t n, int64_t ldq);

    void dgesvdr(Stream& s, char jobu, char jobv, int64_t m, int64_t n, int64_t rank, int64_t p, int64_t iters, double* A, int64_t lda,
      double* S, double* U, int64_t ldu, double* V, int64_t ldv);

  }
}

#endif