
#pragma once
#ifndef _Handle
#define _Handle

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace Hatrix {
  namespace gpu {

    constexpr size_t DEFAULT_LWORK = 0x01000000;
    constexpr size_t DEFAULT_LWORK_HOST = 0x00001000;

    class Stream {
    private:
      cudaStream_t stream;
      cublasHandle_t cublasH;
      cusolverDnHandle_t cusolverH;
      cusolverDnParams_t cusolverParams;
      void* Workspace;
      size_t Lwork;
      int* info;

    public:
      void* Work_host;
      size_t Lwork_host;

      Stream(size_t Lwork = DEFAULT_LWORK, size_t Lwork_host = DEFAULT_LWORK_HOST);

      ~Stream();

      void sync() const;

      operator cudaStream_t();

      operator cublasHandle_t();

      operator cusolverDnHandle_t();

      operator cusolverDnParams_t();

      operator double* ();

      operator float* ();

      operator void* ();

      operator size_t();

      operator int();

      operator int* ();

    };

  }
}

#endif
