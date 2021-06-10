#pragma once

#include <cstddef>
#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#endif

namespace Hatrix {

    constexpr size_t DEFAULT_LWORK = 0x800000ULL;
    constexpr size_t DEFAULT_LWORK_HOST = 0x40000ULL;

    class Context {
    private:
      Context() {};

    public:
#ifdef __CUDACC__
      static size_t nStreams;
      static size_t workspaceInBytesOnDevice;
      static size_t workspaceInBytesOnHost;

      static cudaStream_t* stream;
      static cublasHandle_t* cublasH;
      static cusolverDnHandle_t* cusolverH;
      static cusolverDnParams_t* cusolverParams;
      static curandGenerator_t* curandH;

      static void** bufferOnDevice;
      static void** bufferOnHost;
      static int* info;
      static size_t sid;
      static bool forking;
#endif

      static void init(int argc = 0, const char** argv = nullptr);

      static void finalize();

      static void join();

      static void fork();

      static void critical();

      static void iterate();

    };

}  // namespace Hatrix
