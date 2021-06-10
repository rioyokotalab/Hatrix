#include "Hatrix/util/context.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "curand.h"

namespace Hatrix {
  
  size_t Context::nStreams = 0;
  size_t Context::workspaceInBytesOnDevice = 0;
  size_t Context::workspaceInBytesOnHost = 0;
  
  cudaStream_t* Context::stream = nullptr;
  cublasHandle_t* Context::cublasH = nullptr;
  cusolverDnHandle_t* Context::cusolverH = nullptr;
  cusolverDnParams_t* Context::cusolverParams = nullptr;
  curandGenerator_t* Context::curandH = nullptr;
  
  void** Context::bufferOnDevice = nullptr;
  void** Context::bufferOnHost = nullptr;
  int* Context::info = nullptr;
  size_t Context::sid = 0;
  bool Context::forking = false;
  
  void Context::init(int argc, const char** argv) {
    if (nStreams > 0)
      Context::finalize();
  
    Context::nStreams = (size_t)(argc > 1 ? strtoull(argv[1], nullptr, 0) : 1);
    Context::workspaceInBytesOnDevice = (size_t)(argc > 2 ? strtoull(argv[2], nullptr, 0) : DEFAULT_LWORK);
    Context::workspaceInBytesOnHost = (size_t)(argc > 3 ? strtoull(argv[3], nullptr, 0) : DEFAULT_LWORK_HOST);
  
    Context::stream = new cudaStream_t [Context::nStreams];
    Context::cublasH = new cublasHandle_t [Context::nStreams];
    Context::cusolverH = new cusolverDnHandle_t [Context::nStreams];
    Context::cusolverParams = new cusolverDnParams_t [Context::nStreams];
    Context::curandH = new curandGenerator_t [Context::nStreams];
  
    Context::bufferOnHost = new void* [Context::nStreams];
    Context::bufferOnDevice = new void* [Context::nStreams];
  
    for (size_t i = 0; i < Context::nStreams; i++) {
      cudaStreamCreateWithFlags(Context::stream + i, cudaStreamDefault);
      cublasCreate(Context::cublasH + i);
      cublasSetStream(Context::cublasH[i], Context::stream[i]);
  
      if (Context::workspaceInBytesOnDevice)
        cudaMalloc(reinterpret_cast<void**>(Context::bufferOnDevice + i), Context::workspaceInBytesOnDevice);
      if (Context::workspaceInBytesOnHost)
        Context::bufferOnHost[i] = (void*)malloc(Context::workspaceInBytesOnHost);
  
      cusolverDnCreate(Context::cusolverH + i);
      cusolverDnSetStream(Context::cusolverH[i], Context::stream[i]);
      cusolverDnCreateParams(Context::cusolverParams + i);
  
      curandCreateGenerator(Context::curandH + i, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetStream(Context::curandH[i], Context::stream[i]);
    }
  
    cudaMalloc(reinterpret_cast<void**>(&Context::info), Context::nStreams * sizeof(int));
    Context::sid = 0;
    Context::forking = false;
  }
  
  void Context::finalize() {
    for (size_t i = 0; i < Context::nStreams; i++) {
      if (Context::stream)
        cudaStreamDestroy(Context::stream[i]);
      if (Context::cublasH)
        cublasDestroy(Context::cublasH[i]);
      if (Context::cusolverH)
        cusolverDnDestroy(Context::cusolverH[i]);
      if (Context::cusolverParams[i])
        cusolverDnDestroyParams(Context::cusolverParams[i]);
      if (Context::curandH)
        curandDestroyGenerator(Context::curandH[i]);
      if (Context::bufferOnDevice[i])
        cudaFree(Context::bufferOnDevice[i]);
      if (Context::bufferOnHost[i])
        free(Context::bufferOnHost[i]);
    }
  
    Context::nStreams = 0;
    Context::workspaceInBytesOnDevice = 0;
    Context::workspaceInBytesOnHost = 0;
  
    if (Context::stream)
      delete[] Context::stream;
    if (Context::cublasH)
      delete[] Context::cublasH;
    if (Context::cusolverH)
      delete[] Context::cusolverH;
    if (Context::cusolverParams)
      delete[] Context::cusolverParams;
    if (Context::curandH)
      delete[] Context::curandH;
  
    Context::stream = nullptr;
    Context::cublasH = nullptr;
    Context::cusolverH = nullptr;
    Context::cusolverParams = nullptr;
    Context::curandH = nullptr;
  
    if (Context::bufferOnDevice)
      delete[] Context::bufferOnDevice;
    if (Context::bufferOnHost)
      delete[] Context::bufferOnHost;
    if (Context::info)
      cudaFree(Context::info);
  
    bufferOnDevice = nullptr;
    bufferOnHost = nullptr;
    info = nullptr;
  }
  
  void Context::join() {
    Context::sid = 0;
    Context::forking = false;
    cudaDeviceSynchronize();
  }
  
  void Context::fork() {
    Context::forking = true;
  }

  void Context::critical() {
    Context::forking = false;
  }
  
  void Context::iterate() {
    if (Context::forking)
      Context::sid = Context::sid == Context::nStreams - 1 ? 0 : Context::sid + 1;
  }
  

}  // namespace Hatrix
