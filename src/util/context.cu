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

  class Stream {
  public:
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cusolverDnParams_t cusolverParams;
    curandGenerator_t curandH;
  
    void* Workspace;
    size_t Lwork;
  
    void* Work_host;
    size_t Lwork_host;
    int* info;
  
    cudaEvent_t ev1, ev2;
  
    Stream(size_t Lwork, size_t Lwork_host) {
      Stream::Lwork = Lwork;
      Stream::Lwork_host = Lwork_host;
  
      cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
      cublasCreate(&cublasH);
      cublasSetStream(cublasH, stream);
      cudaMalloc(reinterpret_cast<void**>(&Workspace), Lwork);
      cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int));
      Work_host = (Lwork_host > 0) ? (void*)malloc(Lwork_host) : nullptr;
  
      //cublasSetWorkspace(cublasH, Workspace, Lwork);
      cusolverDnCreate(&cusolverH);
      cusolverDnSetStream(cusolverH, stream);
      cusolverDnCreateParams(&cusolverParams);
  
      curandCreateGenerator(&curandH, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetStream(curandH, stream);
  
      ev1 = nullptr;
      ev2 = nullptr;
  
    }
  
    ~Stream() {
      if (info)
        cudaFree(info);
      if (Workspace)
        cudaFree(Workspace);
      if (Work_host)
        free(Work_host);
      if (cusolverParams)
        cusolverDnDestroyParams(cusolverParams);
      if (cusolverH)
        cusolverDnDestroy(cusolverH);
      if (cublasH)
        cublasDestroy(cublasH);
      if (stream)
        cudaStreamDestroy(stream);
      if (curandH)
        curandDestroyGenerator(curandH);
      if (ev1)
        cudaEventDestroy(ev1);
      if (ev2)
        cudaEventDestroy(ev2);
    }
  
    void sync() const {
      cudaStreamSynchronize(stream);
    }
  
    void createEventOne() {
      if (ev1)
        cudaEventDestroy(ev1);
      cudaEventCreate(&ev1);
      cudaEventRecord(ev1, stream);
    }
  
    void createEventTwo() {
      if (ev2)
        cudaEventDestroy(ev2);
      cudaEventCreate(&ev2);
      cudaEventRecord(ev2, stream);
    }
  
    float getElapsedTime() const {
      if (ev1 && ev2) {
        float ms;
        cudaEventElapsedTime(&ms, ev1, ev2);
        return ms;
      }
      return 0.;
    }
  
    int Info() const {
      int i;
      cudaMemcpy(&i, info, sizeof(int), cudaMemcpyDeviceToHost);
      return i;
    }
  
  };
  
  Stream** lib = nullptr;
  int n_lib = 0;
  mode_t lib_mode = mode_t::SERIAL;
  int lib_ptr = 0;
  
  void init(int nstream, size_t Lwork, size_t Lwork_host) {
    term();
    lib = new Stream * [nstream];
    for (int i = 0; i < nstream; i++)
      lib[i] = new Stream(Lwork, Lwork_host);
    n_lib = nstream;
  }
  
  void term() {
    if (lib) {
      for (int i = 0; i < n_lib; i++)
        delete lib[i];
      delete[] lib;
      lib = nullptr;
    }
    n_lib = 0;
    lib_ptr = 0;
  }
  
  void sync(int stream) {
    if (stream == -1)
      cudaDeviceSynchronize();
    else if (stream < n_lib)
      lib[stream]->sync();
  }
  
  mode_t parallel_mode(mode_t mode) {
    mode_t old = lib_mode;
    lib_mode = mode;
    return old;
  }
  
  int run() {
    int old = lib_ptr;
    lib_ptr += static_cast<int>(lib_mode);
    lib_ptr = lib_ptr >= n_lib ? 0 : lib_ptr;
    return old;
  }
  
  void runtime_args(void** args, arg_t type) {
    if (lib == nullptr)
    { fprintf(stderr, "Lib is not initialized.\n"); assert(0); return; }
    Stream& s = *lib[run()];
    switch (type) {
    case arg_t::STREAM:
      args[0] = s.stream; break;
    case arg_t::BLAS:
      args[0] = s.cublasH; break;
    case arg_t::SOLV:
      args[0] = s.cusolverH;
      args[1] = s.cusolverParams;
      args[2] = s.Workspace;
      args[3] = &s.Lwork;
      args[4] = s.Work_host;
      args[5] = &s.Lwork_host;
      args[6] = s.info;
      break;
    case arg_t::BLAS_SOL:
      args[0] = s.cublasH;
      args[1] = s.cusolverH;
      args[2] = s.cusolverParams;
      args[3] = s.Workspace;
      args[4] = &s.Lwork;
      args[5] = s.Work_host;
      args[6] = &s.Lwork_host;
      args[7] = s.info;
      break;
    case arg_t::RAND:
      args[0] = s.curandH; break;
    default:
      break;
    }
  }
  
  void generator_seed(long long unsigned int seed) {
    for (int i = 0; i < n_lib; i++)
      curandSetPseudoRandomGeneratorSeed(lib[i]->curandH, seed);
  }
  
  void time_start(int stream) {
    if (lib == nullptr)
    { fprintf(stderr, "Lib is not initialized.\n"); assert(0); return; }
    stream = stream == -1 ? lib_ptr : stream;
    if (stream < n_lib)
      lib[stream]->createEventOne();
  }
  
  void time_end(int stream) {
    if (lib == nullptr)
    { fprintf(stderr, "Lib is not initialized.\n"); assert(0); return; }
    stream = stream == -1 ? lib_ptr : stream;
    if (stream < n_lib)
      lib[stream]->createEventTwo();
  }
  
  float get_time(int stream) {
    if (lib == nullptr)
    { fprintf(stderr, "Lib is not initialized.\n"); assert(0); return 0.; }
    stream = stream == -1 ? lib_ptr : stream;
    if (stream < n_lib)
      return lib[stream]->getElapsedTime();
    return 0.;
  }
  
  


}  // namespace Hatrix
