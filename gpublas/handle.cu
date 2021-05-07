
#include <handle.h>

#include <cstdlib>
#include <cuda_runtime_api.h>

using namespace Hatrix::gpu;

Stream::Stream(size_t Lwork, size_t Lwork_host) {
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
}

Stream::~Stream() {
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
}

void Stream::sync() const {
  cudaStreamSynchronize(stream);
}

Stream::operator cudaStream_t() {
  return stream;
}

Stream::operator cublasHandle_t() {
  return cublasH;
}

Stream::operator cusolverDnHandle_t() {
  return cusolverH;
}

Stream::operator cusolverDnParams_t() {
  return cusolverParams;
}

Stream::operator double*() {
  return (double*)Workspace;
}

Stream::operator float* () {
  return (float*)Workspace;
}

Stream::operator void* () {
  return (void*)Workspace;
}

Stream::operator size_t() {
  return Lwork;
}

Stream::operator int() {
  int i;
  cudaMemcpy(&i, info, sizeof(int), cudaMemcpyDeviceToHost);
  return i;
}

Stream::operator int* () {
  return info;
}
