#include "Hatrix/util/handle.h"

#include "Hatrix/classes/Matrix.h"


namespace Hatrix {

  cublasHandle_t blasH = nullptr;
  cusolverDnHandle_t solvH = nullptr;

  void init() {
    cublasCreate(&blasH);
    cusolverDnCreate(&solvH);
  }

  void terminate() {
    cublasDestroy(blasH); blasH = 0;
    cusolverDnDestroy(solvH); blasH = 0;
  }


} // namespace Hatrix
