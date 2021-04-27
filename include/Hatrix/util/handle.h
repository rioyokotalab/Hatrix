
#ifndef _handle_h
#define _handle_h

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace Hatrix {
  extern cublasHandle_t blasH;
  extern cusolverDnHandle_t solvH;

  void init();
  void terminate();
  void sync();

}

#endif