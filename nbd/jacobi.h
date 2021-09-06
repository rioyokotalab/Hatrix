
#pragma once

#include "nbd.h"

namespace nbd {
  
  int h2solve(int max_iters, real_t epi, EvalFunc ef, const Cells& cells, int dim, const Matrices& base, const Matrices& d, real_t* x);

}
