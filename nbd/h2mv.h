
#pragma once
#include "nbd.h"

namespace nbd {

  void upwardPass(const Cell* jcell, const Matrix* base, const real_t* x, Matrix* m);

  void horizontalPass(const Cells& icells, const Cells& jcells, const Matrices& d, const Matrices& m, Matrices& l);

  void downwardPass(const Cell* icell, const Matrix* base, Matrix* l, real_t* b);

  void closeQuarter(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& d, const real_t* x, real_t* b);

  void h2mv_complete(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* x, real_t* b);

}
