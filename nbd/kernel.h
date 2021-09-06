
#pragma once
#include "nbd.h"

namespace nbd {

  EvalFunc r2();

  EvalFunc l2d();

  EvalFunc l3d();

  void eval(EvalFunc ef, const Body* bi, const Body* bj, int dim, real_t* out);

  void mvec_kernel(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, real_t alpha, const real_t* x_vec, int incx, real_t beta, real_t* b_vec, int incb);

  void P2Pnear(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, Matrix& a);

  void P2Pfar(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, Matrix& a, int rank);

  void SampleP2Pi(Matrix& s, const Matrix& a);

  void SampleP2Pj(Matrix& s, const Matrix& a);

  void SampleParent(Matrix& sc, const Matrix& sp, int c_off);

  void BasisOrth(Matrix& s);

  void BasisInvLeft(const Matrix& s, Matrix& a);

  void BasisInvRight(const Matrix& s, Matrix& a);

  void BasisInvMultipleLeft(const Matrix* s, int ls, Matrix& a);

  void MergeS(Matrix& a);

}