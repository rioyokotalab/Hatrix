#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/classes/arithmetics.h"

#include <cassert>
#include <cstdint>
using std::int64_t;

namespace Hatrix {

void gesv_IR(Matrix &A, Matrix &b, int64_t max_iter){
  assert(A.rows == b.rows);

  int64_t mdim = A.min_dim();
  Matrix L(A.rows, mdim), U(mdim, A.cols), P(A.rows, A.rows);
  lup(A, L, U, P);
  Matrix x = P * b;
  solve_triangular(L, x, Left, Lower, true);
  solve_triangular(U, x, Right, Upper, false);
  double res = b - (A * x);
}

} // namespace Hatrix