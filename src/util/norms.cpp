#include "Hatrix/util/norms.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/lapack.h"

namespace Hatrix {

double frobenius_norm(const Matrix& A) {
  return calc_norm(A, Norm::FrobeniusNorm);
}

double one_norm(const Matrix& A) { return calc_norm(A, Norm::OneNorm); }

double max_norm(const Matrix& A) { return calc_norm(A, Norm::MaxNorm); }

double infinity_norm(const Matrix& A) {
  return calc_norm(A, Norm::InfinityNorm);
}

double frobenius_norm_diff(const Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  return frobenius_norm(A - B);
}

}  // namespace Hatrix
