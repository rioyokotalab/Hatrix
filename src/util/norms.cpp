#include "Hatrix/util/norms.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/lapack.h"

#include <cassert>
#include <cstdint>
using std::int64_t;
#include <cstdlib>
#include <cmath>
#include <iostream>


namespace Hatrix {

double frobenius_norm(const Matrix& A) {
  return norm(A, FrobeniusNorm);
}

double one_norm(const Matrix& A) {
  return norm(A, OneNorm);
}

double max_norm(const Matrix& A) {
  return norm(A, MaxNorm);
}

double infinity_norm(const Matrix& A) {
  return norm(A, InfinityNorm);
}

double frobenius_norm_diff(const Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  return frobenius_norm(A - B);
}

} // namespace Hatrix
