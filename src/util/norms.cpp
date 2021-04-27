#include "Hatrix/util/norms.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"

#include <cassert>
#include <cstdint>
using std::int64_t;
#include <cstdlib>
#include <cmath>
#include <iostream>


namespace Hatrix {

double frobenius_norm(const Matrix& A) {
  double norm = 0;
  for (int64_t i=0; i<A.rows; ++i) {
    for (int64_t j=0; j<A.cols; ++j) {
      norm += A(i, j) * A(i, j);
    }
  }
  return std::sqrt(norm);
}

double frobenius_norm_diff(const Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  return frobenius_norm(A - B);
}

} // namespace Hatrix
