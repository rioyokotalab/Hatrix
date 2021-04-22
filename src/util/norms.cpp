#include "Hatrix/util/norms.h"

#include "Hatrix/classes/Matrix.h"

#include <cassert>
#include <cstdint>
using std::int64_t;


namespace Hatrix {

double frobenius_norm_diff(Matrix& A, Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  double norm_diff = 0;
  for (int64_t i=0; i<A.rows; ++i) {
    for (int64_t j=0; j<A.cols; ++j) {
      norm_diff += (A(i, j) - B(i, j)) * (A(i, j) - B(i, j));
    }
  }
  return norm_diff;
}

} // namespace Hatrix
