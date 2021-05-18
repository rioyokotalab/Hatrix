#include "Hatrix/util/error_checking.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/lapack.h"

namespace Hatrix {

double norm_diff(const Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  return norm(A - B);
}

}  // namespace Hatrix
