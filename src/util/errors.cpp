#include "Hatrix/util/errors.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/util/norms.h"
#include "Hatrix/functions/arithmetics.h"

#include <cassert>
#include <cstdint>
using std::int64_t;
#include <cmath>


namespace Hatrix {

double norm_bw_error(const Matrix& residual, const Matrix& A, const Matrix& x, const Matrix& b) {
  assert (residual.cols == 1 && x.cols == 1 && b.cols == 1);
  assert (A.rows == residual.rows && A.rows == b.rows);
  assert (A.cols == x.rows);

  double error = infinity_norm(residual);
  error = error / (infinity_norm(A) * infinity_norm(x) + infinity_norm(b));

  return error;
}

double comp_bw_error(const Matrix& residual, const Matrix& A, const Matrix& x, const Matrix& b) {
  assert (residual.cols == 1 && x.cols == 1 && b.cols == 1);
  assert (A.rows == residual.rows && A.rows == b.rows);
  assert (A.cols == x.rows);

  Matrix temp = abs(A) * abs(x);

  double comp_error, error = 0;
  for (int64_t i=0; i<residual.rows; ++i){
    comp_error = std::abs(residual(i, 0)) / (temp(i, 0) + std::abs(b(i, 0)));
    if (comp_error > error)
      error = comp_error;
  }

  return error;
}

} // namespace Hatrix
