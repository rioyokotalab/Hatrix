#pragma once

#include <vector>

namespace Hatrix {
  std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal, bool endpoint=true);

  // Concat two matrices along the given axis.
  //
  // If axis=0 then stick the rows together, which results in a matrix
  // of size (A.rows + B.rows, A.cols). Both matrices must have same
  // number of cols.
  //
  // If axis=1 then stick the cols together, which results in a matrix
  // of size (A.rows, A.cols + B.cols). Both matrices must have same
  // number of rows.
  Matrix concat(const Matrix& A, const Matrix& B, const int axis);
}
