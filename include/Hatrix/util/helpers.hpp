#pragma once

#include <vector>

namespace Hatrix {
  // Generate a vector of N equally spaced 1D points starting from minVal upto
  // maxVal.
  //
  // @param endpoint If endpoint is true, the distance is calculated using
  // without considering the last point in the sequence.
  std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal,
                                            bool endpoint=true);

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
