#include "Hatrix/util/matrix_generators.h"

#include <vector>
#include <cassert>
#include <iostream>

namespace Hatrix {
  std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
    std::vector<double> res(N, 0.0);
    double rnge = maxVal - minVal;
    for(int i=0; i<N; i++) {
      res[i] = minVal + ((double)i/(double)rnge);
    }
    return res;
  }

  Matrix concat(const Matrix& A, const Matrix& B, const int axis) {
    if (axis == 0) {
      std::cout << "A.cols= " << A.cols << " B.cols= " << B.cols << std::endl;
      assert(A.cols == B.cols);
      Matrix matrix(A.rows + B.rows, A.cols);
      auto matrix_splits = matrix.split(std::vector<int64_t>(1, A.rows), {});
      matrix_splits[0] = A;
      matrix_splits[1] = B;

      return matrix;
    }
    else if (axis == 1) {
      assert(A.rows == B.rows);
      Matrix matrix(A.rows, A.cols + B.cols);
      auto matrix_splits = matrix.split({}, std::vector<int64_t>(1, A.cols));
      matrix_splits[0] = A;
      matrix_splits[1] = B;

      return matrix;
    }
  }
}
