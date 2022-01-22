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
    if (A.numel() == 0) {
      return Matrix(B);
    }
    if (B.numel() == 0) {
      return Matrix(A);
    }
    if (axis == 0) {
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
    else {
      std::cout << "concat axis must be 0 or 1, not axis=" << axis << std::endl;
      abort();
    }
  }
}
