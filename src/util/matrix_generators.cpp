#include "Hatrix/util/matrix_generators.h"

#include <cmath>
#include <cstdint>
using std::int64_t;
#include <random>


namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Matrix out(rows, cols);
  for (int64_t i=0; i<rows; ++i) {
    for (int64_t j=0; j<cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols) {
  // TODO: Might want more sophisticated method, specify rate of decay of
  // singular values etc...
  Matrix out(rows, cols);
  for (int64_t i=0; i<rows; ++i) {
    for (int64_t j=0; j<cols; ++j) {
      out(i, j) = 1.0 / std::abs(i - j + out.max_dim());
    }
  }
  return out;
}

Matrix generate_identity_matrix(int64_t rows, int64_t cols) {
  Matrix out(rows, cols);
  for (int64_t i=0; i<out.min_dim(); ++i) {
    out(i, i) = 1.;
  }
  return out;
}

} // namespace Hatrix
