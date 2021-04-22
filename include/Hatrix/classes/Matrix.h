#pragma once
#include <cstdint>
using std::uint64_t;


namespace Hatrix {

class Matrix {
 public:
  double* data_;
  uint64_t rows, cols;

  Matrix() = delete;

  ~Matrix();

  Matrix(uint64_t rows, uint64_t cols);

  Matrix(const Matrix& A);

  Matrix& operator=(const Matrix& A);

  Matrix(Matrix&& other);

  Matrix& operator=(Matrix&& A);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(uint64_t i, uint64_t j);
  const double& operator()(uint64_t i, uint64_t j) const;

  void shrink(uint64_t rows, uint64_t cols);
  uint64_t min_dim();
};

} // namespace Hatrix
