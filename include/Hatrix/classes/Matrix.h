#pragma once
#include <cstdint>
using std::int64_t;


namespace Hatrix {

class Matrix {
 public:
  double* data_ = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;

  Matrix() = delete;

  ~Matrix();

  Matrix(int64_t rows, int64_t cols);

  Matrix(const Matrix& A);

  Matrix& operator=(const Matrix& A);

  Matrix(Matrix&& other);

  Matrix& operator=(Matrix&& A);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(int64_t i, int64_t j);
  const double& operator()(int64_t i, int64_t j) const;

  void shrink(int64_t rows, int64_t cols);

  int64_t min_dim() const;
  int64_t max_dim() const;

  void print() const;
};

} // namespace Hatrix
