#pragma once
#include <cstdint>
#include <memory>

namespace Hatrix {

class Matrix {
 public:
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t stride = 0;

 private:
  class DataHandler;
  std::shared_ptr<DataHandler> data;
  double* data_ptr = nullptr;

 public:
  Matrix() = default;

  ~Matrix() = default;

  Matrix(const Matrix& A);

  Matrix& operator=(const Matrix& A);

  Matrix(Matrix&& A) = default;

  Matrix& operator=(Matrix&& A) = default;

  Matrix(int64_t rows, int64_t cols);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(int64_t i, int64_t j);
  const double& operator()(int64_t i, int64_t j) const;

  void shrink(int64_t rows, int64_t cols);

  int64_t min_dim() const;
  int64_t max_dim() const;

  void print() const;

  void fetchCPU();
};

}  // namespace Hatrix
