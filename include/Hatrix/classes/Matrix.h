#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

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

  // Split the matrix into n_row_splits * n_col_splits blocks.
  // n_row_splits is the number of blocks in the row dimension and
  // n_col_splits is the number of blocks in the column dimension.
  //
  // All blocks except the last block will be subdivided into equal parts
  // (last block will be different if split dimension is not an exact dimension
  // of the total dimension of the matrix).
  //
  // Example:
  // Matrix A(100, 100);
  // std::vector<Matrix> splits = A.split(10, 10);
  // // splits is now a vector of size 100 containing 100 Matrix objects
  // // of size 10x10 each.
  std::vector<Matrix> split(int64_t n_row_splits, int64_t n_col_splits,
                            bool copy = false) const;

  std::vector<Matrix> split(const std::vector<int64_t>& row_split_indices,
                            const std::vector<int64_t>& col_split_indices,
                            bool copy = false) const;

  int64_t min_dim() const;
  int64_t max_dim() const;

  void print() const;

  size_t memory_used() const;

  size_t shared_memory_used() const;

  Matrix transpose() const;
};

}  // namespace Hatrix
