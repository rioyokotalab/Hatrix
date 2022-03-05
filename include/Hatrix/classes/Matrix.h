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
  // Shows whether a Matrix is a view of an object or the actual copy.
  bool is_view = false;

 private:
  class DataHandler;
  std::shared_ptr<DataHandler> data;
  // data_ptr is a pointer to the memory within data. This is done
  // for easily tracking the location to an offset of data if this Matrix
  // is a view of another matrix.
  double* data_ptr = nullptr;

 public:
  Matrix() = default;

  ~Matrix() = default;

  Matrix(const Matrix& A);

  Matrix& operator=(const Matrix& A);

  Matrix& operator=(Matrix&& A);

  Matrix(Matrix&& A) = default;

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
  // The vector returns the slices stored in row major order.
  //
  // Example:
  //
  // Matrix A(100, 100);
  // std::vector<Matrix> splits = A.split(4, 4);
  // // splits is now a vector of size 16 containing 16 Matrix objects
  // // of size 25x25 each.
  //
  // Results in:
  //       ____25____50______75_____100
  //      |     |     |      |      |
  //  25  |_____|_____|______|______|
  //      |     |     |      |      |
  //  50  |_____L_____L______|______|
  //      |     |     |      |      |
  //  75  |_____L_____L______|______|
  //      |     |     |      |      |
  //  100 |_____L_____L______|______|
  std::vector<Matrix> split(int64_t n_row_splits, int64_t n_col_splits,
                            bool copy = false) const;


  // Split the matrix along the dimensions specifed in each vector
  // in row_split_indices and col_split_indices. This function will
  // return as many tiles that can be generated from the indices supplied.
  // The vector returns the slices stored in row major order.
  //
  // Example:
  // Matrix A(100, 100);
  // std::vector<Matrix> split = A.split({75, 90}, {60, 80});
  //
  // Results in:
  // Example:
  // Matrix A(100, 100);
  // std::vector<Matrix> split = A.split({30, 90}, {60, 80});
  //
  // Results in:
  //       __________________60___80____100
  //      |                  |    |    |
  //      |                  |    |    |
  //      |__________________|____|____|
  // 30   |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |__________________L____L____|
  // 90   |__________________L____L____|
  // 100
  std::vector<Matrix> split(const std::vector<int64_t>& row_split_indices,
                            const std::vector<int64_t>& col_split_indices,
                            bool copy = false) const;

  int64_t min_dim() const;
  int64_t max_dim() const;
  int64_t numel() const;

  void print() const;

  void print_meta() const;

  void read_file(std::string in_file);

  void out_file(std::string out_file) const;

  // Get the size of the memory used by this matrix. If this is a view,
  // this function returns only the memory consumed by the view.
  size_t memory_used() const;

  // Get the size of the memory that is occupied by the whole matrix.
  // If this matrix is part of a view, it will return the memory used
  // by the whole matrix, not just the view.
  size_t shared_memory_used() const;

  Matrix block_ranks(int64_t nblocks, double accuracy) const;
};

}  // namespace Hatrix
