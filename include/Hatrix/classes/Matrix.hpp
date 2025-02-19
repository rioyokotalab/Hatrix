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

 public:
  // Not using a shared_ptr here since it is not capable of handling a
  // dynamic heap-allocated array.
  // https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used
  // data_ptr is a pointer to the memory within data. This is done
  // for easily tracking the location to an offset of data if this Matrix
  // is a view of another matrix.
  double* data_ptr = nullptr;

  Matrix();

  ~Matrix();

  Matrix(const Matrix& A);

  // Copy constructor for Matrix. Create a view object by default. The reason
  // why this is done is mainly to accomodate std::vector#push_back or #emplace_back
  // style functions which call the default copy constructor after they call the
  // move constructor.
  // https://stackoverflow.com/questions/40457302/c-vector-emplace-back-calls-copy-constructor
  Matrix(const Matrix& A, bool copy);

  Matrix& operator=(const Matrix& A);

  Matrix& operator=(Matrix&& A);

  Matrix(Matrix&& A);

  Matrix(int64_t rows, int64_t cols);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(int64_t i, int64_t j);
  const double& operator()(int64_t i, int64_t j) const;

  // WARNING: does not deallocate the extra data!
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

  // Split the matrix of size (rows, cols) into nblocksxnblocks where each
  // block is of the same size. Returns a nblocksxnblocks matrix that contains
  // the rank obtained from an SVD of each individual block.
  Matrix block_ranks(int64_t nblocks, double accuracy) const;

  // Swap the i'th row with the row_indices[i]'th row of the matrix.
  Matrix swap_rows(const std::vector<int64_t>& row_indices);

  // Swap the j'th column with the col_indicies[j]'th column of the matrix.
  Matrix swap_cols(const std::vector<int64_t>& col_indices);

  // Destructively resize. Does not preserve the data of the matrix.
  void destructive_resize(const int64_t nrows, const int64_t ncols);

  // Return the lower triangular matrix after `diag` diagonals.
  // For example, passing diag=-1 will return the strict lower triangular
  // matrix, and diag=0 will return the lower triangular matrix with the
  // diagonal included.
  Matrix tril(const int64_t diag);

  // Pad an extra pad_width_rows rows, and pad_width_cols columns on the matrix.
  Matrix pad(const int64_t pad_width_rows, const int64_t pad_width_cols) const;
};

}  // namespace Hatrix
