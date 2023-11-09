#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

template <typename DT = double>
class LowRank {
 public:
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rank = 0;
  DT error = 0.0;
  Matrix<DT> U, S, V;

  // Shows whether a Matrix is a view of an object or the actual copy.
  // TODO can a LowRank matrix be a view?
  //bool is_view = false;

 

 public:
  LowRank() = default;

  ~LowRank() = default;

  LowRank(const LowRank& A) = default;
  //TODO why is this not done with overloading?
  // Copy constructor for Matrix. Create a view object by default. The reason
  // why this is done is mainly to accomodate std::vector#push_back or #emplace_back
  // style functions which call the default copy constructor after they call the
  // move constructor.
  // https://stackoverflow.com/questions/40457302/c-vector-emplace-back-calls-copy-constructor
  //Matrix(const Matrix& A, bool copy);

  LowRank(const Matrix<DT>& U, const Matrix<DT>& S, const Matrix<DT>& V, bool copy=false);

  // explicit copy intended for type conversion
  template <typename OT>
  LowRank(const LowRank<OT>& A);

  LowRank(Matrix<DT>&& U, Matrix<DT>&& S, Matrix<DT>&& V);

  LowRank& operator=(const LowRank& A) = default;

  LowRank& operator=(LowRank&& A) = default;

  LowRank(LowRank&& A) = default;

  LowRank(const Matrix<DT>& A, int64_t rank);

  const DT& operator()(int64_t i, int64_t j) const;

  Matrix<DT> make_dense() const;

  void print() const;

  void print_approx() const;

  int64_t get_rank(DT error) const;

  // WARNING: does not deallocate the extra data!
  // TODO might be useful
  //void shrink(int64_t rows, int64_t cols);

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
  //std::vector<Matrix> split(int64_t n_row_splits, int64_t n_col_splits,
   //                         bool copy = false) const;


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
  //std::vector<Matrix> split(const std::vector<int64_t>& row_split_indices,
   //                         const std::vector<int64_t>& col_split_indices,
     //                       bool copy = false) const;

  //int64_t min_dim() const;
  //int64_t max_dim() const;
  //int64_t numel() const;

  //void print() const;

  //void print_meta() const;

  //void read_file(std::string in_file);

  //void out_file(std::string out_file) const;

  // Get the size of the memory used by this matrix. If this is a view,
  // this function returns only the memory consumed by the view.
  //size_t memory_used() const;

  //Matrix block_ranks(int64_t nblocks, double accuracy) const;

  //Matrix swap_rows(const std::vector<int64_t>& row_indices);
  //Matrix swap_cols(const std::vector<int64_t>& col_indices);

  // Destructively resize.
  //void destructive_resize(const int64_t nrows, const int64_t ncols);
};

}  // namespace Hatrix
