#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/functions/blas.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace Hatrix {

template <typename DT>
Matrix<DT>::Matrix() {
  rows = -1;
  cols = -1;
  stride = -1;
  data_ptr = nullptr;
  is_view = false;
}

template <typename DT>
Matrix<DT>::~Matrix() {
  if (data_ptr != nullptr && !is_view) {
    delete[] data_ptr;
  }

  if (is_view) {
    data_ptr = nullptr;
  }
}

template <typename DT>
Matrix<DT>::Matrix(const Matrix<DT>& A) : rows(A.rows), cols(A.cols) {
  if (A.is_view) {
    is_view = true;
    stride = A.stride;
    data_ptr = A.data_ptr;
  }
  else {
    try {
      data_ptr = new DT[rows * cols]();
    }
    catch (std::bad_alloc& e) {
      std::cout << "Matrix(const Matrix& A, bool copy) -> "
                << e.what()
                << " rows= " << rows
                << " cols= " << cols
                << std::endl;
    }
    is_view = false;
    stride = A.rows;
    for (int i = 0; i < A.rows; ++i) {
      for (int j = 0; j < A.cols; ++j) {
        (*this)(i, j) = A(i, j);
      }
    }
  }
}

// Copy constructor if you want to explicitly make a copy.
template <typename DT>
Matrix<DT>::Matrix(const Matrix<DT>& A, bool copy)
    : rows(A.rows),
      cols(A.cols) {
  if (A.is_view && !copy) {
    is_view = true;
    stride = A.stride;
    data_ptr = A.data_ptr;
  }
  else {
    try {
      data_ptr = new DT[rows * cols]();
    }
    catch (std::bad_alloc& e) {
      std::cout << "Matrix(const Matrix& A, bool copy) -> "
                << e.what()
                << " rows= " << rows
                << " cols= " << cols
                << std::endl;
    }
    is_view = false;
    stride = A.rows;
    // array_copy(A.data_ptr, data_ptr, rows * cols);
    // // Reinitilize the stride to number of rows if its a view.
    // // stride = A.rows;
    // // Need the for loop and cannot init directly in the initializer list because
    // // the object might be a view and therefore will not get copied properly.
    for (int i = 0; i < A.rows; ++i) {
      for (int j = 0; j < A.cols; ++j) {
        (*this)(i, j) = A(i, j);
      }
    }
  }
}

  // TODO: Is the behaviour of the move constructor and move assigment
  // divergent wrt views? In the move constructor we are assigning
  // A.data_ptr = nullptr unconditionally, which makes A an empty
  // object in the calling code. In the assignment we are performing
  // a copy of the view and keeping the A still valid. Ideally both
  // should do the same thing(?)
template <typename DT>
Matrix<DT>::Matrix(Matrix<DT>&& A) {
  std::swap(rows, A.rows);
  std::swap(cols, A.cols);
  std::swap(stride, A.stride);
  std::swap(data_ptr, A.data_ptr);
  std::swap(is_view, A.is_view);
  A.data_ptr = nullptr;
}

// Move assignment constructor.
template <typename DT>
Matrix<DT>& Matrix<DT>::operator=(Matrix<DT>&& A) {
  // Need to perform a manual copy (vs. swapping) since A might be
  // being assigned to a view.
  if (is_view) {
    assert((*this).rows == A.rows);
    assert((*this).cols == A.cols);
    for (int i = 0; i < A.rows; ++i) {
      for (int j = 0; j < A.cols; ++j) {
        (*this)(i, j) = A(i, j);
      }
    }
  }
  else {
    std::swap(rows, A.rows);
    std::swap(cols, A.cols);
    std::swap(stride, A.stride);
    std::swap(data_ptr, A.data_ptr);
    std::swap(is_view, A.is_view);
  }

  return *this;
}


// Copy assignment operator.
template <typename DT>
Matrix<DT>& Matrix<DT>::operator=(const Matrix<DT>& A) {
  // Manual copy. We dont simply assign the data pointer since we want to
  // the ability to work with Matrix objects that might be views of an
  // underlying parent Matrix object.
  assert((*this).rows == A.rows);
  assert((*this).cols == A.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      (*this)(i, j) = A(i, j);
    }
  }
  return *this;
}

template <typename DT>
Matrix<DT>::Matrix(int64_t rows, int64_t cols)
    : rows(rows),
      cols(cols),
      stride(rows) {
  try {
    if (rows * cols == 0) {
      data_ptr = nullptr;
    }
    else {
      data_ptr = new DT[rows * cols]();
    }
  }
  catch (std::bad_alloc& e) {
    std::cout << "Matrix(rows, cols) -> " << e.what() << std::endl;
  }
}

template <typename DT>
const Matrix<DT>& Matrix<DT>::operator=(const DT a) {
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j) (*this)(i, j) = a;
  return *this;
}

template <typename DT>
DT* Matrix<DT>::operator&() { return data_ptr; }
template <typename DT>
const DT* Matrix<DT>::operator&() const { return data_ptr; }

template <typename DT>
DT& Matrix<DT>::operator()(int64_t i, int64_t j) {
  if (i >= rows || i < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected i < rows && i > 0, but got i= " +
                                std::to_string(i) + " rows= " + std::to_string(rows));
  }
  if (j >= cols || j < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected j > cols && j > 0, but got j=" +
                                std::to_string(j) + " cols= " + std::to_string(cols));
  }
  return data_ptr[i + j * stride];
}

template <typename DT>
const DT& Matrix<DT>::operator()(int64_t i, int64_t j) const {
  if (i >= rows || i < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected i < rows && i > 0, but got i= " +
                                std::to_string(i) + " rows= " + std::to_string(rows));
  }
  if (j >= cols || j < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected j > cols && j > 0, but got j=" +
                                std::to_string(j) + " cols= " + std::to_string(cols));
  }
  return data_ptr[i + j * stride];
}

template <typename DT>
void Matrix<DT>::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  // Only need to reorganize if the number of rows (leading dim) is reduced
  if (new_rows < rows) {
    for (int64_t j = 0; j < new_cols; ++j) {
      for (int64_t i = 0; i < new_rows; ++i) {
        data_ptr[i + j * new_rows] = (*this)(i, j);
      }
    }
  }
  rows = new_rows;
  cols = new_cols;
  stride = rows;
}

template <typename DT>
std::vector<Matrix<DT>> Matrix<DT>::split(int64_t n_row_splits, int64_t n_col_splits,
                                  bool copy) const {
  int64_t row_split_size = rows / n_row_splits;
  std::vector<int64_t> row_split_indices;
  for (int64_t i=1; i<n_row_splits; ++i) {
    row_split_indices.push_back(row_split_size*i);
  }
  int64_t col_split_size = cols / n_col_splits;
  std::vector<int64_t> col_split_indices;
  for (int64_t j=1; j<n_col_splits; ++j) {
    col_split_indices.push_back(col_split_size*j);
  }
  return split(row_split_indices, col_split_indices, copy);
}

template <typename DT>
std::vector<Matrix<DT>> Matrix<DT>::split(const std::vector<int64_t>& _row_split_indices,
                                  const std::vector<int64_t>& _col_split_indices,
                                  bool copy) const {
  std::vector<Matrix<DT>> parts;
  std::vector<int64_t> row_split_indices(_row_split_indices),
    col_split_indices(_col_split_indices);

  // Allow specifying vectors where the last element is the value of the dimension.
  if (row_split_indices.size() > 1 && row_split_indices[row_split_indices.size()-1] == rows) {
    row_split_indices.pop_back();
  }

  if (col_split_indices.size() > 1 && col_split_indices[col_split_indices.size()-1] == cols) {
    col_split_indices.pop_back();
  }

  auto row_iter = row_split_indices.cbegin();
  int64_t row_start = 0;

  while (row_start < rows) {
    int64_t row_end = row_iter == row_split_indices.end() ? rows : *row_iter++;
    int64_t n_rows = row_end - row_start;
    auto col_iter = col_split_indices.cbegin();
    int64_t col_start = 0;
    while (col_start < cols) {
      int64_t col_end =
          col_iter == col_split_indices.end() ? cols : *col_iter++;
      int64_t n_cols = col_end - col_start;
      Matrix<DT> part_of_this;
      if (copy) {
        part_of_this = Matrix<DT>(n_rows, n_cols);
        DT* part_start = &data_ptr[col_start * stride + row_start];
        for (int64_t j = 0; j < part_of_this.cols; j++) {
          for (int64_t i = 0; i < part_of_this.rows; i++) {
            part_of_this(i, j) = part_start[j * stride + i];
          }
        }
      } else {
        part_of_this.rows = n_rows;
        part_of_this.cols = n_cols;
        part_of_this.stride = stride;
        part_of_this.is_view = true;
        part_of_this.data_ptr = &data_ptr[row_start + col_start * stride];
      }
      parts.emplace_back(std::move(part_of_this));
      col_start = col_end;
    }
    row_start = row_end;
  }

  return parts;
}

template <typename DT>
int64_t Matrix<DT>::min_dim() const { return std::min(rows, cols); }
template <typename DT>
int64_t Matrix<DT>::max_dim() const { return std::max(rows, cols); }
template <typename DT>
int64_t Matrix<DT>::numel() const { return rows * cols; }

template <typename DT>
void Matrix<DT>::print() const {
  if (rows == 0 || cols == 0) { return; }
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if ((*this)(i, j) > -1e-10 && (*this)(i, j) < 1e-10) {
        std::cout << std::setw(13) << 0 << " ";
      }
      else {
        std::cout << std::setprecision(8) << std::setw(13) <<  (*this)(i, j) << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <typename DT>
void Matrix<DT>::print_meta() const {
  std::cout << "rows=" << rows << " cols=" << cols << " stride=" << stride << std::endl;
}

template <typename DT>
void Matrix<DT>::read_file(std::string in_file) {
  std::ifstream file(in_file, std::ios::in);

  file >> rows >> cols;
  stride = rows;
  try {
    if (data_ptr) {
      delete[] data_ptr;
    }

    data_ptr = new DT[rows * cols];
  }
  catch (std::bad_alloc& e) {
    std::cout << "Matrix#read_file(string in_file) -> Cannot allocate memory.\n";
  }

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      // We read double values here and rely on implicit conversion
      int64_t irow, jcol; double value;
      file >> irow >> jcol >> value;
      (*this)(irow, jcol) = value;
    }
  }

  file.close();
}

template <typename DT>
void Matrix<DT>::out_file(std::string out_file) const {
  std::ofstream file;
  file.open(out_file, std::ios::out | std::ios::trunc);

  file << rows << " " << cols << std::endl;
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      file << i << " " << j << " " << (*this)(i, j) << std::endl;
    }
  }

  file.close();
}

template <typename DT>
size_t Matrix<DT>::memory_used() const { return rows * cols * sizeof(DT); }

template <typename DT>
Matrix<DT> Matrix<DT>::block_ranks(int64_t nblocks, double accuracy) const {
  Matrix<DT> out(nblocks, nblocks);

  auto this_splits = (*this).split(nblocks, nblocks);
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix<DT> Utemp, Stemp, Vtemp;
      int64_t rank;
      std::tie(Utemp, Stemp, Vtemp, rank) = Hatrix::error_svd(this_splits[i * nblocks + j], accuracy);
      out(i, j) = rank;
    }
  }

  return out;
}

template <typename DT>
Matrix<DT> Matrix<DT>::swap_rows(const std::vector<int64_t>& row_indices) {
  Matrix<DT> out(rows, cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(row_indices[i], j) = (*this)(i, j);
    }
  }

  return out;
}

template <typename DT>
Matrix<DT> Matrix<DT>::swap_cols(const std::vector<int64_t>& col_indices) {
  Matrix<DT> out(rows, cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, col_indices[j]) = (*this)(i, j);
    }
  }

  return out;
}

template <typename DT>
void Matrix<DT>::destructive_resize(const int64_t nrows, const int64_t ncols) {
  rows = nrows;
  cols = ncols;
  stride = nrows;
  is_view = false;
  data_ptr = new DT[rows * cols];
}

// explicit instantiation (these are the only available data-types)
template class Matrix<float>;
template class Matrix<double>;

}  // namespace Hatrix
