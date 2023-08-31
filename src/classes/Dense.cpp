#include "Hatrix/classes/Dense.hpp"
//#include "Hatrix/functions/lapack.h"
//#include "Hatrix/functions/blas.h"

//#include <algorithm>
#include <cassert>
//#include <cstdint>
//#include <cstddef>
#include <iostream>
#include <cstring>
#include <complex>
//#include <iomanip>
//#include <memory>
//#include <vector>
//#include <fstream>
//#include <stdexcept>

namespace Hatrix {

template <typename DT>
Dense<DT>::Dense() :
  rows(0), cols(0), is_view(false), data(nullptr){}

// TODO check
template <typename DT>
Dense<DT>::~Dense() {
  deallocate();
}

template <typename DT>
Dense<DT>::Dense(const Dense<DT>& A) :
  rows(A.rows), cols(A.cols),  is_view(A.is_view) {
  if (this->is_view) {
    // shallow copy
    this->data = A.data;
  } else {
    // deep copy
    this->allocate(false);
    this->copy_values(A);
  }
}

template <typename DT>
Dense<DT>::Dense(const Dense<DT>& A, const bool copy) :
  rows(A.rows), cols(A.cols), is_view(!copy) {
  if (!copy) {
    // shallow copy (creates a view)
    this->data = A.data;
  }
  else {
    // deep copy
    this->allocate(false);
    this->copy_values(A);
  }
}

// Copy assignment operator.
// The behaviour is defined as follows:
// A should contain a copy of all values from B
// If B is a view, A is a view to the same data
// If B is not a view, A is a deep copy of the data
template <typename DT>
Dense<DT>& Dense<DT>::operator=(const Dense<DT>& A) {
  // protect against invalid self-assignment
  //if (this != &A) {
    rows = A.rows;
    cols = A.cols;
    
    if (A.is_view) {
      deallocate();
      data = A.data;
    } else {
      if (rows * cols != A.rows * A.cols) {
        deallocate();
      }
      if (is_view || rows * cols != A.rows * A.cols) {
        allocate(false);
      }
      copy_values(A);
    }
    is_view = A.is_view;
  //}
  /*
  if (this->is_view) {
    if (A.is_view) {
      this->rows = A.rows;
      this->cols = A.cols;
      this->data = A.data;
    } else {
      this->rows = A.rows;
      this->cols = A.cols;
      is_view = false;
      allocate();
      // TODO this should really be handled by cblas (which might provide parallelism) or memset
      for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
          (*this)(i, j) = A(i, j);
        }
      }
    }
  } else {
    if (A.is_view) {
      if (data) {
        delete[] data;
      }
      this->rows = A.rows;
      this->cols = A.cols;
      this->data = A.data;
      is_view = A.is_view;
    } else {
      if (rows * cols != A.rows * A.cols) {
        if (data) {
          delete[] data;
        }
        allocate();
      }
      this->rows = A.rows;
      this->cols = A.cols;
      // TODO this should really be handled by cblas (which might provide parallelism) or memset
      for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
          (*this)(i, j) = A(i, j);
        }
      }
    }
  }*/

  return *this;
}

// Move constructor
template <typename DT>
Dense<DT>::Dense(Dense<DT>&& A)
  : rows(A.rows), cols(A.cols),
  is_view(A.is_view), data(A.data) {
    // TODO maybe A should be emptied even if it is a view
    if (!A.is_view) {
      A.empty();
    }
}

// Move assignment constructor.
// Moves all data from A to this, leaves A empty
// Deallocates memory for this, if necessary
template <typename DT>
Dense<DT>& Dense<DT>::operator=(Dense<DT>&& A) {
  // Protect against invalid self move-assignment
  //if (this != &A) {
    deallocate();
    rows = A.rows;
    cols = A.cols;
    is_view = A.is_view;
    data = A.data;
    // Leave A empty (assures it does not deallocate memory)
    A.empty();
  //}

  return *this;
}


template <typename DT>
Dense<DT>::Dense(const int rows, const int cols, const bool init)
    : rows(rows), cols(cols), is_view(false) {
  this->allocate(init);
}

template <typename DT>
const Dense<DT>& Dense<DT>::operator=(const DT a) {
  // TODO this should also be parallelized or call at least fill or something
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j) (*this)(i, j) = a;
  return *this;
}

template <typename DT>
DT* Dense<DT>::operator&() { return data; }

template <typename DT>
const DT* Dense<DT>::operator&() const { return data; }

template <typename DT>
DT& Dense<DT>::operator()(unsigned int i, unsigned int j) {
  if (i >= rows || i < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected i < rows && i > 0, but got i= " +
                                std::to_string(i) + " rows= " + std::to_string(rows));
  }
  if (j >= cols || j < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected j > cols && j > 0, but got j=" +
                                std::to_string(j) + " cols= " + std::to_string(cols));
  }
  return data[i + j * rows];
}

template <typename DT>
const DT& Dense<DT>::operator()(unsigned int i, unsigned int j) const {
  if (i >= rows || i < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected i < rows && i > 0, but got i= " +
                                std::to_string(i) + " rows= " + std::to_string(rows));
  }
  if (j >= cols || j < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected j > cols && j > 0, but got j=" +
                                std::to_string(j) + " cols= " + std::to_string(cols));
  }
  return data[i + j * rows];
}
/*
void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  // Only need to reorganize if the number of rows (leading dim) is reduced
  if (new_rows < rows) {
    for (int64_t j = 0; j < new_cols; ++j) {
      for (int64_t i = 0; i < new_rows; ++i) {
        data[i + j * new_rows] = (*this)(i, j);
      }
    }
  }
  rows = new_rows;
  cols = new_cols;
  stride = rows;
}

std::vector<Matrix> Matrix::split(int64_t n_row_splits, int64_t n_col_splits,
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

std::vector<Matrix> Matrix::split(const std::vector<int64_t>& _row_split_indices,
                                  const std::vector<int64_t>& _col_split_indices,
                                  bool copy) const {
  std::vector<Matrix> parts;
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
      Matrix part_of_this;
      if (copy) {
        part_of_this = Matrix(n_rows, n_cols);
        double* part_start = &data[col_start * stride + row_start];
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
        part_of_this.data = &data[row_start + col_start * stride];
      }
      parts.emplace_back(std::move(part_of_this));
      col_start = col_end;
    }
    row_start = row_end;
  }

  return parts;
}

int64_t Matrix::min_dim() const { return std::min(rows, cols); }
int64_t Matrix::max_dim() const { return std::max(rows, cols); }
int64_t Matrix::numel() const { return rows * cols; }

void Matrix::print() const {
  if (rows == 0 || cols == 0) { return; }
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if ((*this)(i, j) > -1e-10 && (*this)(i, j) < 1e-10) {
        std::cout << std::setw(15) << 0 << " ";
      }
      else {
        std::cout << std::setprecision(6) << std::setw(15) <<  (*this)(i, j) << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void Matrix::print_meta() const {
  std::cout << "rows=" << rows << " cols=" << cols << " stride=" << stride << std::endl;
}

void Matrix::read_file(std::string in_file) {
  std::ifstream file(in_file, std::ios::in);

  file >> rows >> cols;
  stride = rows;
  try {
    if (data) {
      delete[] data;
    }

    data = new double[rows * cols];
  }
  catch (std::bad_alloc& e) {
    std::cout << "Matrix#read_file(string in_file) -> Cannot allocate memory.\n";
  }

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      int64_t irow, jcol; double value;
      file >> irow >> jcol >> value;
      (*this)(irow, jcol) = value;
    }
  }

  file.close();
}

void Matrix::out_file(std::string out_file) const {
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

size_t Matrix::memory_used() const { return rows * cols * sizeof(double); }

Matrix Matrix::block_ranks(int64_t nblocks, double accuracy) const {
  Matrix out(nblocks, nblocks);

  auto this_splits = (*this).split(nblocks, nblocks);
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix Utemp, Stemp, Vtemp;
      int64_t rank;
      std::tie(Utemp, Stemp, Vtemp, rank) = Hatrix::error_svd(this_splits[i * nblocks + j], accuracy);
      out(i, j) = rank;
    }
  }

  return out;
}

Matrix Matrix::swap_rows(const std::vector<int64_t>& row_indices) {
  Matrix out(rows, cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(row_indices[i], j) = (*this)(i, j);
    }
  }

  return out;
}

Matrix Matrix::swap_cols(const std::vector<int64_t>& col_indices) {
  Matrix out(rows, cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, col_indices[j]) = (*this)(i, j);
    }
  }

  return out;
}

void Matrix::destructive_resize(const int64_t nrows, const int64_t ncols) {
  rows = nrows;
  cols = ncols;
  stride = nrows;
  is_view = false;
  data = new double[rows * cols];
}

Matrix Matrix::tril(const int64_t diag) {
  Matrix out(rows, cols);

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j <= std::min(i + diag, cols); ++j) {
      out(i, j) = (*this)(i, j);
    }
  }

  return out;
}*/

template <typename DT>
void Dense<DT>::empty() {
  this->rows = 0;
  this->cols = 0;
  this->is_view = false;
  this->data = nullptr;
}

template <typename DT>
void Dense<DT>::allocate(const bool init) {
  assert(!this->is_view);

  unsigned int n = this->rows * this->cols;
  if (n < 1) {
    this->data = nullptr;
    return;
  }
  try {
    if (init) {
      this->data = new DT[n]();
    } else {
      this->data = new DT[n];
    }
  }
  catch (std::bad_alloc& e) {
    this->empty();
    std::cout << "Allocation failed for Dense[rows, cols] -> "
      << e.what() << std::endl;
  }
}

template <typename DT>
void Dense<DT>::deallocate() {
  if (!is_view && data) {
      delete[] data;
  }
}

template <typename DT>
void Dense<DT>::copy_values(const Dense<DT>& A) {
  assert(this->rows == A.rows);
  assert(this->cols == A.rows);

  // TODO Should this be handled by cblas (which might provide parallelism)?
  std::memcpy(this->data, A.data, A.rows * A.cols * sizeof(DT));
}


// explicit template instantiations (these are the only available data-types)
template class Dense<float>;
template class Dense<double>;
template class Dense<std::complex<float>>;
template class Dense<std::complex<double>>;

}  // namespace Hatrix
