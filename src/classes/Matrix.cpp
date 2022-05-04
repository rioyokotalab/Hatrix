#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/lapack.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <fstream>

namespace Hatrix {

class Matrix::DataHandler {
 private:
  int64_t data_size;
  double *data_;

 public:
  DataHandler() {
    data_ = nullptr;
  };

  ~DataHandler() {
    if (data_) {
      delete[] data_;
    }
  };

  DataHandler(int64_t size, double init_value) : data_size(size) {
    try {
      data_ = new double[size];
      for (int64_t i = 0; i < size; ++i) {
        data_[i] = init_value;
      }
    }
    catch (std::bad_alloc& e) {
      std::cout << "DataHandler(size, init_value) -> Cannot allocate data." << std::endl;
      exit(2);
    }
  }

  double* get_ptr() { return data_; }

  const double* get_ptr() const { return data_; }

  size_t size() const { return data_size; }

  void resize(int64_t new_size) {
    data_size = new_size;
    try {
      data_ = new double[data_size];
    }
    catch (std::bad_alloc& e) {
      std::cout << "DataHandler#resize(size) -> Cannot allocate data." << std::endl;
      exit(2);
    }
  }
};

Matrix::Matrix() {
  rows = -1;
  cols = -1;
  stride = -1;
  data_ptr = nullptr;
  is_view = false;
}

Matrix::~Matrix() {
  if (data_ptr != nullptr && !is_view) {
    delete[] data_ptr;
  }

  if (is_view) {
    data_ptr = nullptr;
  }
}

// Copy constructor.
Matrix::Matrix(const Matrix& A)
    : rows(A.rows),
      cols(A.cols),
      stride(A.rows) {
  try {
    data_ptr = new double[rows * cols];
  }
  catch (std::bad_alloc& e) {
    std::cout << "Matrix(const Matrix& A) -> "
              << e.what()
              << " rows= " << rows
              << " cols= " << cols
              << std::endl;
  }
  // Reinitilize the stride to number of rows if its a view.
  // stride = A.rows;
  // Need the for loop and cannot init directly in the initializer list because
  // the object might be a view and therefore will not get copied properly.
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      (*this)(i, j) = A(i, j);
    }
  }
}

// Move assignment constructor.
Matrix& Matrix::operator=(Matrix&& A) {
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
Matrix& Matrix::operator=(const Matrix& A) {
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


Matrix::Matrix(int64_t rows, int64_t cols)
    : rows(rows),
      cols(cols),
      stride(rows) {
  try {
    data_ptr = new double[rows * cols];
  }
  catch (std::bad_alloc& e) {
    std::cout << "Matrix(rows, cols) -> " << e.what() << std::endl;
  }
}

const Matrix& Matrix::operator=(const double a) {
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j) (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_ptr; }
const double* Matrix::operator&() const { return data_ptr; }

double& Matrix::operator()(int64_t i, int64_t j) {
  assert(i < rows && i >= 0);
  assert(j < cols && j >= 0);
  return data_ptr[j * stride + i];
}
const double& Matrix::operator()(int64_t i, int64_t j) const {
  assert(i < rows && i >= 0);
  assert(j < cols && j >= 0);
  return data_ptr[j * stride + i];
}

void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  // Only need to reorganize if the number of rows (leading dim) is reduced
  if (new_rows < rows) {
    for (int64_t j = 0; j < new_cols; ++j) {
      for (int64_t i = 0; i < new_rows; ++i) {
        data_ptr[j * new_rows + i] = (*this)(i, j);
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
        double* part_start = &data_ptr[col_start * stride + row_start];
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
        part_of_this.data_ptr = &data_ptr[col_start * stride + row_start];
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
        std::cout << std::setw(10) << 0 << " ";
      }
      else {
        std::cout << std::setprecision(3) << std::setw(10) <<  (*this)(i, j) << " ";
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
    if (data_ptr) {
      delete[] data_ptr;
    }

    data_ptr = new double[rows * cols];
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

size_t Matrix::shared_memory_used() const {
  return rows * cols * sizeof(double);
}

Matrix Matrix::block_ranks(int64_t nblocks, double accuracy) const {
  Matrix out(nblocks, nblocks);

  auto this_splits = (*this).split(nblocks, nblocks);
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix Utemp, Stemp, Vtemp;
      std::tie(Utemp, Stemp, Vtemp) = Hatrix::error_svd(this_splits[i * nblocks + j], accuracy);
      out(i, j) = Stemp.rows;
    }
  }

  return out;
}

}  // namespace Hatrix
