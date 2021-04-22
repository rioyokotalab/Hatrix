#include "Hatrix/classes/Matrix.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
using std::int64_t;
#include <cstdlib>
#include <cstring>


namespace Hatrix {

Matrix::~Matrix() { std::free(data_); }

Matrix::Matrix(int64_t rows, int64_t cols) : rows(rows), cols(cols) {
  data_ = (double*)std::calloc(rows*cols, sizeof(double));
}

Matrix::Matrix(const Matrix& A) : rows(A.rows), cols(A.cols) {
  data_ = (double*)std::malloc(rows*cols*sizeof(double));
  std::memcpy(data_, A.data_, rows*cols*sizeof(double));
}

Matrix& Matrix::operator=(const Matrix& A) {
  rows = A.rows;
  cols = A.cols;
  data_ = (double*)std::malloc(rows*cols*sizeof(double));
  std::memcpy(data_, A.data_, rows*cols*sizeof(double));
  return *this;
}

Matrix::Matrix(Matrix&& A) : rows(std::move(A.rows)), cols(std::move(A.cols)) {
  data_ = A.data_;
  A.data_ = nullptr;
}

Matrix& Matrix::operator=(Matrix&& A) {
  std::swap(rows, A.rows);
  std::swap(cols, A.cols);
  std::swap(data_, A.data_);
  return *this;
}

const Matrix& Matrix::operator=(const double a) {
  for (int64_t i=0; i<rows; ++i) for (int64_t j=0; j<cols; ++j)
    (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_; }
const double* Matrix::operator&() const { return data_; }

double& Matrix::operator()(int64_t i, int64_t j) {
  assert(i < rows);
  assert(j < cols);
  return data_[j*rows+i];
}
const double& Matrix::operator()(int64_t i, int64_t j) const {
  assert(i < rows);
  assert(j < cols);
  return data_[j*rows+i];
}

void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  for (int64_t j=0; j<new_cols; ++j) {
    for (int64_t i=0; i<new_rows; ++i) {
      data_[j*new_rows+i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
  data_ = (double*)std::realloc(data_, rows*cols*sizeof(double));
}

int64_t Matrix::min_dim() const { return std::min(rows, cols); }
int64_t Matrix::max_dim() const { return std::max(rows, cols); }

} // namespace Hatrix
