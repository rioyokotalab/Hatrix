#include "Hatrix/classes/Matrix.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
using std::uint64_t;
#include <cstdlib>
#include <cstring>


namespace Hatrix {

Matrix::~Matrix() { std::free(data_); }

Matrix::Matrix(uint64_t rows, uint64_t cols) : rows(rows), cols(cols) {
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
  for (uint64_t i=0; i<rows; ++i) for (uint64_t j=0; j<cols; ++j)
    (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_; }
const double* Matrix::operator&() const { return data_; }

double& Matrix::operator()(uint64_t i, uint64_t j) { return data_[j*rows+i]; }
const double& Matrix::operator()(uint64_t i, uint64_t j) const {
  return data_[j*rows+i];
}

void Matrix::shrink(uint64_t new_rows, uint64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  for (uint64_t j=0; j<new_cols; ++j) {
    for (uint64_t i=0; i<new_rows; ++i) {
      data_[j*new_rows+i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
  data_ = (double*)std::realloc(data_, rows*cols*sizeof(double));
}

uint64_t Matrix::min_dim() { return std::min(rows, cols); }

} // namespace Hatrix
