#include "Hatrix/classes/Matrix.h"

#include <cassert>
#include <cstdlib>
#include <cstring>


namespace Hatrix {

Matrix::~Matrix() { std::free(data_); }

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
  data_ = (double*)std::calloc(rows*cols, sizeof(double));
}

Matrix::Matrix(const Matrix& A) : rows(A.rows), cols(A.cols) {
  data_ = (double*)std::malloc(rows*cols*sizeof(double));
  std::memcpy(data_, A.data_, rows*cols*sizeof(double));
}

const Matrix& Matrix::operator=(const double a) {
  for (int i=0; i<rows; ++i) for (int j=0; j<cols; ++j)
    (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_; }
const double* Matrix::operator&() const { return data_; }

double& Matrix::operator()(int i, int j) { return data_[j*rows+i]; }
const double& Matrix::operator()(int i, int j) const { return data_[j*rows+i]; }

void Matrix::shrink(int new_rows, int new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  for (int i=0; i<new_rows; ++i) {
    for (int j=0; j<new_cols; ++j) {
      data_[j*new_rows+i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
  data_ = (double*)std::realloc(data_, rows*cols*sizeof(double));
}

int Matrix::min_dim() { return rows > cols ? cols : rows; }

} // namespace Hatrix
