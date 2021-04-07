#include "Hatrix/classes/Matrix.h"

#include <cstdlib>
#include <cstring>


namespace Hatrix {

Matrix::~Matrix() { std::free(data_); }

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
  data_ = (double*)std::malloc(rows*cols*sizeof(double));
  for (int i=0; i<rows; ++i) for (int j=0; j<cols; ++j)
    (*this)(i, j) = 0;
}

Matrix::Matrix(const Matrix& A) : rows(A.rows), cols(A.cols) {
  data_ = (double*)std::malloc(rows*cols*sizeof(double));
  std::memcpy(data_, A.data_, rows*cols*sizeof(double));
}

const Matrix& Matrix::operator=(const double a) {
  for (int i=0; i<rows; ++i) for (int j=0; j<cols; ++j)
    (*this)(i, j) = a;
}

double* Matrix::operator&() { return data_; }
const double* Matrix::operator&() const { return data_; }

double& Matrix::operator()(int i, int j) { return data_[i*cols+j]; }
const double& Matrix::operator()(int i, int j) const { return data_[i*cols+j]; }

} // namespace Hatrix
