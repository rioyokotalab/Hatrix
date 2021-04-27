#include "Hatrix/classes/Matrix.h"

#include <cstdint>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

namespace Hatrix {

Matrix::~Matrix() { cudaFree(data_); }

Matrix::Matrix(int64_t rows, int64_t cols) : rows(rows), cols(cols) {
  cudaMallocManaged(reinterpret_cast<void**>(&data_), rows*cols*sizeof(double));
  cudaMemset(data_, 0, rows*cols*sizeof(double));
}

Matrix::Matrix(const Matrix& A) : rows(A.rows), cols(A.cols) {
  cudaMallocManaged(reinterpret_cast<void**>(&data_), rows*cols*sizeof(double));
  cudaMemcpy(data_, A.data_, rows*cols*sizeof(double), cudaMemcpyDefault);
}

const Matrix& Matrix::operator=(const double a) {
  for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j)
    (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_; }
const double* Matrix::operator&() const { return data_; }

double& Matrix::operator()(int64_t i, int64_t j) { return data_[j*rows+i]; }
const double& Matrix::operator()(int64_t i, int64_t j) const { return data_[j*rows+i]; }

void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  for (int i=0; i<new_rows; ++i) {
    for (int j=0; j<new_cols; ++j) {
      data_[j*new_rows+i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;

  double* new_data_;
  cudaMallocManaged(reinterpret_cast<void**>(&new_data_), rows*cols*sizeof(double));
  cudaMemcpy(new_data_, data_, rows*cols*sizeof(double), cudaMemcpyDefault);
  cudaFree(data_);
  data_ = new_data_;
}

int64_t Matrix::min_dim() const { return rows > cols ? cols : rows; }
int64_t Matrix::max_dim() const { return rows > cols ? rows : cols; }

void Matrix::print() const {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%f ", data_[j * rows + i]);
    printf("\n");
  }
  printf("\n");
}

} // namespace Hatrix
