#include "Hatrix/classes/Matrix.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace Hatrix {

Matrix::Matrix(int64_t rows, int64_t cols)
    : rows(rows),
      cols(cols),
      stride(rows),
      data(std::make_shared<std::vector<double>>(rows * cols, 0)),
      data_ptr(data->data()) {}

Matrix::Matrix(const Matrix& A)
    : rows(A.rows),
      cols(A.cols),
      stride(A.stride),
      data(std::make_shared<std::vector<double>>(*A.data)),  // Manual deep copy
      data_ptr(data->data()) {}

Matrix& Matrix::operator=(const Matrix& A) {
  rows = A.rows;
  cols = A.cols;
  stride = A.stride;
  // Manual deep copy
  data = std::make_shared<std::vector<double>>(*A.data);
  data_ptr = data->data();
  return *this;
}

const Matrix& Matrix::operator=(const double a) {
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j) (*this)(i, j) = a;
  return *this;
}

double* Matrix::operator&() { return data_ptr; }
const double* Matrix::operator&() const { return data_ptr; }

double& Matrix::operator()(int64_t i, int64_t j) {
  assert(i < rows);
  assert(j < cols);
  return data_ptr[j * stride + i];
}
const double& Matrix::operator()(int64_t i, int64_t j) const {
  assert(i < rows);
  assert(j < cols);
  return data_ptr[j * stride + i];
}

void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  assert(data->size() == rows * cols);
  assert(data->data() == data_ptr);
  for (int64_t j = 0; j < new_cols; ++j) {
    for (int64_t i = 0; i < new_rows; ++i) {
      data_ptr[j * new_rows + i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
  stride = rows;
  data->resize(rows * cols);
}

int64_t Matrix::min_dim() const { return std::min(rows, cols); }
int64_t Matrix::max_dim() const { return std::max(rows, cols); }

}  // namespace Hatrix
