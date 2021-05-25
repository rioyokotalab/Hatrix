#include "Hatrix/classes/Matrix.h"

#include <stdio.h>

#include <cassert>
#include <cstdint>
#include <memory>

#include "cuda_runtime_api.h"

namespace Hatrix {

class Matrix::DataHandler {
 private:
  double* data_ = nullptr;

 public:
  int64_t size;

  DataHandler() = default;

  ~DataHandler() { cudaFree(data_); }

  DataHandler(const DataHandler& A) : size(A.size) {
   cudaMallocManaged(reinterpret_cast<void **>(&data_), size * sizeof(double));
   cudaMemcpy(data_, A.data_, size * sizeof(double), cudaMemcpyDefault);
  }

  DataHandler& operator=(const DataHandler& A) {
    size = A.size;
    cudaMallocManaged(reinterpret_cast<void **>(&data_), size * sizeof(double));
    cudaMemcpy(data_, A.data_, size * sizeof(double), cudaMemcpyDefault);
    return *this;
  }

  DataHandler(DataHandler&& A) : size(std::move(A.size)) {
    std::swap(data_, A.data_);
  }

  DataHandler& operator=(DataHandler&& A) {
    std::swap(size, A.size);
    std::swap(data_, A.data_);
    return *this;
  }

  DataHandler(int64_t size, double init_value) : size(size) {
   cudaMallocManaged(reinterpret_cast<void **>(&data_), size * sizeof(double));
   cudaMemset(data_, 0, size * sizeof(double));
  }

  double* get_ptr() { return data_; }

  const double* get_ptr() const { return data_; }

  void resize(int64_t new_size) {
    double* new_data_;
    cudaMallocManaged(reinterpret_cast<void **>(&new_data_),
                      new_size * sizeof(double));
    cudaMemcpy(new_data_, data_, new_size * sizeof(double), cudaMemcpyDefault);
    cudaFree(data_);
    size = new_size;
    data_ = new_data_;
  }
};

Matrix::Matrix(const Matrix &A)
    : rows(A.rows),
      cols(A.cols),
      stride(A.stride),
      data(std::make_shared<DataHandler>(*A.data)),  // Manual deep copy
      data_ptr(data->get_ptr()) {}

Matrix &Matrix::operator=(const Matrix &A) {
  rows = A.rows;
  cols = A.cols;
  stride = A.stride;
  // Manual deep copy
  data = std::make_shared<DataHandler>(*A.data);
  data_ptr = data->get_ptr();
  return *this;
}

Matrix::Matrix(int64_t rows, int64_t cols)
    : rows(rows),
      cols(cols),
      stride(rows),
      data(std::make_shared<DataHandler>(rows * cols, 0)),
      data_ptr(data->get_ptr()) {}

const Matrix &Matrix::operator=(const double a) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) (*this)(i, j) = a;
  return *this;
}

double *Matrix::operator&() { return data->get_ptr(); }
const double *Matrix::operator&() const { return data->get_ptr(); }

double &Matrix::operator()(int64_t i, int64_t j) {
  return data_ptr[j * stride + i];
}
const double &Matrix::operator()(int64_t i, int64_t j) const {
  return data_ptr[j * stride + i];
}

void Matrix::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  assert(data->size == rows * cols);
  assert(data->get_ptr() == data_ptr);
  for (int j = 0; j < new_cols; ++j) {
    for (int i = 0; i < new_rows; ++i) {
      data_ptr[j * new_rows + i] = (*this)(i, j);
    }
  }
  rows = new_rows;
  cols = new_cols;
  stride = rows;
  data->resize(rows * cols);
  data_ptr = data->get_ptr();
}

int64_t Matrix::min_dim() const { return rows > cols ? cols : rows; }
int64_t Matrix::max_dim() const { return rows > cols ? rows : cols; }

void Matrix::print() const {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) printf("%f ", data_ptr[j * stride + i]);
    printf("\n");
  }
  printf("\n");
}

}  // namespace Hatrix
