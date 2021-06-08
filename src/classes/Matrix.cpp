#include "Hatrix/classes/Matrix.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace Hatrix {

class Matrix::DataHandler {
 private:
  std::vector<double> data_;

 public:
  DataHandler() = default;

  ~DataHandler() = default;

  DataHandler(int64_t size, double init_value) : data_(size, init_value) {}

  double* get_ptr() { return data_.data(); }

  const double* get_ptr() const { return data_.data(); }

  size_t size() const { return data_.size(); }

  void resize(int64_t size) { data_.resize(size); }
};

Matrix::Matrix(const Matrix& A)
    : rows(A.rows),
      cols(A.cols),
      stride(A.stride),
      data(std::make_shared<DataHandler>(*A.data)),  // Manual deep copy
      data_ptr(data->get_ptr()) {}

Matrix& Matrix::operator=(const Matrix& A) {
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
  assert(data->get_ptr() == data_ptr);
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
  data->resize(rows * cols);
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

std::vector<Matrix> Matrix::split(const std::vector<int64_t>& row_split_indices,
                                  const std::vector<int64_t>& col_split_indices,
                                  bool copy) const {
  std::vector<Matrix> parts;
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
      Matrix part_of_this;
      if (copy) {
        part_of_this = Matrix(n_rows, col_end - col_start);
        double* part_start = &data_ptr[col_start * stride + row_start];
        for (int64_t j = 0; j < part_of_this.cols; j++) {
          for (int64_t i = 0; i < part_of_this.rows; i++) {
            part_of_this(i, j) = part_start[j * stride + i];
          }
        }
      } else {
        part_of_this.rows = n_rows;
        part_of_this.cols = col_end - col_start;
        part_of_this.stride = stride;
        part_of_this.data = std::make_shared<DataHandler>(*data);
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

void Matrix::print() const {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

size_t Matrix::memory_used() const { return rows * cols * sizeof(double); }

size_t Matrix::shared_memory_used() const {
  return data->size() * sizeof(double);
}

}  // namespace Hatrix
