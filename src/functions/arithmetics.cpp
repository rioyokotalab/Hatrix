#include "Hatrix/functions/arithmetics.h"

#include <cassert>
#include <cmath>
#include <cstdint>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"

namespace Hatrix {

Matrix& operator+=(Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A(i, j) += B(i, j);

  return A;
}

Matrix operator+(const Matrix& A, const Matrix& B) {
  Matrix C(A);
  C += B;
  return C;
}

Matrix& operator-=(Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A(i, j) -= B(i, j);

  return A;
}

Matrix operator-(const Matrix& A, const Matrix& B) {
  Matrix C(A);
  C -= B;
  return C;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.rows, B.cols);
  Hatrix::matmul(A, B, C, false, false, 1, 0);
  return C;
}

Matrix& operator*=(Matrix& A, double alpha) {
  Hatrix::scale(A, alpha);
  return A;
}

Matrix operator*(const Matrix& A, double alpha) {
  Matrix C(A);
  C *= alpha;
  return C;
}

Matrix operator*(double alpha, const Matrix& A) {
  Matrix C(A);
  C *= alpha;
  return C;
}

Matrix abs(const Matrix& A) {
  Matrix A_abs(A.rows, A.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A_abs(i, j) = std::abs(A(i, j));

  return A_abs;
}

Matrix transpose(const Matrix& A) {
  Matrix A_trans(A.cols, A.rows);
  for (int64_t i = 0; i < A_trans.rows; i++)
    for (int64_t j = 0; j < A_trans.cols; j++) A_trans(i, j) = A(j, i);

  return A_trans;
}

}  // namespace Hatrix
