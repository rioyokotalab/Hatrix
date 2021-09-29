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
  return Hatrix::matmul(A, B, false, false, 1);
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

Matrix lower_tri(const Matrix& A, bool diag) {
  Matrix A_lower(A.rows, A.cols);
  for(int64_t i = 0; i < A.rows; i++) {
    for(int64_t j = 0; j < std::min(i+1, A.cols); j++) {
      A_lower(i, j) = (i == j && diag ? 1. : A(i, j));
    }
  }
  return A_lower;
}

Matrix upper_tri(const Matrix& A, bool diag) {
  Matrix A_upper(A.rows, A.cols);
  for(int64_t i = 0; i < A.rows; i++) {
    for(int64_t j = i; j < A.cols; j++) {
      A_upper(i, j) = (i == j && diag ? 1. : A(i, j));
    }
  }
  return A_upper;
}
   

}  // namespace Hatrix
