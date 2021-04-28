#include "Hatrix/functions/arithmetics.h"

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"

#include <cmath>
#include <cassert>
#include <cstdint>
using std::int64_t;


namespace Hatrix {

Matrix& operator+=(Matrix& A, const Matrix& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  for (int64_t j=0; j<A.cols; ++j) for (int64_t i=0; i<A.rows; ++i) {
    A(i, j) += B(i, j);
  }
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
  for (int64_t j=0; j<A.cols; ++j) for (int64_t i=0; i<A.rows; ++i) {
    A(i, j) -= B(i, j);
  }
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

Matrix abs(const Matrix& A) {
  Matrix A_abs(A.rows, A.cols);
  for (int64_t j=0; j<A.cols; ++j)
    for (int64_t i=0; i<A.rows; ++i)
      A_abs(i, j) = std::abs(A(i, j));
  
  return A_abs;
}

} // namespace Hatrix
