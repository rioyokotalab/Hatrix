#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/classes/Matrix.h"

#include <cassert>

namespace Hatrix {

void matadd(
  const Matrix& A, const Matrix& B, Matrix& C) {
  assert(A.rows == B.rows && A.rows == C.rows);
  assert(A.cols == B.cols && A.cols == C.cols);

  for (int64_t j=0; j<A.cols; ++j)
    for (int64_t i=0; i<A.rows; ++i)
      C(i, j) += A(i, j) + B(i, j);
};

void matsub(
  const Matrix& A, const Matrix& B, Matrix& C) {
  assert(A.rows == B.rows && A.rows == C.rows);
  assert(A.cols == B.cols && A.cols == C.cols);

  for (int64_t j=0; j<A.cols; ++j)
    for (int64_t i=0; i<A.rows; ++i)
      C(i, j) += A(i, j) - B(i, j);
};

Matrix operator+(const Matrix& A, const Matrix& B){
  Matrix C(A.rows, A.cols);
  matadd(A, B, C);

  return C;
}

Matrix operator-(const Matrix& A, const Matrix& B){
  Matrix C(A.rows, A.cols);
  matsub(A, B, C);
  
  return C;
}

} // namespace Hatrix