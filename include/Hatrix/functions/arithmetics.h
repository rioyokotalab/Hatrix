#pragma once

namespace Hatrix {

class Matrix;

void matadd(
  const Matrix& A, const Matrix& B, Matrix& C);

void matsub(
  const Matrix& A, const Matrix& B, Matrix& C);

Matrix operator+(const Matrix& A, const Matrix& B);

Matrix operator-(const Matrix& A, const Matrix& B);

} // namespace Hatrix