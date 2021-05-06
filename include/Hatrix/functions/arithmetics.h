#pragma once
#include "Hatrix/classes/Matrix.h"


namespace Hatrix {

Matrix& operator+=(Matrix& A, const Matrix& B);
Matrix operator+(const Matrix& A, const Matrix& B);

Matrix& operator-=(Matrix& A, const Matrix& B);
Matrix operator-(const Matrix& A, const Matrix& B);

Matrix operator*(const Matrix& A, const Matrix& B);
Matrix operator*(const Matrix& A, double alpha);
Matrix operator*(double alpha, const Matrix& A);

Matrix abs(const Matrix& A);

} // namespace Hatrix
