#pragma once
#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

Matrix& operator+=(Matrix& A, const Matrix& B);
Matrix operator+(const Matrix& A, const Matrix& B);

Matrix& operator-=(Matrix& A, const Matrix& B);
Matrix operator-(const Matrix& A, const Matrix& B);

Matrix operator*(const Matrix& A, const Matrix& B);
Matrix& operator*=(Matrix& A, double alpha);
Matrix operator*(const Matrix& A, double alpha);
Matrix operator*(double alpha, const Matrix& A);

Matrix& operator/=(Matrix& A, double alpha);

Matrix abs(const Matrix& A);
Matrix transpose(const Matrix& A);
Matrix lower_tri(const Matrix& A, bool diag=false);
Matrix upper_tri(const Matrix& A, bool diag=false);

}  // namespace Hatrix
