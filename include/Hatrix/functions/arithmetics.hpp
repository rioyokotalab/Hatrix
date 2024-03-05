#pragma once
#include "Hatrix/classes/Matrix.hpp"

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

// Compute the abs() value of each entry of A and return the result.
Matrix abs(const Matrix& A);

// Transpose the matrix A and return a new matrix.
Matrix transpose(const Matrix& A);
// Return the lower triangular part of this matrix. Pass diag as false to
// exclude the diagonal.
Matrix lower_tri(const Matrix& A, bool diag=false);

// Return the upper triangular part of this matrix. Pass diag as false to
// exclude the diagonal.
Matrix upper_tri(const Matrix& A, bool diag=false);

}  // namespace Hatrix
