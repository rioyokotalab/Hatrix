#pragma once
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/math_common.h"

namespace Hatrix {
void swap_row(Matrix& A, int64_t r1, int64_t r2);

void swap_col(Matrix& A, int64_t c1, int64_t c2);

void array_copy(const double* from, double* to, int64_t size);

// Perform a matrix multiplication C = beta * C + alpha * A * B.
// Optionally transpose A and B.
void matmul(const Matrix& A, const Matrix& B, Matrix& C, bool transA = false,
            bool transB = false, double alpha = 1.0, double beta = 1.0);

Matrix matmul(const Matrix& A, const Matrix& B, bool transA = false,
              bool transB = false, double alpha = 1.0);

// If transA is false:
//   C = alpha * A * A' + beta * C
// if transA is true:
//   C = alpha * A' * A + beta * C
// A has to be symmetric.
void syrk(const Matrix& A, Matrix& C, Mode uplo, bool transA, double alpha,
            double beta);


// Compute matrix product with one triangular matrix. A is the triangular matrix here.
// B = alpha * op(A) * B
// op(A) is one of op(A) = A, or op(A) = A'
void triangular_matmul(const Matrix& A, Matrix& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha = 1.0);

Matrix triangular_matmul_out(const Matrix& A, const Matrix& B, Side side, Mode uplo,
			     bool transA, bool diag, double alpha = 1.0);

void solve_triangular(const Matrix& A, Matrix& B, Side side, Mode uplo,
                      bool unit_diag, bool transA = false, double alpha = 1.0);

// Solve D*X = alpha*B or X*D = alpha*B
// Ignore non-diagonal elements of D
void solve_diagonal(const Matrix& D, Matrix& B, Side side, double alpha = 1.0);

void scale(Matrix& A, double alpha);

// Scale the rows of A using diagonal elements of D, i.e. perform A = D*A
// Ignore non-diagonal elements of D
void row_scale(Matrix& A, const Matrix& D);

// Scale the columns of A using diagonal elements of D, i.e. perform A = A*D
// Ignore non-diagonal elements of D
void column_scale(Matrix& A, const Matrix& D);

}  // namespace Hatrix
