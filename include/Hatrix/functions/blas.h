#pragma once
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/math_common.h"

namespace Hatrix {

template <typename DT>
void array_copy(const DT* from, DT* to, int64_t size);

// Perform a matrix multiplication C = beta * C + alpha * A * B.
// Optionally transpose A and B.
template <typename DT>
void matmul(const Matrix<DT>& A, const Matrix<DT>& B, Matrix<DT>& C, bool transA = false,
            bool transB = false, double alpha = 1.0, double beta = 1.0);

template <typename DT>
Matrix<DT> matmul(const Matrix<DT>& A, const Matrix<DT>& B, bool transA = false,
              bool transB = false, double alpha = 1.0);

// If transA is false:
//   C = alpha * A * A' + beta * C
// if transA is true:
//   C = alpha * A' * A + beta * C
// A has to be symmetric.
template <typename DT>
void syrk(const Matrix<DT>& A, Matrix<DT>& C, Mode uplo, bool transA, double alpha,
            double beta);


// Compute matrix product with one triangular matrix. A is the triangular matrix here.
// B = alpha * op(A) * B
// op(A) is one of op(A) = A, or op(A) = A'
template <typename DT>
void triangular_matmul(const Matrix<DT>& A, Matrix<DT>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha = 1.0);

template <typename DT>
Matrix<DT> triangular_matmul_out(const Matrix<DT>& A, const Matrix<DT>& B, Side side, Mode uplo,
			     bool transA, bool diag, double alpha = 1.0);

template <typename DT>
void solve_triangular(const Matrix<DT>& A, Matrix<DT>& B, Side side, Mode uplo,
                      bool unit_diag, bool transA = false, double alpha = 1.0);

// Solve D*X = alpha*B or X*D = alpha*B
// Ignore non-diagonal elements of D
template <typename DT>
void solve_diagonal(const Matrix<DT>& D, Matrix<DT>& B, Side side, double alpha = 1.0);

template <typename DT>
void scale(Matrix<DT>& A, double alpha);

// Scale the rows of A using diagonal elements of D, i.e. perform A = D*A
// Ignore non-diagonal elements of D
template <typename DT>
void row_scale(Matrix<DT>& A, const Matrix<DT>& D);

// Scale the columns of A using diagonal elements of D, i.e. perform A = A*D
// Ignore non-diagonal elements of D
template <typename DT>
void column_scale(Matrix<DT>& A, const Matrix<DT>& D);

}  // namespace Hatrix
