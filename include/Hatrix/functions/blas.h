#pragma once

namespace Hatrix {

enum Side { Left, Right };
enum Mode { Upper, Lower };

class Matrix;

void matmul(const Matrix& A, const Matrix& B, Matrix& C, bool transA = false,
            bool transB = false, double alpha = 1.0, double beta = 1.0);

void triangular_matmul(const Matrix& A, Matrix& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha = 1.0);

void solve_triangular(const Matrix& A, Matrix& B, Side side, Mode uplo,
                      bool diag, bool transA = false, double alpha = 1.0);

void scale(Matrix& A, double alpha);

}  // namespace Hatrix
