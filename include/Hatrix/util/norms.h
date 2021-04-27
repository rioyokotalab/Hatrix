#pragma once


namespace Hatrix {

class Matrix;

double frobenius_norm(const Matrix& A);

double frobenius_norm_diff(const Matrix& A, const Matrix& B);

} // namespace Hatrix
