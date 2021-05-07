#pragma once


namespace Hatrix {

class Matrix;

double norm_bw_error(const Matrix& residual, const Matrix& A, const Matrix& x, const Matrix& b);

double comp_bw_error(const Matrix& residual, const Matrix& A, const Matrix& x, const Matrix& b);

} // namespace Hatrix
