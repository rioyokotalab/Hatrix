#pragma once
#include <cstdint>

namespace Hatrix {

class Matrix;

void lu(Matrix& A, Matrix& L, Matrix& U);

void qr(Matrix& A, Matrix& Q, Matrix& R);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

double norm(const Matrix& A);

void householder_qr_compact_wy(Matrix& A, Matrix& T);

void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans);

}  // namespace Hatrix
