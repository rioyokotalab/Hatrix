#pragma once
#include <cstdint>
using std::int64_t;


namespace Hatrix {

enum class Norm {
  MaxNorm = 'M',
  OneNorm = 'O',
  InfinityNorm  = 'I',
  FrobeniusNorm = 'F'
};

class Matrix;

void lu(Matrix& A, Matrix& L, Matrix& U);

void qr(Matrix& A, Matrix& Q, Matrix& R);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

double calc_norm(const Matrix& A, Norm norm);

void householder_qr_compact_wy(Matrix& A, Matrix& T);

void apply_block_reflector(
  const Matrix& V, const Matrix& T, Matrix& C,
  int side, bool trans
);

} // namespace Hatrix
