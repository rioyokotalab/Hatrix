#pragma once
#include <cstdint>
using std::int64_t;


namespace Hatrix {

enum {
  MaxNorm,
  OneNorm,
  InfinityNorm,
  FrobeniusNorm
};

class Matrix;

void lu(Matrix& A, Matrix& L, Matrix& U);

void qr(Matrix& A, Matrix& Q, Matrix& R);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

double norm(const Matrix& A, int norm);

} // namespace Hatrix
