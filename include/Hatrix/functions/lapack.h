#pragma once
#include <cstdint>
using std::uint64_t;


namespace Hatrix {

class Matrix;

void lu(Matrix& A, Matrix& L, Matrix& U);

void qr(Matrix& A, Matrix& Q, Matrix& R);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, uint64_t rank);

} // namespace Hatrix
