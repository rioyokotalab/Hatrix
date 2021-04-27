#pragma once
#include <cstdint>
using std::int64_t;


namespace Hatrix {

class Matrix;

void lu(Matrix& A, Matrix& L, Matrix& U);

void lup(Matrix& A, Matrix& L, Matrix& U, Matrix& P);

void qr(Matrix& A, Matrix& Q, Matrix& R);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

void gesv_IR(Matrix &A, Matrix &b, int64_t max_iter);

} // namespace Hatrix
