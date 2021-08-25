#pragma once
#include <cstdint>
#include <tuple>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

void lu(Matrix& A, Matrix& L, Matrix& U);

void qr(Matrix& A, Matrix& Q, Matrix& R);

// Perform pivoted QR factorization using GEQP3
std::tuple<Matrix, Matrix> pivoted_qr(Matrix& A, const int rank, const bool transpose=false);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank);

double norm(const Matrix& A);

void householder_qr_compact_wy(Matrix& A, Matrix& T);

void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans);

}  // namespace Hatrix
