#pragma once
#include <cstdint>
#include <tuple>
#include <string>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

namespace Lapack {
  enum QR_mode { Full };
  enum QR_ret { QAndR, OnlyQ };
}


// Compute the LU factorization of A and store in L and U. Over-writes A.
void lu(Matrix& A, Matrix& L, Matrix& U);

// Compute the in-place LU factorization of A. Non-pivoted CBLAS version.
void lu(Matrix& A);

// Use getrs for solving dense matrix A w.r.t RHS b.
Matrix lu_solve(Matrix& A, const Matrix& b);

void qr(Matrix& A, Matrix& Q, Matrix& R);

// Compute the storage for Q and R automatically from mode and qr_ret values
// and return Q and R matrices.
std::tuple<Matrix, Matrix> qr(const Matrix& A, Lapack::QR_mode mode, Lapack::QR_ret qr_ret);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

std::tuple<Matrix, Matrix, Matrix> svd(const Matrix& A);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank);

double norm(const Matrix& A);

void householder_qr_compact_wy(Matrix& A, Matrix& T);

void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans);

}  // namespace Hatrix
