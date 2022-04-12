#pragma once
#include <cstdint>
#include <tuple>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

namespace Lapack {
  enum QR_mode { Full };
  enum QR_ret { QAndR, OnlyQ };
}

// Compute in-place inverse using GETRF + GETRI.
void inverse(Matrix& A);

// Compute the LU factorization of A and store in L and U. Over-writes A.
void lu(Matrix& A, Matrix& L, Matrix& U);

// Compute the in-place LU factorization of A. Non-pivoted CBLAS version.
void lu(Matrix& A);

// Compute pivoted LU factorization using LAPACK.
std::vector<int> lup(Matrix& A);

// Use getrs for solving dense matrix A w.r.t RHS b.
Matrix lu_solve(const Matrix& A, const Matrix& b);

// Compute the in-place non-pivoted LDLT factorization of A.
void ldl(Matrix& A);

void qr(Matrix& A, Matrix& Q, Matrix& R);

// Compute the storage for Q and R automatically from mode and qr_ret values
// and return Q and R matrices.
std::tuple<Matrix, Matrix> qr(const Matrix& A, Lapack::QR_mode mode, Lapack::QR_ret qr_ret);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix&& A,
                                                         int64_t rank);

// Compute truncated SVD for given accuracy threshold.
std::tuple<Matrix, Matrix, Matrix> error_svd(Matrix& A, double error);

double norm(const Matrix& A);

void householder_qr_compact_wy(Matrix& A, Matrix& T);

void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans);

std::tuple<Matrix, Matrix> error_interpolate(Matrix& A, bool transpose, double error);

// One-sided truncated interpolative decomposition. Refer to section 4 of
// https://amath.colorado.edu/faculty/martinss/Pubs/2004_skeletonization.pdf for the
// full algorithm.
//
// Inputs
// ------
//
// Matrix A - Matrix to perform ID on. MODIFIED IN-PLACE.
//
// Returns
// -------
//
// std::tuple<Matrix, Matrix> - The first Matrix is a (A.rows x rank) block with the interpolation
// matrix obtained from the left-sided QR decomposition of A. The second Matrix is a (rankx1) vector
// denoting the first rank pivot columns from A that are chosen as the basis vectors.
std::tuple<Matrix, Matrix> truncated_interpolate(Matrix& A, bool transpose, int64_t rank);

}  // namespace Hatrix
