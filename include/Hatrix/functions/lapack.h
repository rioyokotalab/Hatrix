#pragma once
#include <cstdint>
#include <tuple>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/math_common.h"

namespace Hatrix {

namespace Lapack {
  enum QR_mode { Full };
  enum QR_ret { QAndR, OnlyQ };
}

// Call LASWP for swapping the rows.
void swap_rows(Matrix& A, std::vector<int> pivots);

// Compute condition number with dgecon.
double cond(const Matrix& A);

// Compute in-place inverse using GETRF + GETRI.
void inverse(Matrix& A);

// Compute the LU factorization of A and store in L and U. Over-writes A.
void lu(Matrix& A, Matrix& L, Matrix& U);

// Compute the in-place LU factorization of A. Non-pivoted CBLAS version.
void lu(Matrix& A);

// Compute in-place Cholesky factorization fo A with LAPACK DPOTRF.
void cholesky(Matrix& A, Mode uplo);

std::vector<int> cholesky_piv(Matrix& A, Mode uplo);

// Compute pivoted LU factorization using LAPACK.
std::vector<int> lup(Matrix& A);

// Use getrs for solving dense matrix A w.r.t RHS b.
Matrix lu_solve(const Matrix& A, const Matrix& b);

// Use potrf for solving dense matrix A w.r.t RHS b.
Matrix cholesky_solve(const Matrix& A, const Matrix& b, const Mode uplo);

// Compute the in-place non-pivoted LDLT factorization of A.
void ldl(Matrix& A);

void qr(Matrix& A, Matrix& Q, Matrix& R);

// Return <Q, pivots>
std::tuple<Matrix, std::vector<int64_t>> pivoted_qr(const Matrix& A, int64_t rank);

std::tuple<Matrix,Matrix> pivoted_qr_nopiv_return(const Matrix& A, int64_t rank);

// Returns <Q, pivots, rank> by computing the first 'rank' orthogonal factors of A
// until the error 'error' has been met.
std::tuple<Matrix, std::vector<int64_t>, int64_t> error_pivoted_qr_max_rank(const Matrix& A,
                                                                   double error,
                                                                   int64_t max_rank=-1);
void rq(Matrix& A, Matrix& R, Matrix& Q);

// Compute the storage for Q and R automatically from mode and qr_ret values
// and return Q and R matrices.
std::tuple<Matrix, Matrix> qr(const Matrix& A,
                              Lapack::QR_mode mode,
                              Lapack::QR_ret qr_ret,
                              bool pivoted=false);

std::vector<double> get_singular_values(Matrix& A);

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V);

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank);

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix&& A,
                                                         int64_t rank);

/*
  Compute truncated SVD for given accuracy threshold, such that
    |A-USV|_2 <= eps * |A|_2  ,if relative = true
    |A-USV|_2 <= eps          ,otherwise
  where |A|_2 is the 2-norm of the matrix A
  @param A The matrix to be approximated
  @param eps The desired accuracy threshold
  @param relative If true use relative error, otherwise use absolute error
  @param ret_truncated If true return truncated U,S,V, otherwise return full, non-truncated U,S,V
  @return tuple(U, S, V, rank)
*/
std::tuple<Matrix, Matrix, Matrix, int64_t> error_svd(Matrix& A, double eps,
                                                      bool relative=true,
                                                      bool ret_truncated=true);

/*
  Compute truncated pivoted QR that stops as soon as the desired accuracy
  is reached, such that
    |A-QR|_F <= eps * |A|_F,  ,if relative = true
    |A-QR|_F <= eps           ,otherwise
  where |A|_F is the Frobenius norm of the matrix A
  This is a modification of LAPACK's dgeqp3 routine
  @param A The matrix to be approximated
  @param eps The desired accuracy threshold
  @param relative If true use relative error, otherwise use absolute error
  @param ret_truncated If true return truncated Q and R, otherwise return full, non-truncated Q and R
  @return tuple(Q, R, rank)
*/
std::tuple<Matrix, Matrix, int64_t> error_pivoted_qr(Matrix& A, double eps,
                                                     bool relative=true,
                                                     bool ret_truncated=true);

// Compute the Frobenius norm of a matrix
double norm(const Matrix& A);

// Compute the Frobenius norm of a matrix
double one_norm(const Matrix& A);

void householder_qr_compact_wy(Matrix& A, Matrix& T);
void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans);

std::tuple<Matrix, std::vector<int64_t>, int64_t> error_interpolate(Matrix& A, double error);

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
std::tuple<Matrix, Matrix> truncated_interpolate(Matrix& A, int64_t rank);

std::tuple<Matrix, std::vector<int64_t>> truncated_id_row(Matrix& A, int64_t rank);
std::tuple<Matrix, std::vector<int64_t>> error_id_row(Matrix& A, double error, bool relative);

std::vector<double> get_eigenvalues(const Matrix& A);

std::tuple<Matrix, Matrix>
truncated_pivoted_qr(Matrix& A, const int64_t rank);

}  // namespace Hatrix
