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

// Compute in-place inverse using GETRF + GETRI.
template <typename DT>
void inverse(Matrix<DT>& A);

// Compute the LU factorization of A and store in L and U. Over-writes A.
template <typename DT>
void lu(Matrix<DT>& A, Matrix<DT>& L, Matrix<DT>& U);

// Compute the in-place LU factorization of A. Non-pivoted CBLAS version.
template <typename DT>
void lu(Matrix<DT>& A);

// Compute the in-place LU factorization of A. Non-pivoted MKL version
template <typename DT>
void lu_nopiv(Matrix<DT>& A);

// Compute in-place Cholesky factorization fo A with LAPACK DPOTRF.
template <typename DT>
void cholesky(Matrix<DT>& A, Mode uplo);

// Compute pivoted LU factorization using LAPACK.
template <typename DT>
std::vector<int> lup(Matrix<DT>& A);

// Use getrs for solving dense matrix A w.r.t RHS b.
template <typename DT>
Matrix<DT> lu_solve(const Matrix<DT>& A, const Matrix<DT>& b);

// Use potrf for solving dense matrix A w.r.t RHS b.
template <typename DT>
Matrix<DT> cholesky_solve(const Matrix<DT>& A, const Matrix<DT>& b, const Mode uplo);

// Compute the in-place non-pivoted LDLT factorization of A.
template <typename DT>
void ldl(Matrix<DT>& A);

template <typename DT>
void qr(Matrix<DT>& A, Matrix<DT>& Q, Matrix<DT>& R);

// Return <Q, pivots>
template <typename DT>
std::tuple<Matrix<DT>, std::vector<int64_t>> pivoted_qr(const Matrix<DT>& A, int64_t rank);

template <typename DT>
std::tuple<Matrix<DT>,Matrix<DT>> pivoted_qr_nopiv_return(const Matrix<DT>& A, int64_t rank);

// Returns <Q, pivots, rank>
template <typename DT>
std::tuple<Matrix<DT>, std::vector<int64_t>, int64_t> error_pivoted_qr_max_rank(const Matrix<DT>& A,
                                                                   double error, int64_t max_rank=-1);

template <typename DT>                              
void rq(Matrix<DT>& A, Matrix<DT>& R, Matrix<DT>& Q);

// Compute the storage for Q and R automatically from mode and qr_ret values
// and return Q and R matrices.
template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>> qr(const Matrix<DT>& A,
                              Lapack::QR_mode mode,
                              Lapack::QR_ret qr_ret,
                              bool pivoted=false);

template <typename DT>
void svd(Matrix<DT>& A, Matrix<DT>& U, Matrix<DT>& S, Matrix<DT>& V);

template <typename DT>
double truncated_svd(Matrix<DT>& A, Matrix<DT>& U, Matrix<DT>& S, Matrix<DT>& V, int64_t rank);

template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>, Matrix<DT>, double> truncated_svd(Matrix<DT>& A,
                                                         int64_t rank);

template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>, Matrix<DT>, double> truncated_svd(Matrix<DT>&& A,
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
template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>, Matrix<DT>, int64_t> error_svd(Matrix<DT>& A, double eps,
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
template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>, int64_t> error_pivoted_qr(Matrix<DT>& A, double eps,
                                                     bool relative=true,
                                                     bool ret_truncated=true);

// Compute the Frobenius norm of a matrix
template <typename DT>
DT norm(const Matrix<DT>& A);

template <typename DT>
void householder_qr_compact_wy(Matrix<DT>& A, Matrix<DT>& T);
template <typename DT>
void apply_block_reflector(const Matrix<DT>& V, const Matrix<DT>& T, Matrix<DT>& C,
                           int side, bool trans);

template <typename DT>
std::tuple<Matrix<DT>, std::vector<int64_t>, int64_t> error_interpolate(Matrix<DT>& A, double error);

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
template <typename DT>
std::tuple<Matrix<DT>, Matrix<DT>> truncated_interpolate(Matrix<DT>& A, int64_t rank);

template <typename DT>
std::tuple<Matrix<DT>, std::vector<int64_t>> truncated_id_row(Matrix<DT>& A, int64_t rank);
template <typename DT>
std::tuple<Matrix<DT>, std::vector<int64_t>> error_id_row(Matrix<DT>& A, double error, bool relative);

template <typename DT>
std::vector<DT> get_eigenvalues(const Matrix<DT>& A);

//TODO should this be private?
template <typename DT>
std::tuple<int64_t, std::vector<int64_t>, std::vector<DT>>
partial_pivoted_qr(Matrix<DT>& A, const double stop_tol, bool relative);

}  // namespace Hatrix
