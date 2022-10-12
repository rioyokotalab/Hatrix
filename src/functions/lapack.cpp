#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/math_common.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/util/matrix_generators.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>
#include <iostream>

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"

namespace Hatrix {

void inverse(Matrix& A) {
  std::vector<int> ipiv(A.min_dim());
  int info;
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRF failed in inverse().\n";
  }
  info = LAPACKE_dgetri(LAPACK_COL_MAJOR, A.rows, &A, A.stride, ipiv.data());
  if (info != 0) {
    std::cout << "DGETRI failed in inverse().\n";
  }
}

void lu(Matrix& A, Matrix& L, Matrix& U) {
  // check dimensions
  assert(L.rows == A.rows);
  assert(L.cols == U.rows && L.cols == A.min_dim());
  assert(U.cols == A.cols);

  std::vector<int> ipiv(A.min_dim());

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, ipiv.data());

  // U: set lower triangular matrix to 0
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', U.rows, U.cols, 0, 0, &L, U.stride);

  // copy out U and L
  int64_t n_diag = U.min_dim();
  for (int64_t j = 0; j < n_diag; ++j) {
    cblas_dcopy(j + 1, &A(0, j), 1, &U(0, j), 1);
    cblas_dcopy(A.rows - j, &A(j, j), 1, &L(j, j), 1);
  }

  // copy out the rest of U if trapezoidal
  if (U.cols > U.rows) {
    cblas_dcopy((U.cols - U.rows) * U.rows, &A(0, n_diag), 1, &U(0, n_diag), 1);
  }

  // L: set diagonal to 1 and upper triangular matrix to 0
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'U', L.rows, L.cols, 0, 1, &L, L.stride);
}

void lu(Matrix& A) {
  double * a = &A;
  int m = A.rows;
  int n = A.cols;
  int lda = A.stride;
  int k = std::min(A.rows, A.cols);
  for (int i = 0; i < k; i++) {
    double p = 1. / a[i + (size_t)i * lda];
    int mi = m - i - 1;
    int ni = n - i - 1;

    double* ax = a + i + (size_t)i * lda + 1;
    double* ay = a + i + (size_t)i * lda + lda;
    double* an = ay + 1;

    cblas_dscal(mi, p, ax, 1);
    cblas_dger(CblasColMajor, mi, ni, -1., ax, 1, ay, lda, an, lda);
  }
}

void cholesky(Matrix& A, Mode uplo) {
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo == Lower ? 'L' : 'U', A.rows, &A, A.stride);
}

std::vector<int> lup(Matrix& A) {
  std::vector<int> ipiv(A.rows);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, ipiv.data());
  return ipiv;
}

Matrix lu_solve(const Matrix& A, const Matrix& b) {
  Matrix x(b);
  Matrix Ac(A);
  std::vector<int> ipiv(Ac.rows);

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, Ac.rows, Ac.cols, &Ac, A.stride, ipiv.data());
  LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', Ac.rows, b.cols, &Ac, Ac.stride, ipiv.data(),
    &x, x.stride);

  return x;
}

Matrix cholesky_solve(const Matrix&A, const Matrix& b, const Mode uplo) {
  Matrix x(b);
  Matrix Ac(A);

  cholesky(Ac, uplo);
  solve_triangular(Ac, x, Hatrix::Left, uplo, false, false, 1.0);
  solve_triangular(Ac, x, Hatrix::Left, uplo, false, true, 1.0);

  return x;
}

void ldl(Matrix& A) {
  assert(A.rows == A.cols);

  double* a = &A;
  int n = A.rows;
  int lda = A.stride;
  for(int j = 0; j < n; j++) {
    double p = 1. / a[j + (size_t)j * lda];
    double* ax = a + j + 1 + (size_t)j * lda;
    int nj = n - j - 1;
    cblas_dscal(nj, p, ax, 1);

    for(int i = j + 1; i < n; i++) {
      double c = a[j + (size_t)j * lda] * a[i + (size_t)j * lda];
      double* aii = a + i + (size_t)i * lda;
      double* aij = a + i + (size_t)j * lda;
      int ni = n - i;
      cblas_daxpy(ni, -c, aij, 1, aii, 1);
    }
  }
}


void qr(Matrix& A, Matrix& Q, Matrix& R) {
  // check dimensions
  assert(Q.rows == A.rows);
  assert(Q.cols == R.rows);
  assert(R.cols == A.cols);
  assert(Q.cols <= A.rows); // Q is orthogonal bases of columns of A

  int64_t k = A.min_dim();
  std::vector<double> tau(k);
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, tau.data());
  // Copy upper triangular (or trapezoidal) part of A into R
  for (int64_t j = 0; j < R.cols; j++) {
    cblas_dcopy(std::min(j + 1, R.rows), &A(0, j), 1, &R(0, j), 1);
  }
  // Copy strictly lower triangular (or trapezoidal) part of A into Q
  for (int64_t j = 0; j < std::min(A.cols, Q.cols); j++) {
    cblas_dcopy(Q.rows - j, &A(j, j), 1, &Q(j, j), 1);
  }
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.cols, k, &Q, Q.stride, tau.data());
}

void rq(Matrix& A, Matrix& R, Matrix& Q) {
  // check dimensions
  assert(R.rows == A.rows);
  assert(R.cols == Q.rows);
  assert(Q.cols == A.cols);
  assert(Q.rows <= A.cols); // Q is orthogonal bases of rows of A

  int64_t k = A.min_dim();
  std::vector<double> tau(k);
  LAPACKE_dgerqf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, tau.data());
  // Copy upper triangular (or trapezoidal) part of A into R
  for (int64_t i = 0; i < R.rows; i++) {
    for (int64_t j = std::max(i + A.cols - A.rows, int64_t{0}); j < A.cols; j++) {
      R(i, j + R.cols - A.cols) = A(i, j);
    }
  }
  // Copy strictly lower triangular (or trapezoidal) part of A into Q
  for (int64_t i = std::max(A.rows - A.cols, int64_t{0}); i < A.rows; i++) {
    for (int64_t j = 0; j < (i + A.cols - A.rows); j++) {
      Q(i + Q.rows - A.rows, j) = A(i, j);
    }
  }
  LAPACKE_dorgrq(LAPACK_COL_MAJOR, Q.rows, Q.cols, k, &Q, Q.stride, tau.data());
}

// TODO: complete this function  get rid of return warnings.
// Also return empty R. Needs dummy alloc now.
std::tuple<Matrix, Matrix> qr(const Matrix& A, Lapack::QR_mode mode,
                              Lapack::QR_ret qr_ret, bool pivoted) {
  Matrix R(1, 1);

  if (mode == Lapack::Full) {
    if (qr_ret == Lapack::OnlyQ) {
      Matrix Q(A.rows, A.rows);
      std::vector<double> tau(Q.rows);
      for (int i = 0; i < Q.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
          Q(i, j) = A(i, j);
        }
      }
      if (pivoted) {
        std::vector<int> jpvt(A.cols);
        LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Q.rows, A.cols, &Q, Q.stride, jpvt.data(), tau.data());
      }
      else {
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q.rows, A.cols, &Q, Q.stride, tau.data());
      }

      LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.rows, Q.cols, &Q,
                     Q.stride, tau.data());

      return {Q, R};
    }
  }
  abort();
}

std::tuple<Matrix, std::vector<int64_t>> pivoted_qr(const Matrix& A, int64_t rank) {
  if (rank < 0) {
    std::invalid_argument("pivoted_qr()-> expected rank > 0, but got rank=" +
                          std::to_string(rank));
  }

  Matrix Q(A, true);
  std::vector<double> tau(Q.rows);
  std::vector<int> jpvt(Q.cols);

  LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Q.rows, Q.cols, &Q, Q.stride, jpvt.data(), tau.data());
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.min_dim(), Q.min_dim(), &Q, Q.stride, tau.data());

  Q.shrink(Q.rows, rank);

  // c-style pivots
  std::vector<int64_t> pivots(Q.cols);
  for (unsigned i = 0; i < jpvt.size(); ++i) { pivots[i] = jpvt[i] - 1; }

  return std::make_tuple(std::move(Q), std::move(pivots));
}

std::tuple<Matrix, std::vector<int64_t>, int64_t>
error_pivoted_qr(const Matrix& A, double error, int64_t max_rank) {
  Matrix Q(A, true);
  std::vector<double> tau(Q.rows);
  std::vector<int> jpvt(Q.cols);
  int64_t rank = 1;

  // if (max_rank > Q.min_dim()) {
  //   throw std::invalid_argument("error_pivoted_qr() -> max_rank <= Q.min_dim() is necessary, but got "
  //                               "max_rank= " + std::to_string(max_rank) + " Q.min_dim= " + std::to_string(Q.min_dim()));
  // }

  LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Q.rows, Q.cols, &Q, Q.stride, jpvt.data(), tau.data());
  for (int64_t i = 1; i < Q.min_dim(); ++i) {
    if ((rank >= max_rank && max_rank > 0) || std::abs(Q(i,i)) < error) {
      break;
    }
    rank++;
  }

  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.min_dim(), Q.min_dim(), &Q, Q.stride, tau.data());

  Q.shrink(Q.rows, rank);

  // c-style pivots
  std::vector<int64_t> pivots(jpvt.size());
  for (unsigned i = 0; i < jpvt.size(); ++i) { pivots[i] = jpvt[i] - 1; }

  return std::make_tuple(std::move(Q), std::move(pivots), rank);
}

void svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V) {
  // check dimensions
  assert(U.rows == A.rows);
  assert(S.cols == S.rows && S.cols == A.min_dim());
  assert(U.cols == S.cols && V.rows == S.rows);
  assert(V.cols == A.cols);

  std::vector<double> Sdiag(S.rows);
  std::vector<double> work(S.rows - 1);
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', A.rows, A.cols, &A, A.stride,
                 Sdiag.data(), &U, U.stride, &V, V.stride, work.data());
  S = 0;
  for (int64_t i = 0; i < S.rows; i++) {
    S(i, i) = Sdiag[i];
  }
}

double truncated_svd(Matrix& A, Matrix& U, Matrix& S, Matrix& V, int64_t rank) {
  assert(rank <= A.min_dim());
  svd(A, U, S, V);
  double expected_err = 0;
  for (int64_t k = rank; k < A.min_dim(); ++k)
    expected_err += S(k, k) * S(k, k);
  U.shrink(U.rows, rank);
  S.shrink(rank, rank);
  V.shrink(rank, V.cols);
  return std::sqrt(expected_err);
}

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix& A,
                                                         int64_t rank) {
  Matrix U(A.rows, A.min_dim());
  Matrix S(A.min_dim(), A.min_dim());
  Matrix V(A.min_dim(), A.cols);
  double expected_err = truncated_svd(A, U, S, V, rank);
  return std::make_tuple(std::move(U), std::move(S), std::move(V), expected_err);
}

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix&& A, int64_t rank) {
  Matrix Ac = std::move(A);
  return truncated_svd(Ac, rank);
}

std::tuple<Matrix, Matrix, Matrix, int64_t> error_svd(Matrix& A, double eps,
                                                      bool relative,
                                                      bool ret_truncated) {
  Matrix U(A.rows, A.min_dim());
  Matrix S(A.min_dim(), A.min_dim());
  Matrix V(A.min_dim(), A.cols);

  svd(A, U, S, V);

  double error = eps;
  if(relative) error *= S(0, 0);

  int64_t rank = 1;
  int64_t irow = 1;
  while (rank < S.rows && S(irow, irow) > error) {
    rank += 1;
    irow += 1;
  }

  if (ret_truncated) {
    U.shrink(U.rows, rank);
    S.shrink(rank, rank);
    V.shrink(rank, V.cols);
  }

  return std::make_tuple(std::move(U), std::move(S), std::move(V), rank);
}

std::tuple<int64_t, std::vector<int64_t>, std::vector<double>>
partial_pivoted_qr(Matrix& A, const double stop_tol, bool relative) {
  // Pointer aliases
  double* a = &A;
  const int64_t m = A.rows;
  const int64_t n = A.cols;
  const int64_t lda = A.stride;

  // Initialize variables for pivoted QR
  double error = stop_tol;
  if(relative) error *= norm(A);
  const double tol = LAPACKE_dlamch('e');
  const double tol3z = std::sqrt(tol);
  const int min_dim = A.min_dim();
  std::vector<double> tau(min_dim, 0);
  std::vector<int64_t> ipiv(n, 0);
  std::vector<double> cnorm(n, 0);
  std::vector<double> partial_cnorm(n, 0);
  for(int64_t j = 0; j < n; j++) {
    ipiv[j] = j;
    cnorm[j] = cblas_dnrm2(m, a + j * lda, 1);
    partial_cnorm[j] = cnorm[j];
  }

  // Begin pivoted QR
  int64_t r = 0;
  double max_cnorm = *std::max_element(cnorm.begin(), cnorm.end());
  // Handle zero matrix case
  if(max_cnorm <= tol) {
    return std::make_tuple(0, std::move(ipiv), std::move(tau));
  }
  while((r < min_dim) && (max_cnorm > error)) {
    // Select pivot column and swap
    const int64_t k = std::max_element(cnorm.begin() + r, cnorm.end()) - cnorm.begin();
    cblas_dswap(m, a + r * lda, 1, a + k * lda, 1);
    std::swap(cnorm[r], cnorm[k]);
    std::swap(partial_cnorm[r], partial_cnorm[k]);
    std::swap(ipiv[r], ipiv[k]);

    // Generate householder reflector to annihilate A(r+1:m, r)
    double *arr = a + r + (r * lda);
    if(r < (m - 1)) {
      LAPACKE_dlarfg(m-r, arr, arr+1, 1, &tau[r]);
    }
    else {
      LAPACKE_dlarfg(1, arr, arr, 1, &tau[r]);
    }

    if(r < (min_dim - 1)) {
      // Apply reflector to A(r:m,r+1:n) from left
      const double _arr = *arr;
      *arr = 1.0;
      // w = A(r:m, r+1:n)^T * v
      std::vector<double> w(n-r-1, 0);
      double *arj = a + r + (r+1) * lda;
      cblas_dgemv(CblasColMajor, CblasTrans,
                  m-r, n-r-1, 1, arj, lda, arr, 1, 0, &w[0], 1);
      // A(r:m,r+1:n) = A(r:m,r+1:n) - tau * v * w^T
      cblas_dger(CblasColMajor, m-r, n-r-1, -tau[r], arr, 1, &w[0], 1, arj, lda);
      *arr = _arr;
    }
    // Update partial column norm
    for(int64_t j = r + 1; j < n; j++) {
      // See LAPACK Working Note 176 (Section 3.2.1) for detail
      // https://netlib.org/lapack/lawnspdf/lawn176.pdf
      if(cnorm[j] != 0.0) {
        double temp = std::fabs(A(r, j)/cnorm[j]);
        temp = std::fmax(0.0, (1-temp)*(1+temp));
        const double temp2 =
            temp * (cnorm[j]/partial_cnorm[j]) * (cnorm[j]/partial_cnorm[j]);
        if(temp2 > tol3z) {
          cnorm[j] = cnorm[j] * std::sqrt(temp);
        }
        else {
          if(r < (m-1)) {
            cnorm[j] = cblas_dnrm2(m-r-1, a+(r+1)+j*lda, 1);
            partial_cnorm[j] = cnorm[j];
          }
          else {
            cnorm[j] = 0.0;
            partial_cnorm[j] = 0.0;
          }
        }
      }
    }
    r++;
    max_cnorm = *std::max_element(cnorm.begin() + r, cnorm.end());
  }
  return std::make_tuple(std::move(r), std::move(ipiv), std::move(tau));
}

std::tuple<Matrix, Matrix, int64_t> error_pivoted_qr(Matrix& A, double eps,
                                                     bool relative,
                                                     bool ret_truncated) {
  const int64_t m = A.rows;
  const int64_t n = A.cols;
  int64_t rank;
  std::vector<int64_t> ipiv;
  std::vector<double> tau;
  std::tie(rank, ipiv, tau) = partial_pivoted_qr(A, eps, relative);
  // Handle zero matrix case
  if(rank == 0) {
    Matrix Q = generate_identity_matrix(m, m);
    Matrix R(m, n);
    if (ret_truncated) {
      Q.shrink(m, 1);
      R.shrink(1, n);
    }
    return std::make_tuple(std::move(Q), std::move(R), 1);
  }
  // Construct full Q
  Matrix Q(m, m);
  // Copy strictly lower triangular (or trapezoidal) part of A into Q
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < std::min(i, rank); j++) {
      Q(i, j) = A(i, j);
    }
  }
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.cols, rank, &Q, Q.stride, &tau[0]);
  // Construct full R
  Matrix R(m, n);
  // Copy first m rows of upper triangular part of A into R
  for(int64_t i = 0; i < m; i++) {
    for(int j = i; j < n; j++) {
      R(i, j) = A(i, j);
    }
  }
  // Permute columns of R
  std::vector<int> ipivT(ipiv.size(), 0);
  for(int64_t i = 0; i < ipiv.size(); i++) ipivT[ipiv[i]] = i;
  Matrix RP(R);
  for(int64_t i = 0; i < R.rows; i++) {
    for(int64_t j = 0; j < R.cols; j++) {
      RP(i, j) = R(i, ipivT[j]);
    }
  }
  if (ret_truncated) {
    Q.shrink(m, rank);
    RP.shrink(rank, n);
  }
  return std::make_tuple(std::move(Q), std::move(RP), std::move(rank));
}

double norm(const Matrix& A) {
  return LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', A.rows, A.cols, &A, A.stride);
}

void householder_qr_compact_wy(Matrix& A, Matrix& T) {
  assert(T.rows == T.cols);
  assert(T.cols == A.cols);
  LAPACKE_dgeqrt3(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, &T, T.stride);
}

void apply_block_reflector(const Matrix& V, const Matrix& T, Matrix& C,
                           int side, bool trans) {
  assert(V.cols == T.rows);
  assert(T.rows == T.cols);
  assert(V.rows == (side == Left ? C.rows : C.cols));
  LAPACKE_dlarfb(LAPACK_COL_MAJOR, side == Left ? 'L' : 'R', trans ? 'T' : 'N',
                 'F', 'C', C.rows, C.cols, T.cols, &V, V.stride, &T, T.stride,
                 &C, C.stride);
}

void solve_r_block(Matrix& interp, const Matrix& A, const int64_t rank) {
  Matrix R11(rank, rank), R12(rank, A.cols - rank);
  // copy
  for (int i = 0; i < rank; ++i) {
    for (int j = i; j < rank; ++j) {
      R11(i, j) = A(i, j);
    }
  }

  for (int i = 0; i < R12.rows; ++i) {
    for (int j = 0; j < R12.cols; ++j) {
      R12(i, j) = A(i, j + rank);
    }
  }

  solve_triangular(R11, R12, Left, Upper, false, false, 1.0);

  // Copy the interpolation matrix from TRSM into the
  for (int i = 0; i < rank; ++i) {
    interp(i, i) = 1.0;
  }
  for (int i = 0; i < rank; ++i) {
    for (int j = 0; j < R12.cols; ++j) {
      interp(j + rank, i) = R12(i, j);
    }
  }
}

std::tuple<Matrix, std::vector<int64_t>, int64_t> error_interpolate(Matrix& A, double error) {
  std::vector<double> tau(std::min(A.rows, A.cols));
  std::vector<int> jpvt(A.cols);
  LAPACKE_dgeqp3(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, jpvt.data(), tau.data());

  int64_t min_dim = A.min_dim();
  int64_t rank = 1;
  // find the right rank for this.
  for (int64_t i = 1; i < min_dim; ++i) {
    if (std::abs(A(i, i)) < error) { break; }
    rank++;
  }

  Matrix interp(A.cols, rank);
  solve_r_block(interp, A, rank);
  // Bring pivots in C-style.
  std::vector<int64_t> c_pivots(jpvt.size());
  for (unsigned i = 0; i < jpvt.size(); ++i) {
    c_pivots[i] = jpvt[i] - 1;
  }

  return std::make_tuple(std::move(interp), std::move(c_pivots), rank);
}

std::tuple<Matrix, Matrix> truncated_interpolate(Matrix& A, int64_t rank) {
  Matrix interp(A.rows, rank), pivots(A.cols, 1);
  std::vector<double> tau(std::min(A.rows, A.cols));
  std::vector<int> jpvt(A.cols);
  Matrix R11(rank, rank), R12(rank, A.cols - rank);

  LAPACKE_dgeqp3(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, jpvt.data(), tau.data());
  solve_r_block(interp, A, rank);
  for (int i = 0; i < A.cols; ++i) {
    pivots(i, 0) = jpvt[i];
  }
  return std::make_tuple(std::move(interp), std::move(pivots));
}

std::tuple<Matrix, std::vector<int64_t>> truncated_id_row(Matrix& A, int64_t rank) {
  assert(rank <= A.min_dim());
  Matrix ATrans = transpose(A);
  std::vector<double> tau(ATrans.min_dim());
  std::vector<int> jpvt(ATrans.cols);
  LAPACKE_dgeqp3(LAPACK_COL_MAJOR, ATrans.rows, ATrans.cols, &ATrans, ATrans.stride, jpvt.data(), tau.data());

  Matrix U(ATrans.cols, rank);
  solve_r_block(U, ATrans, rank);
  std::vector<int64_t> skel_rows(jpvt.size()), U_rows(jpvt.size());
  for (int64_t i = 0; i < skel_rows.size(); i++) {
    skel_rows[i] = jpvt[i] - 1;
    U_rows[skel_rows[i]] = i;
  }
  // Permute rows of U
  Matrix PU(U.rows, U.cols);
  for (int64_t i = 0; i < PU.rows; i++) {
    const auto row = U_rows[i];
    for (int64_t j = 0; j < PU.cols; j++) {
      PU(i, j) = U(row, j);
    }
  }
  return std::make_tuple(std::move(PU), std::move(skel_rows));
}

std::tuple<Matrix, std::vector<int64_t>> error_id_row(Matrix& A, double eps, bool relative) {
  // Perform partial pivoted qr on A^T
  Matrix ATrans = transpose(A);
  int64_t rank;
  std::vector<int64_t> skel_rows;
  std::vector<double> tau;
  std::tie(rank, skel_rows, tau) = partial_pivoted_qr(ATrans, eps, relative);
  // Handle zero matrix case
  if (rank == 0) {
    rank = 1;
    Matrix U = generate_identity_matrix(ATrans.cols, rank);
    return std::make_tuple(std::move(U), std::move(skel_rows));
  }
  // Construct interpolation matrix U
  Matrix U(ATrans.cols, rank);
  solve_r_block(U, ATrans, rank);
  // Permute rows of U
  std::vector<int64_t> U_rows(skel_rows.size());
  for (int64_t i = 0; i < skel_rows.size(); i++) {
    U_rows[skel_rows[i]] = i;
  }
  Matrix PU(U.rows, U.cols);
  for (int64_t i = 0; i < PU.rows; i++) {
    const auto row = U_rows[i];
    for (int64_t j = 0; j < PU.cols; j++) {
      PU(i, j) = U(row, j);
    }
  }
  return std::make_tuple(std::move(PU), std::move(skel_rows));
}

std::vector<double> get_eigenvalues(const Matrix& A) {
  assert(A.rows == A.cols);
  Matrix Ac(A);
  std::vector<double> eigv(Ac.rows, 0);
  LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', Ac.rows, &Ac, Ac.stride, eigv.data());
  return eigv;
}

}  // namespace Hatrix
