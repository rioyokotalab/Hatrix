#include "Hatrix/functions/lapack.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

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

void qr(Matrix& A, Matrix& Q, Matrix& R) {
  // check dimensions
  assert(Q.rows == A.rows);
  assert(Q.cols == R.rows);
  assert(R.cols == A.cols);

  int64_t k = A.min_dim();
  std::vector<double> tau(k);
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, tau.data());
  // Copy upper triangular (or trapezoidal) part of A to R
  for (int64_t j = 0; j < R.cols; j++) {
    cblas_dcopy(std::min(j + 1, R.rows), &A(0, j), 1, &R(0, j), 1);
  }
  // Copy lower triangular of A to Q
  for (int64_t j = 0; j < std::min(A.cols, Q.cols); j++) {
    cblas_dcopy(Q.rows - j, &A(j, j), 1, &Q(j, j), 1);
  }
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.cols, k, &Q, Q.stride, tau.data());
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
  return {std::move(U), std::move(S), std::move(V), expected_err};
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

std::tuple<Matrix, Matrix> pivoted_qr(Matrix& A, const int rank, const bool transpose) {
  int Q_rows = transpose ? A.cols : A.rows;
  int R_cols = transpose ? A.rows : A.cols;
  Matrix Q(Q_rows, rank), R(rank, R_cols);
  std::vector<double> tau(std::max(A.rows, A.cols));
  std::vector<int> jpvt(R_cols);
  for (int i = 0; i < std::min(Q.rows, Q.cols); ++i) {
    Q(i, i) = 1.0;
  }

  LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, Q_rows, R_cols, &A, A.stride, jpvt.data(), tau.data());

  return {Q, R};
}

}  // namespace Hatrix
