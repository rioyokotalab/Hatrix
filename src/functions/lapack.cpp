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

void inverse(Matrix& A) {
  std::vector<int> ipiv(A.min_dim());
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, ipiv.data());

  // Determine the ideal workspace size.
  // LAPACKE_dgetri(LAPACK_COL_MAJOR, A.cols, &A, A.stride, ipiv.data(), );
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

std::vector<int> lup(Matrix& A) {
  std::vector<int> ipiv(A.rows);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, ipiv.data());
  return ipiv;
}

Matrix lu_solve(Matrix& A, const Matrix& b) {
  Matrix x(b);
  Matrix Ac(A);
  std::vector<int> ipiv(Ac.rows);

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, Ac.rows, Ac.cols, &Ac, A.stride, ipiv.data());
  LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', Ac.rows, b.cols, &Ac, Ac.stride, ipiv.data(),
    &x, x.stride);

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

// TODO: complete this function  get rid of return warnings. Also return empty R. Needs dummy alloc now.
std::tuple<Matrix, Matrix> qr(const Matrix& A, Lapack::QR_mode mode, Lapack::QR_ret qr_ret) {
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
      LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q.rows, A.cols, &Q, Q.stride, tau.data());
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.rows, Q.cols, &Q,
                     Q.stride, tau.data());

      return {Q, R};
    }
  }
  else {
    abort();
  }
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

std::tuple<Matrix, Matrix, Matrix, double> truncated_svd(Matrix&& A, int64_t rank) {
  Matrix Ac = std::move(A);
  return truncated_svd(Ac, rank);
}

std::tuple<Matrix, Matrix, Matrix> truncated_svd(Matrix& A, double error) {
  Matrix U(A.rows, A.min_dim());
  Matrix S(A.min_dim(), A.min_dim());
  Matrix V(A.min_dim(), A.cols);

  svd(A, U, S, V);

  int rank = 1;
  int irow = 1;
  while (rank < S.rows || S(irow, irow) > error) {
    rank += 1;
    irow += 1;
  }

  U.shrink(U.rows, rank);
  S.shrink(rank, rank);
  V.shrink(rank, V.cols);

  return {std::move(U), std::move(S), std::move(V)};
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
}  // namespace Hatrix
