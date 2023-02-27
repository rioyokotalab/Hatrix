#include "Hatrix/functions/blas.h"

#include <cassert>
#include <iostream>

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/LowRank.h"
#include "Hatrix/functions/arithmetics.h"

namespace Hatrix {

template <>
void array_copy(const float* from, float* to, int64_t size) {
  cblas_scopy(size, from, 1, to, 1);
}

template <>
void array_copy(const double* from, double* to, int64_t size) {
  cblas_dcopy(size, from, 1, to, 1);
}

template <>
void matmul(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, bool transA,
            bool transB, double alpha, double beta) {
  assert(transA ? A.cols : A.rows == C.rows);
  assert(transB ? B.rows : B.cols == C.cols);
  assert(transA ? A.rows : A.cols == transB ? B.cols : B.rows);
  cblas_sgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, C.rows, C.cols,
              transA ? A.rows : A.cols, static_cast<float>(alpha),
              &A, A.stride, &B, B.stride, static_cast<float>(beta),
              &C, C.stride);
};

template <>
void matmul(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, bool transA,
            bool transB, double alpha, double beta) {
  assert(transA ? A.cols : A.rows == C.rows);
  assert(transB ? B.rows : B.cols == C.cols);
  assert(transA ? A.rows : A.cols == transB ? B.cols : B.rows);
  cblas_dgemm(CblasColMajor, transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans, C.rows, C.cols,
              transA ? A.rows : A.cols, alpha,
              &A, A.stride, &B, B.stride, beta,
              &C, C.stride);
};

template <typename DT>
void matmul(const LowRank<DT>& A, const LowRank<DT>& B, Matrix<DT>& C, bool transA,
            bool transB, double alpha, double beta) {
  // LR LR D
  // TODO Many optimizations possible
  // Even in non-shared case, UxS, SxV may be optimized across blocks!
  Matrix<DT> Abasis_inner_matrices = matmul(
    transA ? A.V : A.U, matmul(
      matmul(
        A.S, matmul(transA ? A.U : A.V, transB ? B.V : B.U, transA, transB, alpha),
        transA, false
      ), B.S,false, transB
    ), transA, false
  );
  matmul(Abasis_inner_matrices, transB ? B.U : B.V, C, false, transB, 1, beta);
}

template <typename DT>
LowRank<DT> matmul(const LowRank<DT>& A, const LowRank<DT>& B, bool transA,
            bool transB, double alpha) {
  // LR LR LR
  // TODO Not implemented
  if (transA || transB) std::abort();
  assert(A.rank == B.rank);
  Matrix<DT> SxVxU = matmul(A.S, matmul(A.V, B.U, transA, transB, alpha));
  Matrix<DT> SxVxUxS = matmul(SxVxU, B.S);
  LowRank<DT> AxB(A.U, SxVxUxS, B.V, false);

  return AxB;
}

template <typename DT>
void matmul(const LowRank<DT>& A, const LowRank<DT>& B, LowRank<DT>& C, bool transA,
            bool transB, double alpha, double beta, bool add) {
  // LR LR LR
  // TODO Not implemented
  if (transA || transB) std::abort();
  assert(A.rank == B.rank);
  Matrix<DT> SxVxU = matmul(A.S, matmul(A.V, B.U, transA, transB, alpha));
  Matrix<DT> SxVxUxS = matmul(SxVxU, B.S);
  if (add) {
    C.S *= (DT) beta;
    LowRank<DT> AxB(A.U, SxVxUxS, B.V, false);

    C += AxB;
  } else {
    C.U.create_view(A.U);
    // move SxVxUxS to C.S
    C.S = std::move(SxVxUxS);
    C.V.create_view(B.V);
    C.cols = B.V.cols;
    C.rows = A.U.rows;
  }
}

template <typename DT>
void matmul(const LowRank<DT>& A, const Matrix<DT>& B, Matrix<DT>& C, bool transA,
            bool transB, double alpha, double beta) {
  // LR D D
  Matrix<DT> AS_basis_B = matmul(
    A.S, matmul(transA ? A.U : A.V, B, transA, transB, alpha),
    transA, false
  );
  matmul(transA ? A.V : A.U, AS_basis_B, C, transA, false, 1, beta);
}

template <typename DT>
void matmul(const Matrix<DT>& A, const LowRank<DT>& B, Matrix<DT>& C, bool transA,
            bool transB, double alpha, double beta) {
  // D LR D
  Matrix<DT> A_basis_BS = matmul(
    matmul(A, transB ? B.V : B.U, transA, transB, alpha), B.S,
    false, transB
  );
  matmul(A_basis_BS, transB ? B.U : B.V, C, false, transB, 1, beta);
}

template <typename DT>
Matrix<DT> matmul(const Matrix<DT>& A, const Matrix<DT>& B, bool transA, bool transB,
              double alpha) {
  // TODO is the same check performed by matmul anyway?
  if (transA) {
    if (transB) { assert(A.rows == B.cols); }
    else        { assert(A.rows == B.rows); }
  }
  else {
    if (transB) { assert(A.cols == B.cols); }
    else        { assert(A.cols == B.rows); }
  }

  Matrix<DT> C(transA ? A.cols : A.rows, transB ? B.rows : B.cols);
  matmul(A, B, C, transA, transB, alpha, 0);
  return C;
}

template <>
void syrk(const Matrix<float>& A, Matrix<float>& C, Mode uplo, bool transA, double alpha,
     double beta) {
  assert(C.rows == C.cols);
  cblas_ssyrk(CblasColMajor,
              uplo == Lower ? CblasLower : CblasUpper,
              transA ? CblasTrans : CblasNoTrans,
              C.rows,
              transA ? A.rows : A.cols,
              static_cast<float>(alpha),
              &A,
              A.stride,
              static_cast<float>(beta),
              &C,
              C.stride);
}

template <>
void syrk(const Matrix<double>& A, Matrix<double>& C, Mode uplo, bool transA, double alpha,
     double beta) {
  assert(C.rows == C.cols);
  cblas_dsyrk(CblasColMajor,
              uplo == Lower ? CblasLower : CblasUpper,
              transA ? CblasTrans : CblasNoTrans,
              C.rows,
              transA ? A.rows : A.cols,
              alpha,
              &A,
              A.stride,
              beta,
              &C,
              C.stride);
}

template <>
void triangular_matmul(const Matrix<float>& A, Matrix<float>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cblas_strmm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols,
              static_cast<float>(alpha), &A, A.stride, &B, B.stride);
}

template <>
void triangular_matmul(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  assert(side == Left ? (transA ? A.rows == B.rows : A.cols == B.rows)
                      : (transA ? B.cols == A.cols : B.cols == A.rows));
  cblas_dtrmm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols,
              alpha, &A, A.stride, &B, B.stride);
}

template <typename DT>
Matrix<DT> triangular_matmul_out(const Matrix<DT>& A, const Matrix<DT>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha) {
  Matrix<DT> C(B);
  triangular_matmul(A, C, side, uplo, transA, diag, alpha);
  return C;
}

template <>
void solve_triangular(const Matrix<float>& A, Matrix<float>& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cblas_strsm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols,
              static_cast<float>(alpha), &A, A.stride, &B, B.stride);
}

template <>
void solve_triangular(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                      bool diag, bool transA, double alpha) {
  cblas_dtrsm(CblasColMajor, side == Left ? CblasLeft : CblasRight,
              uplo == Upper ? CblasUpper : CblasLower,
              transA ? CblasTrans : CblasNoTrans,
              diag ? CblasUnit : CblasNonUnit, B.rows, B.cols,
              alpha, &A, A.stride, &B, B.stride);
}

template <typename DT>
void solve_diagonal(const Matrix<DT>& D, Matrix<DT>& B, Side side, double alpha) {
  assert(side == Left ? D.cols == B.rows : B.cols == D.rows);

  DT alpha_dt = static_cast<DT>(alpha);
  for(int64_t j = 0; j < B.cols; j++) {
    for(int64_t i = 0; i < B.rows; i++) {
      B(i, j) = alpha_dt * B(i, j) / (side == Left ? D(i, i) : D(j, j));
    }
  }
}

// TODO should this be exposed via the header?
template <>
void scale(const int64_t n, float* data, const float alpha, const int64_t stride) {
   cblas_sscal(n, alpha, data, stride);
}

template <>
void scale(const int64_t n, double* data, const double alpha, const int64_t stride) {
   cblas_dscal(n, alpha, data, stride);
}

template <typename DT>
void scale(Matrix<DT>& A, const double alpha) {
  DT scale_factor = static_cast<DT>(alpha);
  for (int64_t j=0; j<A.cols; ++j) {
    scale(A.rows, &A(0, j), scale_factor, 1);
  }
}

template <typename DT>
void row_scale(Matrix<DT>& A, const Matrix<DT>& D) {
  assert(D.rows == D.cols);
  assert(D.cols == A.rows);

  for(int i = 0; i < A.rows; i++) {
    scale(A.cols, &A(i, 0), D(i, i), A.stride);
  }
}

template <typename DT>
void column_scale(Matrix<DT>& A, const Matrix<DT>& D) {
  assert(D.rows == D.cols);
  assert(D.rows == A.cols);

  for(int j = 0; j < A.cols; j++) {
    scale(A.rows, &A(0, j), D(j, j), 1);
  }
}

// explicit instantiation (these are the only available data-types)
/* Note that scalar values are passed as doubles to facilitate template argument deduction
   e.g. scale(A, 1) would lead to a conflict since 1 is interpreted as an integer
   and implicit conversions are not considered for template deductions
*/
template void array_copy(const float* from, float* to, int64_t size);
template void array_copy(const double* from, double* to, int64_t size);

template void matmul(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, bool transA,
            bool transB, double alpha, double beta);
template Matrix<float> matmul(const Matrix<float>& A, const Matrix<float>& B, bool transA,
              bool transB, double alpha);
template void matmul(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, bool transA,
            bool transB, double alpha, double beta);
template Matrix<double> matmul(const Matrix<double>& A, const Matrix<double>& B, bool transA,
              bool transB, double alpha);

template void matmul(const LowRank<float>& A, const Matrix<float>& B, Matrix<float>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const Matrix<float>& A, const LowRank<float>& B, Matrix<float>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const LowRank<float>& A, const LowRank<float>& B, Matrix<float>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const LowRank<float>& A, const LowRank<float>& B, LowRank<float>& C, bool transA,
            bool transB, double alpha, double beta, bool add);
template LowRank<float> matmul(const LowRank<float>& A, const LowRank<float>& B, bool transA,
            bool transB, double alpha);
template void matmul(const LowRank<double>& A, const Matrix<double>& B, Matrix<double>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const Matrix<double>& A, const LowRank<double>& B, Matrix<double>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const LowRank<double>& A, const LowRank<double>& B, Matrix<double>& C, bool transA,
            bool transB, double alpha, double beta);
template void matmul(const LowRank<double>& A, const LowRank<double>& B, LowRank<double>& C, bool transA,
            bool transB, double alpha, double beta, bool add);
template LowRank<double> matmul(const LowRank<double>& A, const LowRank<double>& B, bool transA,
            bool transB, double alpha);

template void syrk(const Matrix<float>& A, Matrix<float>& C, Mode uplo, bool transA, double alpha,
            double beta);
template void syrk(const Matrix<double>& A, Matrix<double>& C, Mode uplo, bool transA, double alpha,
            double beta);

template void triangular_matmul(const Matrix<float>& A, Matrix<float>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha);
template Matrix<float> triangular_matmul_out(const Matrix<float>& A, const Matrix<float>& B, Side side, Mode uplo,
			     bool transA, bool diag, double alpha);
template void triangular_matmul(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                       bool transA, bool diag, double alpha);
template Matrix<double> triangular_matmul_out(const Matrix<double>& A, const Matrix<double>& B, Side side, Mode uplo,
			     bool transA, bool diag, double alpha);

template void solve_triangular(const Matrix<float>& A, Matrix<float>& B, Side side, Mode uplo,
                      bool unit_diag, bool transA, double alpha);
template void solve_triangular(const Matrix<double>& A, Matrix<double>& B, Side side, Mode uplo,
                      bool unit_diag, bool transA, double alpha);

template void solve_diagonal(const Matrix<float>& D, Matrix<float>& B, Side side, double alpha);
template void solve_diagonal(const Matrix<double>& D, Matrix<double>& B, Side side, double alpha);

template void scale(const int64_t n, float* data, const float alpha, const int64_t stride);
template void scale(const int64_t n, double* data, const double alpha, const int64_t stride);

template void scale(Matrix<float>& A, const double alpha);
template void scale(Matrix<double>& A, const double alpha);

template void row_scale(Matrix<float>& A, const Matrix<float>& D);
template void row_scale(Matrix<double>& A, const Matrix<double>& D);

template void column_scale(Matrix<float>& A, const Matrix<float>& D);
template void column_scale(Matrix<double>& A, const Matrix<double>& D);

}  // namespace Hatrix
