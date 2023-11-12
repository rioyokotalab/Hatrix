#include "Hatrix/functions/arithmetics.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/LowRank.h"
#include "Hatrix/classes/LowRank2.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/functions/lapack.h"

namespace Hatrix {

template <typename DT>
Matrix<DT>& operator+=(Matrix<DT>& A, const Matrix<DT>& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A(i, j) += B(i, j);

  return A;
}

template <typename DT>
Matrix<DT>& operator+=(Matrix<DT>& A, const LowRank<DT>& B) {
  A += B.make_dense();
  return A;
}

template <typename DT>
Matrix<DT>& operator+=(Matrix<DT>& A, const LowRank2<DT>& B) {
  A += B.make_dense();
  return A;
}

template <typename DT>
LowRank<DT>& operator+=(LowRank<DT>& A, const LowRank<DT>& B) {
 
  // Merge Column Basis
  assert(A.U.rows == B.U.rows);
  int64_t rank = A.U.cols;
  assert(rank == B.U.cols);

  Matrix<DT> InnerU(rank+rank, rank);
  std::vector<Matrix<DT>> InnerH = InnerU.split(2, 1);
  matmul(A.U, B.U, InnerH[0], true, false, 1, 0); 
  
  Matrix<DT> Bu_AuAtBu(B.U, true);
  matmul(A.U, InnerH[0], Bu_AuAtBu, false, false, -1, 1);
  Matrix<DT> OuterU_1(A.U.rows, rank);
  qr(Bu_AuAtBu, OuterU_1, InnerH[1]); 

  // changed to copy here because move would change the pre-allocated view
  Matrix<DT> OuterU_0(A.U, true);
 
  // Merge row basis
  assert(A.V.cols == B.V.cols);
  rank = A.V.rows;
  assert(rank == B.V.rows);
  Matrix<DT> InnerV(rank, rank+rank);
  InnerH = InnerV.split(1,2);
  matmul(B.V, A.V, InnerH[0], false, true, 1, 0);
  Matrix<DT> Bv_BvAvtAv(B.V, true);
  matmul(InnerH[0], A.V, Bv_BvAvtAv, false, false, -1, 1);
  Matrix<DT> OuterV_1(rank, A.V.cols);
  rq(Bv_BvAvtAv, InnerH[1], OuterV_1);

  // changed to copy here because move would change the pre-allocated view
  Matrix<DT> OuterV_0(A.V, true);
 
  // Merge S
  rank = A.S.rows;
  assert(rank == A.S.cols);
  Matrix<DT> InnerUBs = matmul(InnerU, B.S);
  Matrix<DT> M = matmul(InnerUBs, InnerV);
  std::vector<Matrix<DT>> MH = M.split(2,2);
  MH[0] += A.S;
  Matrix<DT> Uhat(M.rows, M.rows);
  Matrix<DT> Shat(M.rows, M.cols);
  Matrix<DT> Vhat(M.cols, M.cols);
  svd(M, Uhat, Shat, Vhat);
  Uhat.shrink(Uhat.rows, rank);
  Shat.shrink(rank, rank);
  Vhat.shrink(rank, Vhat.cols);


  //A.U = Matrix<DT>(A.rows, rank);
  //A.V = Matrix<DT>(rank, A.cols);
  // This should be save since all views point to the same S
  A.S = std::move(Shat);
  std::vector<Matrix<DT>> Uhat_split = Uhat.split(2,1);
  matmul(OuterU_0, Uhat_split[0], A.U, false, false, 1, 0);
  matmul(OuterU_1, Uhat_split[1], A.U);

  std::vector<Matrix<DT>> Vhat_split = Vhat.split(1,2);
  matmul(Vhat_split[0], OuterV_0, A.V, false, false, 1, 0);
  matmul(Vhat_split[1], OuterV_1, A.V);
  
  return A;
}

template <typename DT>
LowRank2<DT>& operator+=(LowRank2<DT>& A, const LowRank2<DT>& B) {
 
  // Merge Column Basis
  assert(A.U.rows == B.U.rows);
  int64_t rank = A.U.cols;
  assert(rank == B.U.cols);

  Matrix<DT> InnerU(rank+rank, rank);
  std::vector<Matrix<DT>> InnerH = InnerU.split(2, 1);
  matmul(A.U, B.U, InnerH[0], true, false, 1, 0); 
  
  Matrix<DT> Bu_AuAtBu(B.U, true);
  matmul(A.U, InnerH[0], Bu_AuAtBu, false, false, -1, 1);
  Matrix<DT> OuterU_1(A.U.rows, rank);
  qr(Bu_AuAtBu, OuterU_1, InnerH[1]); 

  // changed to copy here because move would change the pre-allocated view
  Matrix<DT> OuterU_0(A.U, true);
 
  // Merge row basis
  assert(A.V.cols == B.V.cols);
  rank = A.V.rows;
  assert(rank == B.V.rows);
  Matrix<DT> InnerV(rank, rank+rank);
  InnerH = InnerV.split(1,2);
  matmul(B.V, A.V, InnerH[0], false, true, 1, 0);
  Matrix<DT> Bv_BvAvtAv(B.V, true);
  matmul(InnerH[0], A.V, Bv_BvAvtAv, false, false, -1, 1);
  Matrix<DT> OuterV_1(rank, A.V.cols);
  rq(Bv_BvAvtAv, InnerH[1], OuterV_1);

  // changed to copy here because move would change the pre-allocated view
  Matrix<DT> OuterV_0(A.V, true);
 
  // Merge S
  rank = A.S.rows;
  assert(rank == A.S.cols);
  Matrix<DT> InnerUBs = matmul(InnerU, B.S);
  Matrix<DT> M = matmul(InnerUBs, InnerV);
  std::vector<Matrix<DT>> MH = M.split(2,2);
  MH[0] += A.S;
  Matrix<DT> Uhat(M.rows, M.rows);
  Matrix<DT> Shat(M.rows, M.cols);
  Matrix<DT> Vhat(M.cols, M.cols);
  svd(M, Uhat, Shat, Vhat);
  // TODO, this is where the recompression takes place and the ranks increase
  std::cout<<"Addition: " << A.error << "(before) vs ";
  double expected_err = 0;
  for (int64_t k = rank; k < Shat.min_dim(); ++k)
    expected_err += Shat(k, k) * Shat(k, k);
  std::cout << std::sqrt(expected_err) << "(afer)" <<std::endl;
  Uhat.shrink(Uhat.rows, rank);
  Shat.shrink(rank, rank);
  Vhat.shrink(rank, Vhat.cols);


  //A.U = Matrix<DT>(A.rows, rank);
  //A.V = Matrix<DT>(rank, A.cols);
  // This should be save since all views point to the same S
  A.S = std::move(Shat);
  std::vector<Matrix<DT>> Uhat_split = Uhat.split(2,1);
  matmul(OuterU_0, Uhat_split[0], A.U, false, false, 1, 0);
  matmul(OuterU_1, Uhat_split[1], A.U);

  std::vector<Matrix<DT>> Vhat_split = Vhat.split(1,2);
  matmul(Vhat_split[0], OuterV_0, A.V, false, false, 1, 0);
  matmul(Vhat_split[1], OuterV_1, A.V);
  
  return A;
}

template <typename DT>
Matrix<DT> operator+(const Matrix<DT>& A, const Matrix<DT>& B) {
  Matrix<DT> C(A, true);
  C += B;
  return C;
}

template <typename DT>
Matrix<DT>& operator-=(Matrix<DT>& A, const Matrix<DT>& B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A(i, j) -= B(i, j);

  return A;
}

template <typename DT>
Matrix<DT> operator-(const Matrix<DT>& A, const Matrix<DT>& B) {
  Matrix<DT> C(A, true);
  C -= B;
  return C;
}

template <typename DT>
void operator*=(Matrix<DT>& A, const Matrix<DT>& B) {
  assert (A.cols == B.rows);
  assert (B.rows == B.cols);

  A = Hatrix::matmul(A, B, false, false, 1);
}

template <typename DT>
Matrix<DT> operator*(const Matrix<DT>& A, const Matrix<DT>& B) {
  return Hatrix::matmul(A, B, false, false, 1);
}

template <typename DT>
Matrix<DT>& operator*=(Matrix<DT>& A, const DT alpha) {
  Hatrix::scale(A, alpha);
  return A;
}

template <typename DT>
Matrix<DT>& operator/=(Matrix<DT>& A, const DT alpha) {
  Hatrix::scale(A, 1/alpha);
  return A;
}

template <typename DT>
Matrix<DT> operator/(const Matrix<DT>& A, const DT alpha) {
  Matrix<DT> C(A, true);
  C/=alpha;
  return C;
}

template <typename DT>
Matrix<DT> operator*(const Matrix<DT>& A, const DT alpha) {
  Matrix<DT> C(A, true);
  C *= alpha;
  return C;
}

template <typename DT>
Matrix<DT> operator*(const DT alpha, const Matrix<DT>& A) {
  Matrix<DT> C(A, true);
  C *= alpha;
  return C;
}

template <typename DT>
Matrix<DT> abs(const Matrix<DT>& A) {
  Matrix<DT> A_abs(A.rows, A.cols);
  for (int64_t j = 0; j < A.cols; ++j)
    for (int64_t i = 0; i < A.rows; ++i) A_abs(i, j) = std::abs(A(i, j));

  return A_abs;
}

template <typename DT>
Matrix<DT> transpose(const Matrix<DT>& A) {
  Matrix<DT> A_trans(A.cols, A.rows);
  for (int64_t i = 0; i < A_trans.rows; i++)
    for (int64_t j = 0; j < A_trans.cols; j++) A_trans(i, j) = A(j, i);

  return A_trans;
}

template <typename DT>
Matrix<DT> lower_tri(const Matrix<DT>& A, bool diag) {
  Matrix<DT> A_lower(A.rows, A.cols);
  for(int64_t i = 0; i < A.rows; i++) {
    for(int64_t j = 0; j < std::min(i+1, A.cols); j++) {
      A_lower(i, j) = (i == j && diag ? 1. : A(i, j));
    }
  }

  return A_lower;
}

template <typename DT>
Matrix<DT> upper_tri(const Matrix<DT>& A, bool diag) {
  Matrix<DT> A_upper(A.rows, A.cols);
  for(int64_t i = 0; i < A.rows; i++) {
    for(int64_t j = i; j < A.cols; j++) {
      A_upper(i, j) = (i == j && diag ? 1. : A(i, j));
    }
  }
  return A_upper;
}

// explicit instantiation (these are the only available data-types)
template Matrix<float>& operator+=(Matrix<float>& A, const Matrix<float>& B);
template Matrix<float> operator+(const Matrix<float>& A, const Matrix<float>& B);
template Matrix<double>& operator+=(Matrix<double>& A, const Matrix<double>& B);
template Matrix<double> operator+(const Matrix<double>& A, const Matrix<double>& B);

template Matrix<float>& operator+=(Matrix<float>& A, const LowRank<float>& B);
template Matrix<double>& operator+=(Matrix<double>& A, const LowRank<double>& B);

template Matrix<float>& operator+=(Matrix<float>& A, const LowRank2<float>& B);
template Matrix<double>& operator+=(Matrix<double>& A, const LowRank2<double>& B);

template LowRank<float>& operator+=(LowRank<float>& A, const LowRank<float>& B);
template LowRank<double>& operator+=(LowRank<double>& A, const LowRank<double>& B);

template LowRank2<float>& operator+=(LowRank2<float>& A, const LowRank2<float>& B);
template LowRank2<double>& operator+=(LowRank2<double>& A, const LowRank2<double>& B);

template Matrix<float>& operator-=(Matrix<float>& A, const Matrix<float>& B);
template Matrix<float> operator-(const Matrix<float>& A, const Matrix<float>& B);
template Matrix<double>& operator-=(Matrix<double>& A, const Matrix<double>& B);
template Matrix<double> operator-(const Matrix<double>& A, const Matrix<double>& B);

template void operator*=(Matrix<float>& A, const Matrix<float>& B);
template Matrix<float> operator*(const Matrix<float>& A, const Matrix<float>& B);
template void operator*=(Matrix<double>& A, const Matrix<double>& B);
template Matrix<double> operator*(const Matrix<double>& A, const Matrix<double>& B);

template Matrix<float>& operator*=(Matrix<float>& A, const float alpha);
template Matrix<float> operator*(const Matrix<float>& A, const float alpha);
template Matrix<float> operator*(const float alpha, const Matrix<float>& A);
template Matrix<double>& operator*=(Matrix<double>& A, const double alpha);
template Matrix<double> operator*(const Matrix<double>& A, const double alpha);
template Matrix<double> operator*(const double alpha, const Matrix<double>& A);

template Matrix<float>& operator/=(Matrix<float>& A, const float alpha);
template Matrix<float> operator/(const Matrix<float>& A, const float alpha);
template Matrix<double>& operator/=(Matrix<double>& A, const double alpha);
template Matrix<double> operator/(const Matrix<double>& A, const double alpha);

template Matrix<float> abs(const Matrix<float>& A);
template Matrix<double> abs(const Matrix<double>& A);

template Matrix<float> transpose(const Matrix<float>& A);
template Matrix<double> transpose(const Matrix<double>& A);

template Matrix<float> lower_tri(const Matrix<float>& A, bool diag);
template Matrix<double> lower_tri(const Matrix<double>& A, bool diag);

template Matrix<float> upper_tri(const Matrix<float>& A, bool diag);
template Matrix<double> upper_tri(const Matrix<double>& A, bool diag);

}  // namespace Hatrix
