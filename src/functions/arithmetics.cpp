#include "Hatrix/functions/arithmetics.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"

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
Matrix<DT> operator*(const Matrix<DT>& A, const Matrix<DT>& B) {
  return Hatrix::matmul(A, B, false, false, 1);
}

template <typename DT>
Matrix<DT>& operator*=(Matrix<DT>& A, DT alpha) {
  Hatrix::scale(A, alpha);
  return A;
}

template <typename DT>
Matrix<DT>& operator/=(Matrix<DT>& A, DT alpha) {
  Hatrix::scale(A, 1/alpha);
  return A;
}

template <typename DT>
Matrix<DT> operator*(const Matrix<DT>& A, DT alpha) {
  Matrix<DT> C(A, true);
  C *= alpha;
  return C;
}

template <typename DT>
Matrix<DT> operator*(DT alpha, const Matrix<DT>& A) {
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

template Matrix<float>& operator-=(Matrix<float>& A, const Matrix<float>& B);
template Matrix<float> operator-(const Matrix<float>& A, const Matrix<float>& B);
template Matrix<double>& operator-=(Matrix<double>& A, const Matrix<double>& B);
template Matrix<double> operator-(const Matrix<double>& A, const Matrix<double>& B);

template Matrix<float> operator*(const Matrix<float>& A, const Matrix<float>& B);
template Matrix<float>& operator*=(Matrix<float>& A, float alpha);
template Matrix<float> operator*(const Matrix<float>& A, float alpha);
template Matrix<float> operator*(float alpha, const Matrix<float>& A);
template Matrix<double> operator*(const Matrix<double>& A, const Matrix<double>& B);
template Matrix<double>& operator*=(Matrix<double>& A, double alpha);
template Matrix<double> operator*(const Matrix<double>& A, double alpha);
template Matrix<double> operator*(double alpha, const Matrix<double>& A);

template Matrix<float>& operator/=(Matrix<float>& A, float alpha);
template Matrix<double>& operator/=(Matrix<double>& A, double alpha);

template Matrix<float> abs(const Matrix<float>& A);
template Matrix<double> abs(const Matrix<double>& A);

template Matrix<float> transpose(const Matrix<float>& A);
template Matrix<double> transpose(const Matrix<double>& A);

template Matrix<float> lower_tri(const Matrix<float>& A, bool diag);
template Matrix<double> lower_tri(const Matrix<double>& A, bool diag);

template Matrix<float> upper_tri(const Matrix<float>& A, bool diag);
template Matrix<double> upper_tri(const Matrix<double>& A, bool diag);

}  // namespace Hatrix
