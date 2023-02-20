#pragma once
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/LowRank.h"

namespace Hatrix {

template <typename DT>
Matrix<DT>& operator+=(Matrix<DT>& A, const Matrix<DT>& B);
template <typename DT>
Matrix<DT> operator+(const Matrix<DT>& A, const Matrix<DT>& B);

template <typename DT>
Matrix<DT>& operator+=(Matrix<DT>& A, const LowRank<DT>& B);

template <typename DT>
LowRank<DT>& operator+=(LowRank<DT>& A, const LowRank<DT>& B);

template <typename DT>
Matrix<DT>& operator-=(Matrix<DT>& A, const Matrix<DT>& B);
template <typename DT>
Matrix<DT> operator-(const Matrix<DT>& A, const Matrix<DT>& B);

template <typename DT>
void operator*=(Matrix<DT>& A, const Matrix<DT>& B);
template <typename DT>
Matrix<DT> operator*(const Matrix<DT>& A, const Matrix<DT>& B);

template <typename DT>
Matrix<DT>& operator*=(Matrix<DT>& A, const DT alpha);
template <typename DT>
Matrix<DT> operator*(const Matrix<DT>& A, const DT alpha);
template <typename DT>
Matrix<DT> operator*(const DT alpha, const Matrix<DT>& A);

template <typename DT>
Matrix<DT>& operator/=(Matrix<DT>& A, const DT alpha);
template <typename DT>
Matrix<DT> operator/(const Matrix<DT>& A, const DT alpha);

template <typename DT>
Matrix<DT> abs(const Matrix<DT>& A);
template <typename DT>
Matrix<DT> transpose(const Matrix<DT>& A);
template <typename DT>
Matrix<DT> lower_tri(const Matrix<DT>& A, bool diag=false);
template <typename DT>
Matrix<DT> upper_tri(const Matrix<DT>& A, bool diag=false);

}  // namespace Hatrix
