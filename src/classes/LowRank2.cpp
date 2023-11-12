#include "Hatrix/classes/LowRank2.h"
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/util/matrix_generators.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <fstream>
#include <stdexcept>

namespace Hatrix {

template <typename DT>
LowRank2<DT>::LowRank2(const Matrix<DT>& A, int64_t rank, Approx scheme)
    : rows(A.rows),
      cols(A.cols),
      rank(rank) {
  if (scheme == Approx::SVD) {
    int64_t dmin = A.min_dim();
    Matrix<DT> U(A.rows, dmin);
    Matrix<DT> S(dmin, dmin);
    Matrix<DT> V(dmin, A.cols);

    this->error = truncated_svd(A, U, S, V, rank);
    this->U = std::move(U);
    this->S = std::move(S);
    this->V = std::move(V);
  } else {
    int sample_size = rank + 5;
    Matrix<DT> RN = generate_random_matrix<DT>(A.cols, sample_size);

    Matrix<DT> Y = matmul(A, RN);
    Matrix<DT> Q(Y.rows, Y.cols);
    Matrix<DT> R(Y.cols, Y.cols);
    qr(Y, Q, R);
    Matrix<DT> QtA = matmul(Q, A, true, false);

    Matrix<DT> Ub(QtA.rows, sample_size);
    this->S = Matrix<DT>(sample_size, sample_size);
    this->V = Matrix<DT>(sample_size, QtA.cols);
    svd(QtA, Ub, S, V);

    this->error = S(rank, rank);
    this->U = matmul(Q, Ub);
    this->U.shrink(U.rows, rank);
    this->S.shrink(rank, rank);
    this->V.shrink(rank, V.cols);
  }
}

template <typename DT> template <typename OT>
LowRank2<DT>::LowRank2(const LowRank2<OT>& A)
: rows(A.rows), cols(A.cols), rank(A.rank), error(A.error),
  U(A.U), S(A.S), V(A.V) {
    assert(U.cols == S.rows);
    assert(S.rows == S.cols);
    assert(S.cols == V.rows);
}

template <typename DT>
LowRank2<DT>::LowRank2(const Matrix<DT>& U, const Matrix<DT>& S, const Matrix<DT>& V, bool copy)
: rows(U.rows), cols(V.cols), rank(S.rows),
  U(U, copy), S(S, copy), V(V, copy) {
    assert(U.cols == S.rows);
    assert(S.rows == S.cols);
    assert(S.cols == V.rows);
}

template <typename DT>
LowRank2<DT>::LowRank2(Matrix<DT>&& U, Matrix<DT>&& S, Matrix<DT>&& V)
: rows(U.rows), cols(V.cols), rank(S.rows),
  U(std::move(U)), S(std::move(S)), V(std::move(V)) {
    assert(U.cols == S.rows);
    assert(S.rows == S.cols);
    assert(S.cols == V.rows);
}

template <typename DT>
Matrix<DT> LowRank2<DT>::make_dense() const {
  return matmul(this->U, matmul(this->S, this->V));
}

template <typename DT>
void LowRank2<DT>::print() const {
  std::cout<<"U:"<<std::endl;
  U.print();
  std::cout<<"S:"<<std::endl;
  S.print();
  std::cout<<"V:"<<std::endl;
  V.print();
};

template <typename DT>
void LowRank2<DT>::print_approx() const {
  std::cout<<"Singular Values:"<<std::endl;
  for (int64_t i = 0; i < rank; ++i) {
    std::cout<<S(i, i)<<std::endl;
  }
  std::cout<<"Error: "<<error<<std::endl;
};

template <typename DT>
int64_t LowRank2<DT>::get_rank(DT error) const {
  int64_t l = 0;
  int64_t r = rank - 1;
  while (l < r) {
    int64_t m = (l + r) / 2;
    if (S(m, m) < error) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return r + 1;
}

template <typename DT>
const DT& LowRank2<DT>::operator()(int64_t i, int64_t j) const {
  /*if (i >= rows || i < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected i < rows && i > 0, but got i= " +
                                std::to_string(i) + " rows= " + std::to_string(rows));
  }
  if (j >= cols || j < 0) {
    throw std::invalid_argument("Matrix#operator() -> expected j > cols && j > 0, but got j=" +
                                std::to_string(j) + " cols= " + std::to_string(cols));
  }
  return data_ptr[i + j * stride];
  */
}

/*
template <typename DT>
void Matrix<DT>::shrink(int64_t new_rows, int64_t new_cols) {
  assert(new_rows <= rows);
  assert(new_cols <= cols);
  // Only need to reorganize if the number of rows (leading dim) is reduced
  if (new_rows < rows) {
    for (int64_t j = 0; j < new_cols; ++j) {
      for (int64_t i = 0; i < new_rows; ++i) {
        data_ptr[i + j * new_rows] = (*this)(i, j);
      }
    }
  }
  rows = new_rows;
  cols = new_cols;
  stride = rows;
}
*/

// explicit instantiation (these are the only available data-types)
template class LowRank2<float>;
template class LowRank2<double>;
template LowRank2<double>::LowRank2(const LowRank2<float>&);
template LowRank2<float>::LowRank2(const LowRank2<double>&);

}  // namespace Hatrix
