#include "Hatrix/classes/LowRank.h"
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/functions/blas.h"

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
LowRank<DT>::LowRank(Matrix<DT>& A, int64_t rank)
    : rows(rows),
      cols(cols),
      rank(rank) {
    int64_t dmin = A.min_dim();
    Matrix<DT> U(A.rows, dmin);
    Matrix<DT> S(dmin, dmin);
    Matrix<DT> V(dmin, A.cols);

    this->error = truncated_svd(A, U, S, V, rank);
    //TODO move?
    this->U = U;
    this->S = S;
    this->V = transpose(V);
}


template <typename DT>
const DT& LowRank<DT>::operator()(int64_t i, int64_t j) const {
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
template class LowRank<float>;
template class LowRank<double>;

}  // namespace Hatrix
