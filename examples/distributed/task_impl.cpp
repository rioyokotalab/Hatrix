#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

using namespace Hatrix;


// Really dumb wrapper over Matrix.
class MatrixWrapper : public Hatrix::Matrix {
public:
  MatrixWrapper(double* data, int64_t _rows, int64_t _cols, int64_t _stride) {
    data_ptr = data;
    rows = _rows;
    cols = _cols;
    stride = _stride;
  }

  void copy_mem(const Matrix& A) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        (*this)(i, j) = A(i, j);
      }
    }
  }

  ~MatrixWrapper() {
    data_ptr = nullptr;
    rows = -1;
    cols = -1;
    stride = -1;
  }
};

void CORE_multiply_complement(int64_t D_nrows, int64_t D_ncols, int64_t D_row_rank, int64_t D_col_rank,
                              int64_t U_nrows, int64_t U_ncols, double* _D, double* _U, char which) {
  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  Matrix UF = make_complement(U);
  Matrix product;
  std::vector<Matrix> D_splits;

  if (which == 'F') {           // multiply complements from the left and right
    Matrix product = matmul(matmul(UF, D, true), UF);
    D.copy_mem(product);
  }
  else if (which == 'L') {      // left multiplication
    auto D_splits = D.split({},
                            std::vector<int64_t>(1,
                                                 D_ncols - D_col_rank));
    D_splits[1] = matmul(UF, D_splits[1], true);
  }
  else if (which == 'R') {      // right multiplication
    Matrix product = matmul(D, UF);
    D.copy_mem(product);
  }
}

void CORE_factorize_diagonal(int64_t D_nrows, int64_t rank_nrows, double *_D) {
  auto D_splits = split_dense(D,
                              D_nrows - rank_nrows,
                              D_nrows - rank_nrows);

  cholesky(D_splits[0], Hatrix::Lower);
  solve_triangular(D_splits[0], D_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  syrk(D_splits[2], D_splits[3], Hatrix::Lower, false, -1, 1);
}
