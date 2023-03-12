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
  MatrixWrapper D(_D, D_nrows, D_nrows, D_nrows);
  auto D_splits = split_dense(D,
                              D_nrows - rank_nrows,
                              D_nrows - rank_nrows);

  cholesky(D_splits[0], Hatrix::Lower);
  solve_triangular(D_splits[0], D_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  syrk(D_splits[2], D_splits[3], Hatrix::Lower, false, -1, 1);
}

void CORE_cholesky_full(int64_t D_nrows, int64_t D_ncols, double* _D) {
  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);

  cholesky(D, Hatrix::Lower);
}

void CORE_solve_triangular_full(int64_t D_dd_nrows, int64_t D_dd_ncols, double* _D_dd,
                                int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id) {
  MatrixWrapper D_dd(_D_dd, D_dd_nrows, D_dd_ncols, D_dd_nrows);
  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);

  solve_triangular(D_dd, D_id, Hatrix::Right, Hatrix::Lower, false, true, 1.0);
}

void CORE_syrk_full(int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id,
                    int64_t D_ij_nrows, int64_t D_ij_ncols, double* _D_ij) {
  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);
  MatrixWrapper D_ij(_D_ij, D_ij_nrows, D_ij_ncols, D_ij_nrows);

  syrk(D_id, D_ij, Hatrix::Lower, false, -1.0, 1.0);
}

void CORE_matmul_full(int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id,
                      int64_t D_jd_nrows, int64_t D_jd_ncols, double* _D_jd,
                      int64_t D_ij_nrows, int64_t D_ij_ncols, double* _D_ij) {

  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);
  MatrixWrapper D_jd(_D_jd, D_jd_nrows, D_jd_ncols, D_jd_nrows);
  MatrixWrapper D_ij(_D_ij, D_ij_nrows, D_ij_ncols, D_ij_nrows);

  matmul(D_id, D_jd, D_ij, false, true, -1.0, 1.0);
}

void CORE_trsm(int64_t D_rows, int64_t D_cols, int64_t D_row_rank, int64_t D_col_rank, double* _diagonal,
               int64_t O_rows, int64_t O_cols, int64_t O_row_rank, int64_t O_col_rank, double* _other,
               char which) {
  MatrixWrapper diagonal(_diagonal, D_rows, D_cols, D_rows);
  MatrixWrapper other(_other, O_rows, O_cols, O_rows);

  auto diagonal_splits = split_dense(diagonal,
                                     D_rows - D_row_rank,
                                     D_cols - D_col_rank);
  auto other_splits = split_dense(other,
                                  O_rows - O_row_rank,
                                  O_cols - O_col_rank);

  if (which == 'T') {
    solve_triangular(diagonal_splits[0], other_splits[0], Hatrix::Right, Hatrix::Lower,
                     false, true, 1.0);
    solve_triangular(diagonal_splits[0], other_splits[2], Hatrix::Right, Hatrix::Lower,
                     false, true, 1.0);
  }
  else if (which == 'B') {
    solve_triangular(diagonal_splits[0], other_splits[1], Hatrix::Left, Hatrix::Lower,
                     false, false, 1.0);
  }
}
