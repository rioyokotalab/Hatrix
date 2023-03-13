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

void CORE_copy_blocks(bool copy_dense,
                      double *_D_unelim,
                      int64_t D_unelim_rows, int64_t D_unelim_cols, int64_t D_unelim_row_rank, int64_t D_unelim_col_rank,
                      double *_D_c1c2,
                      int64_t D_c1c2_rows, int64_t D_c1c2_cols, int64_t D_c1c2_row_rank, int64_t D_c1c2_col_rank,
                      int D_unelim_split_index) {
  MatrixWrapper D_unelim(_D_unelim, D_unelim_rows, D_unelim_cols, D_unelim_rows);
  MatrixWrapper D_c1c2(_D_c1c2, D_c1c2_rows, D_c1c2_cols, D_c1c2_rows);

  auto D_unelim_splits = split_dense(D_unelim,
                                     D_unelim_row_rank,
                                     D_unelim_col_rank);

  if (copy_dense) {
    auto D_c1c2_splits = split_dense(D_c1c2,
                                     D_c1c2_rows - D_c1c2_row_rank,
                                     D_c1c2_cols - D_c1c2_col_rank);
    D_unelim_splits[D_unelim_split_index] = D_c1c2_splits[3];
  }
  else {
    assert(D_c1c2_row_rank == -1);
    assert(D_c1c2_col_rank == -1);

    D_unelim_splits[D_unelim_split_index] = D_c1c2;
  }
}


void CORE_nb_nb_fill_in(int64_t D_i_block_rows, int64_t D_i_block_cols,int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                        double *_D_i_block,
                        int64_t D_j_block_rows, int64_t  D_j_block_cols,int64_t D_j_block_row_rank,int64_t D_j_block_col_rank,
                        double *_D_j_block,
                        int64_t F_ij_rows, int64_t F_ij_cols,int64_t  F_ij_row_rank,int64_t  F_ij_col_rank,
                        double *_F_ij) {
  MatrixWrapper D_i_block(_D_i_block, D_i_block_rows, D_i_block_cols, D_i_block_rows);
  MatrixWrapper D_j_block(_D_j_block, D_j_block_rows, D_j_block_cols, D_j_block_rows);
  MatrixWrapper F_ij(_F_ij, F_ij_rows, F_ij_cols, F_ij_rows);

  auto fill_in_splits = split_dense(F_ij,
                                    F_ij_rows - F_ij_row_rank,
                                    F_ij_cols - F_ij_col_rank);
  auto D_i_block_splits = split_dense(D_i_block,
                                      D_i_block_rows - D_i_block_row_rank,
                                      D_j_block_cols - D_j_block_col_rank);
  auto D_j_block_splits = split_dense(D_j_block,
                                      D_j_block_rows - D_j_block_row_rank,
                                      D_j_block_cols - D_j_block_col_rank);
  matmul(D_i_block_splits[0], D_j_block_splits[0], fill_in_splits[0],
         false, true, -1, 1); // cc
  matmul(D_i_block_splits[2], D_j_block_splits[0], fill_in_splits[2],
         false, true, -1, 1); // oc
  matmul(D_i_block_splits[2], D_j_block_splits[2], fill_in_splits[3],
         false, true, -1, 1); // oo
}

void CORE_nb_rank_fill_in(int64_t D_i_block_rows, int64_t D_i_block_cols, int64_t D_i_block_row_rank,
                          int64_t D_i_block_col_rank,
                          double *_D_i_block,
                          int64_t D_block_j_rows, int64_t D_block_j_cols, int64_t D_block_j_row_rank, int64_t D_block_j_col_rank,
                          double *_D_block_j,
                          int64_t U_j_rows, int64_t U_j_cols,
                          double *_U_j,
                          int64_t F_ij_rows, int64_t F_ij_cols, int64_t F_ij_row_rank, int64_t F_ij_col_rank,
                          double *_F_ij) {
  MatrixWrapper D_i_block(_D_i_block, D_i_block_rows, D_i_block_cols, D_i_block_rows);
  MatrixWrapper D_block_j(_D_block_j, D_block_j_rows, D_block_j_cols, D_block_j_rows);
  MatrixWrapper U_j(_U_j, U_j_rows, U_j_cols, U_j_rows);
  MatrixWrapper F_ij(_F_ij, F_ij_rows, F_ij_cols, F_ij_rows);

  auto D_i_block_splits = D_i_block.split({},
                                          std::vector<int64_t>(1,
                                                               D_i_block_cols -
                                                               D_i_block_col_rank));
  auto D_block_j_splits = split_dense(D_block_j,
                                      D_block_j_rows - D_block_j_row_rank,
                                      D_block_j_cols - D_block_j_col_rank);

  Matrix fill_in = matmul(D_i_block_splits[0], D_block_j_splits[1], false, false, -1);
  Matrix projected_fill_in = matmul(fill_in, U_j, false, true);

  F_ij += projected_fill_in;
}

void CORE_fill_in_addition(int64_t F_nrows,  int64_t F_ncols,
                             char which,
                             double *_F,
                             int64_t block_size,
                             double *_fill_in) {
  MatrixWrapper F(_F, F_nrows, F_ncols, F_ncols);
  MatrixWrapper fill_in(_fill_in, block_size, block_size, block_size);

  switch(which) {
  case 'R':
    fill_in += matmul(F, F, false, true);
    break;
  case 'C':
    fill_in += matmul(F, F, true, false);
    break;
  }
}

void CORE_fill_in_recompression(  int64_t block_size,
                                  double *_fill_in,
                                  int64_t rank,
                                  double *_US,
                                  int64_t U_nrows, int64_t U_ncols,
                                  double *_U,
                                  int64_t proj_nrows,
                                  double *_proj,
                                  char which) {
  MatrixWrapper fill_in(_fill_in, block_size, block_size, block_size);
  MatrixWrapper US(_US, rank, rank, rank);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);
  MatrixWrapper proj(_proj, proj_nrows, proj_nrows, proj_nrows);

  Matrix Q, Si, VT; double err;

  if (which == 'R') {           // row fill in recompresion
    fill_in += matmul(matmul(U, US), U, false, true);

    std::tie(Q, Si, VT, err) = truncated_svd(fill_in, rank);
    Matrix proj_row = matmul(Q, U, true, false);
    proj.copy_mem(proj_row);
  }
  else if (which == 'C') {      // col fill in recompression
    fill_in += matmul(U, matmul(US, U, false, true));

    Matrix fill_in_cols_T = transpose(fill_in);
    std::tie(Q, Si, VT, err) = truncated_svd(fill_in_cols_T, rank);
    Matrix proj_col = matmul(Q, U, true, false);
    proj.copy_mem(proj_col);
  }

  U.copy_mem(Q);
  US.copy_mem(Si);
}

void CORE_project_S(  int64_t S_nrows,
                      double *_S,
                      int64_t proj_nrows,
                      double *_proj,
                      char which) {
  MatrixWrapper S(_S, S_nrows, S_nrows, S_nrows);
  MatrixWrapper proj(_proj, proj_nrows, proj_nrows, proj_nrows);

  if (which == 'R') {           // fill in projection along the rows.
    Matrix proj_S = matmul(proj, S);
    S.copy_mem(proj_S);
  }
  else if (which == 'C') {       // fill in projection along the cols.
    Matrix proj_S = matmul(S, proj, false, true);
    S.copy_mem(proj_S);
  }
}

void CORE_schurs_complement_1(int64_t D_block_block_nrows,
                             int64_t D_block_rank,
                             double *_D_block_block,
                             int64_t D_i_block_nrows,
                             int64_t D_i_block_ncols,
                             double *_D_i_block) {
  MatrixWrapper D_block_block(_D_block_block, D_block_block_nrows,
                              D_block_block_nrows, D_block_block_nrows);
  MatrixWrapper D_i_block(_D_i_block, D_i_block_nrows, D_i_block_ncols, D_i_block_nrows);

  auto D_block_block_split = split_dense(D_block_block,
                                          D_block_block_nrows - D_block_rank,
                                          D_block_block_nrows - D_block_rank);

  auto D_i_block_split = D_i_block.split({},
                                         std::vector<int64_t>(1,
                                                              D_i_block_ncols - D_block_rank));

  matmul(D_i_block_split[0], D_block_block_split[2], D_i_block_split[1], false, true, -1, 1);
}

void CORE_schurs_complement_3(int64_t D_block_block_nrows,
                             int64_t D_block_j_ncols,
                             int64_t D_block_rank,
                             int64_t D_j_rank,
                             double *_D_block_block, double *_D_block_j) {
  MatrixWrapper D_block_block(_D_block_block,
                              D_block_block_nrows, D_block_block_nrows, D_block_block_nrows);
  MatrixWrapper D_block_j(_D_block_j, D_block_block_nrows, D_block_j_ncols, D_block_block_nrows);

  auto D_block_block_split = split_dense(D_block_block,
                                         D_block_block_nrows - D_block_rank,
                                         D_block_block_nrows - D_block_rank);
  auto D_block_j_split = split_dense(D_block_j,
                                     D_block_block_nrows - D_block_rank,
                                     D_block_j_ncols - D_j_rank);

  matmul(D_block_block_split[2], D_block_j_split[1], D_block_j_split[3], false, false, -1, 1);
}

void CORE_schurs_complement_2(int64_t D_i_block_nrows, int64_t D_i_block_ncols, int64_t D_block_rank,
                              double *_D_i_block,
                              int64_t D_j_block_nrows, int64_t D_j_block_ncols, double *_D_j_block,
                              int64_t D_i_j_nrows, int64_t D_i_j_ncols, double *_D_i_j) {
  MatrixWrapper D_i_block(_D_i_block, D_i_block_nrows, D_i_block_ncols, D_i_block_nrows);
  MatrixWrapper D_j_block(_D_j_block, D_j_block_nrows, D_j_block_ncols, D_j_block_nrows);
  MatrixWrapper D_i_j(_D_i_j, D_i_j_nrows, D_i_j_ncols, D_i_j_nrows);

  auto D_i_block_split =
    D_i_block.split({},
                    std::vector<int64_t>(1,
                                         D_i_block_ncols - D_block_rank));
  auto D_j_block_split =
    D_j_block.split({},
                    std::vector<int64_t>(1,
                                         D_j_block_ncols - D_block_rank));

  matmul(D_i_block_split[0], D_j_block_split[0], D_i_j, false, true, -1, 1);
}

void CORE_syrk_2(int64_t D_i_block_nrows, int64_t D_i_block_ncols, int64_t D_block_rank,
                 double *_D_i_block,
                 int64_t D_i_j_nrows, int64_t D_i_j_ncols,
                 double *_D_i_j) {
  MatrixWrapper D_i_block(_D_i_block, D_i_block_nrows, D_i_block_ncols, D_i_block_nrows);
  MatrixWrapper D_i_j(_D_i_j, D_i_j_nrows, D_i_j_ncols, D_i_j_nrows);

  auto D_i_block_split = D_i_block.split({},
                                         std::vector<int64_t>(1, D_i_block_ncols - D_block_rank));

  syrk(D_i_block_split[0], D_i_j, Hatrix::Lower, false, -1, 1);
}

void CORE_schurs_complement_4(int64_t D_i_dim, int64_t D_j_dim, int64_t D_block_dim,
                              int64_t A_i_rank, int64_t A_j_rank, int64_t A_block_rank,
                              double *_D_i_j, double *_D_block_j, double *_D_i_block) {
    MatrixWrapper D_i_j(_D_i_j, D_i_dim, D_j_dim, D_i_dim);
  MatrixWrapper D_i_block(_D_i_block, D_i_dim, D_block_dim, D_i_dim);
  MatrixWrapper D_block_j(_D_block_j, D_block_dim, D_j_dim, D_block_dim);

  auto D_i_block_split = D_i_block.split({},
                                         std::vector<int64_t>(1, D_j_dim - A_j_rank));
  auto D_block_j_split = split_dense(D_block_j,
                                     D_block_dim - A_block_rank,
                                     D_j_dim - A_j_rank);
  auto D_i_j_split = D_i_j.split({},
                                 std::vector<int64_t>(1, D_j_dim - A_j_rank));

  matmul(D_i_block_split[0], D_block_j_split[1], D_i_j_split[1], false, false, -1, 1);
}

void CORE_transfer_basis_update(int64_t U_nrows, int64_t U_ncols, int64_t rank_c1, int64_t rank_c2,
                                double *_proj_c1, double *_proj_c2, double *_U) {
  MatrixWrapper proj_c1(_proj_c1, rank_c1, rank_c1, rank_c1);
  MatrixWrapper proj_c2(_proj_c2, rank_c2, rank_c2, rank_c2);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  Matrix Utransfer_new(U, true);

  auto Utransfer_new_splits = Utransfer_new.split(std::vector<int64_t>(1, rank_c1),
                                                  {});
  auto Utransfer_splits     = U.split(std::vector<int64_t>(1, rank_c1),
                                      {});

  matmul(proj_c1, Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1.0, 0.0);
  matmul(proj_c2, Utransfer_splits[1], Utransfer_new_splits[1], false, false, 1.0, 0.0);
  U.copy_mem(Utransfer_new);
}

void CORE_project_fill_in(int64_t nrows, int64_t ncols, int64_t rank_i, int64_t rank_j,
                          double *_Ui, double *_Uj, double *_Fij, double *_Sij) {
  MatrixWrapper Ui(_Ui, nrows, rank_i, nrows);
  MatrixWrapper Uj(_Uj, ncols, rank_j, ncols);
  MatrixWrapper Fij(_Fij, nrows, ncols, nrows);
  MatrixWrapper Sij(_Sij, rank_i, rank_j, rank_i);

  Matrix temp = matmul(matmul(Ui, Fij, true), Uj);
  Sij += temp;
}
