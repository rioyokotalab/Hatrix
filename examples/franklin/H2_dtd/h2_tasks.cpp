#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_tasks.hpp"

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

parsec_hook_return_t
task_multiply_full_complement(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  bool left;
  int64_t D_nrows, D_ncols, U_nrows, U_ncols;
  double *_D, *_U;

  parsec_dtd_unpack_args(this_task,
                         &left,
                         &_D,
                         &D_nrows, &D_ncols,
                         &_U,
                         &U_nrows, &U_ncols);

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  Matrix UF = make_complement(U);
  Matrix product = left ? matmul(UF, D, true) : matmul(D, UF);

  D.copy_mem(product);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_factorize_diagonal(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, rank_nrows;
  double *_D;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &rank_nrows, &_D);

  MatrixWrapper D(_D, D_nrows, D_nrows, D_nrows);
  auto D_splits = split_dense(D,
                              D_nrows - rank_nrows,
                              D_nrows - rank_nrows);

  cholesky(D_splits[0], Hatrix::Lower);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_multiply_partial_complement(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  return PARSEC_HOOK_RETURN_DONE;
}


parsec_hook_return_t
task_partial_trsm(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_rows, D_cols, D_row_rank, D_col_rank;
  int64_t O_rows, O_cols, O_row_rank, O_col_rank;
  double *_diagonal, *_other;
  Hatrix::Side side;
  Hatrix::Mode uplo;
  bool unit_diag, trans_A;
  int64_t split_index;

  parsec_dtd_unpack_args(this_task,
                         &D_rows, &D_cols, &D_row_rank, &D_col_rank, &_diagonal,
                         &O_rows, &O_cols, &O_row_rank, &O_col_rank, &_other,
                         &side, &uplo, &unit_diag, &trans_A, &split_index);

  MatrixWrapper diagonal(_diagonal, D_rows, D_cols, D_rows);
  MatrixWrapper other(_other, O_rows, O_cols, O_rows);

  auto diagonal_splits = split_dense(diagonal,
                                     D_rows - D_row_rank,
                                     D_cols - D_col_rank);
  auto other_splits = split_dense(other,
                                  O_rows - O_row_rank,
                                  O_cols - O_col_rank);

  solve_triangular(diagonal_splits[0], other_splits[split_index], side, uplo,
                   unit_diag, trans_A, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_partial_syrk(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_rows, D_i_block_cols, D_i_block_row_rank,
    D_i_block_col_rank, D_i_block_split_index;
  double *_D_i_block;
  int64_t D_ij_rows, D_ij_cols, D_ij_row_rank, D_ij_col_rank,
    D_ij_split_index;
  double *_D_ij;
  Hatrix::Mode uplo;
  bool unit_diag;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_rows, &D_i_block_cols, &D_i_block_row_rank,
                         &D_i_block_col_rank, &D_i_block_split_index,
                         &_D_i_block,
                         &D_ij_rows, &D_ij_cols, &D_ij_row_rank, &D_ij_col_rank,
                         &D_ij_split_index,
                         &_D_ij, &uplo, &unit_diag);

  MatrixWrapper D_i_block(_D_i_block, D_i_block_rows, D_i_block_cols, D_i_block_rows);
  MatrixWrapper D_ij(_D_ij, D_ij_rows, D_ij_cols, D_ij_rows);

  auto D_i_block_splits = split_dense(D_i_block,
                                      D_i_block_rows - D_i_block_row_rank,
                                      D_i_block_cols - D_i_block_col_rank);
  auto D_ij_splits = split_dense(D_ij,
                                 D_ij_rows - D_ij_row_rank,
                                 D_ij_cols - D_ij_col_rank);

  syrk(D_i_block_splits[D_i_block_split_index], D_ij_splits[D_ij_split_index],
       uplo, unit_diag, -1.0, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_partial_matmul(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_rows, D_i_block_cols, D_i_block_row_rank,
    D_i_block_col_rank, D_i_block_split_index;
  double *_D_i_block;

  int64_t D_j_block_rows, D_j_block_cols, D_j_block_row_rank,
    D_j_block_col_rank, D_j_block_split_index;
  double *_D_j_block;

  int64_t D_ij_rows, D_ij_cols, D_ij_row_rank, D_ij_col_rank,
    D_ij_split_index;
  double *_D_ij;
  bool transA, transB;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_rows, &D_i_block_cols, &D_i_block_row_rank,
                         &D_i_block_col_rank, &D_i_block_split_index,
                         &_D_i_block,
                         &D_j_block_rows, &D_j_block_cols, &D_j_block_row_rank,
                         &D_j_block_col_rank, &D_j_block_split_index,
                         &_D_j_block,
                         &D_ij_rows, &D_ij_cols, &D_ij_row_rank, &D_ij_col_rank,
                         &D_ij_split_index,
                         &_D_ij,
                         &transA, &transB);

  MatrixWrapper D_i_block(_D_i_block, D_i_block_rows, D_i_block_cols, D_i_block_rows);
  MatrixWrapper D_j_block(_D_j_block, D_j_block_rows, D_j_block_cols, D_j_block_rows);
  MatrixWrapper D_ij(_D_ij, D_ij_rows, D_ij_cols, D_ij_rows);

  auto D_i_block_splits = split_dense(D_i_block,
                                      D_i_block_rows - D_i_block_row_rank,
                                      D_i_block_cols - D_i_block_col_rank);
  auto D_j_block_splits = split_dense(D_j_block,
                                      D_j_block_rows - D_j_block_row_rank,
                                      D_j_block_cols - D_j_block_col_rank);
  auto D_ij_splits = split_dense(D_ij,
                                 D_ij_rows - D_ij_row_rank,
                                 D_ij_cols - D_ij_col_rank);

  matmul(D_i_block_splits[D_i_block_split_index],
         D_j_block_splits[D_j_block_split_index],
         D_ij_splits[D_ij_split_index], transA, transB,
         -1.0, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_copy_blocks(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  bool copy_dense;
  double *_D_unelim;
  int64_t D_unelim_rows, D_unelim_cols, D_unelim_row_rank, D_unelim_col_rank;

  double *_D_c1c2;
  int64_t D_c1c2_rows, D_c1c2_cols, D_c1c2_row_rank, D_c1c2_col_rank;
  int D_unelim_split_index;

  parsec_dtd_unpack_args(this_task, &copy_dense,
                         &_D_unelim,
                         &D_unelim_rows, &D_unelim_cols, &D_unelim_row_rank, &D_unelim_col_rank,
                         &_D_c1c2,
                         &D_c1c2_rows, &D_c1c2_cols, &D_c1c2_row_rank, &D_c1c2_col_rank,
                         &D_unelim_split_index);

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


  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_nb_nb_fill_in(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank;
  double *_D_i_block;
  int64_t D_j_block_rows, D_j_block_cols, D_j_block_row_rank, D_j_block_col_rank;
  double *_D_j_block;
  int64_t F_ij_rows, F_ij_cols, F_ij_row_rank, F_ij_col_rank;
  double *_F_ij;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_rows, &D_i_block_cols, &D_i_block_row_rank,
                         &D_i_block_col_rank, &_D_i_block,
                         &D_j_block_rows, &D_j_block_cols, &D_j_block_row_rank,
                         &D_j_block_col_rank, &_D_j_block,
                         &F_ij_rows, &F_ij_cols, &F_ij_row_rank,
                         &F_ij_col_rank, &_F_ij);

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

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_nb_rank_fill_in(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank;
  double *_D_i_block;
  int64_t D_block_j_rows, D_block_j_cols, D_block_j_row_rank, D_block_j_col_rank;
  double *_D_block_j;
  int64_t U_j_rows, U_j_cols;
  double *_U_j;
  int64_t F_ij_rows, F_ij_cols, F_ij_row_rank, F_ij_col_rank;
  double *_F_ij;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_rows, &D_i_block_cols, &D_i_block_row_rank,
                         &D_i_block_col_rank, &_D_i_block,
                         &D_block_j_rows, &D_block_j_cols, &D_block_j_row_rank,
                         &D_block_j_col_rank, &_D_block_j,
                         &U_j_rows, &U_j_cols,
                         &_U_j,
                         &F_ij_rows, &F_ij_cols, &F_ij_row_rank,
                         &F_ij_col_rank, &_F_ij);

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

  return PARSEC_HOOK_RETURN_DONE;
}
