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
task_cholesky_full(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, D_ncols;
  double *_D;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &D_ncols,
                         &_D);

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);

  cholesky(D, Hatrix::Lower);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_solve_triangular_full(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_dd_nrows, D_dd_ncols;
  double *_D_dd;
  int64_t D_id_nrows, D_id_ncols;
  double *_D_id;

  parsec_dtd_unpack_args(this_task,
                         &D_dd_nrows, &D_dd_ncols, &_D_dd,
                         &D_id_nrows, &D_id_ncols, &_D_id);

  MatrixWrapper D_dd(_D_dd, D_dd_nrows, D_dd_ncols, D_dd_nrows);
  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);

  solve_triangular(D_dd, D_id, Hatrix::Right, Hatrix::Lower, false, true, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_syrk_full(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_id_nrows, D_id_ncols;
  double *_D_id;
  int64_t D_ij_nrows, D_ij_ncols;
  double *_D_ij;

  parsec_dtd_unpack_args(this_task,
                         &D_id_nrows, &D_id_ncols, &_D_id,
                         &D_ij_nrows, &D_ij_ncols, &_D_ij);

  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);
  MatrixWrapper D_ij(_D_ij, D_ij_nrows, D_ij_ncols, D_ij_nrows);

  syrk(D_id, D_ij, Hatrix::Lower, false, -1.0, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_matmul_full(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_id_nrows, D_id_ncols;
  double *_D_id;
  int64_t D_jd_nrows, D_jd_ncols;
  double *_D_jd;
  int64_t D_ij_nrows, D_ij_ncols;
  double *_D_ij;

  parsec_dtd_unpack_args(this_task,
                         &D_id_nrows, &D_id_ncols, &_D_id,
                         &D_jd_nrows, &D_jd_ncols, &_D_jd,
                         &D_ij_nrows, &D_ij_ncols, &_D_ij);

  MatrixWrapper D_id(_D_id, D_id_nrows, D_id_ncols, D_id_nrows);
  MatrixWrapper D_jd(_D_jd, D_jd_nrows, D_jd_ncols, D_jd_nrows);
  MatrixWrapper D_ij(_D_ij, D_ij_nrows, D_ij_ncols, D_ij_nrows);

  matmul(D_id, D_jd, D_ij, false, true, -1.0, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_multiply_full_complement(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, D_ncols, U_nrows, U_ncols;
  double *_D, *_U;

  parsec_dtd_unpack_args(this_task,
                         &_D,
                         &D_nrows, &D_ncols,
                         &_U,
                         &U_nrows, &U_ncols);

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  // Matrix UF = make_complement(U);
  // Matrix product = matmul(matmul(UF, D, true), UF);

  // D.copy_mem(product);

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
  solve_triangular(D_splits[0], D_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_multiply_partial_complement_left(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, D_ncols, D_col_rank;
  double *_D;
  int64_t U_nrows, U_ncols;
  double *_U;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &D_ncols, &D_col_rank, &_D,
                         &U_nrows, &U_ncols, &_U);

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  // Matrix U_F = make_complement(U);

  // auto D_splits = D.split({},
  //                         std::vector<int64_t>(1, D_ncols - D_col_rank));
  // D_splits[1] = matmul(U_F, D_splits[1], true);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_multiply_partial_complement_right(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, D_ncols, D_row_rank;
  double *_D;
  int64_t U_nrows, U_ncols;
  double *_U;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &D_ncols, &D_row_rank, &_D,
                         &U_nrows, &U_ncols, &_U);

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);

  // Matrix U_F = make_complement(U);
  // auto D_splits = D.split(std::vector<int64_t>(1, D_nrows - D_row_rank), {});

  // D_splits[1] = matmul(D_splits[1], U_F);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_trsm_co(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_rows, D_cols, D_row_rank, D_col_rank;
  int64_t O_rows, O_cols, O_row_rank, O_col_rank;
  double *_diagonal, *_other;

  parsec_dtd_unpack_args(this_task,
                         &D_rows, &D_cols, &D_row_rank, &D_col_rank, &_diagonal,
                         &O_rows, &O_cols, &O_row_rank, &O_col_rank, &_other);

  MatrixWrapper diagonal(_diagonal, D_rows, D_cols, D_rows);
  MatrixWrapper other(_other, O_rows, O_cols, O_rows);

  auto diagonal_splits = split_dense(diagonal,
                                     D_rows - D_row_rank,
                                     D_cols - D_col_rank);
  auto other_splits = split_dense(other,
                                  O_rows - O_row_rank,
                                  O_cols - O_col_rank);

  solve_triangular(diagonal_splits[0], other_splits[1], Hatrix::Left, Hatrix::Lower,
                   false, false, 1.0);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_trsm_cc_oc(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_rows, D_cols, D_row_rank, D_col_rank;
  int64_t O_rows, O_cols, O_row_rank, O_col_rank;
  double *_diagonal, *_other;

  parsec_dtd_unpack_args(this_task,
                         &D_rows, &D_cols, &D_row_rank, &D_col_rank, &_diagonal,
                         &O_rows, &O_cols, &O_row_rank, &O_col_rank, &_other);

  MatrixWrapper diagonal(_diagonal, D_rows, D_cols, D_rows);
  MatrixWrapper other(_other, O_rows, O_cols, O_rows);

  auto diagonal_splits = split_dense(diagonal,
                                     D_rows - D_row_rank,
                                     D_cols - D_col_rank);
  auto other_splits = split_dense(other,
                                  O_rows - O_row_rank,
                                  O_cols - O_col_rank);

  solve_triangular(diagonal_splits[0], other_splits[0], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  solve_triangular(diagonal_splits[0], other_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);

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

  // std::cout << "done NB RANK fill in task.\n";

  return PARSEC_HOOK_RETURN_DONE;
}

void
parsec_dtd_unpack_refs(parsec_task_t *this_task, ...)
{
    parsec_dtd_task_t *current_task = (parsec_dtd_task_t *)this_task;
    parsec_dtd_task_param_t *current_param = GET_HEAD_OF_PARAM_LIST(current_task);
    int i = 0;
    void *tmp_val;
    void **tmp_ref;
    va_list arguments;

    va_start(arguments, this_task);
    while( current_param != NULL) {
      if((current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_INPUT ||
         (current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_INOUT ||
         (current_param->op_type & PARSEC_GET_OP_TYPE) == PARSEC_OUTPUT ) {
        tmp_ref = va_arg(arguments, void**);
        *tmp_ref = this_task->data[i].data_in;
        i++;
      }
      current_param = current_param->next;
    }
    va_end(arguments);
}

parsec_hook_return_t
task_fill_in_addition(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t F_block_j_nrows,  F_block_j_ncols;
  double *_F_block_j;
  int64_t block_size;
  double *_fill_in;

  parsec_dtd_unpack_args(this_task,
                         &F_block_j_nrows, &F_block_j_ncols, &_F_block_j,
                         &block_size, &_fill_in);

  MatrixWrapper F_block_j(_F_block_j, F_block_j_nrows, F_block_j_ncols, F_block_j_ncols);
  MatrixWrapper fill_in(_fill_in, block_size, block_size, block_size);

  fill_in += matmul(F_block_j, F_block_j, false, true);

  // parsec_data_copy_t* _F_block_j_tile, *_fill_in_tile;
  // parsec_dtd_unpack_refs(this_task, &_F_block_j_tile, &_fill_in_tile);

  // PARSEC_OBJ_RELEASE(_F_block_j_tile->data_copy->original); // release the parsec reference.

  // TODO: free the F(block, j) fill in with a custom desctructor..

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_fill_in_QR(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t block_size;
  double *_fill_in;
  int64_t rank;
  double *_US;
  int64_t U_nrows, U_ncols;
  double *_U;
  int64_t r_nrows;
  double *_r;

  parsec_dtd_unpack_args(this_task,
                         &block_size, &_fill_in,
                         &rank, &_US,
                         &U_nrows, &U_ncols, &_U,
                         &r_nrows, &_r);

  MatrixWrapper fill_in(_fill_in, block_size, block_size, block_size);
  MatrixWrapper US(_US, rank, rank, rank);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);
  MatrixWrapper r(_r, r_nrows, r_nrows, r_nrows);

  fill_in += matmul(matmul(U, US), U, false, true);

  Matrix Q,R;
  std::tie(Q, R) = pivoted_qr_nopiv_return(fill_in, rank);

  Matrix r_row = matmul(Q, U, true, false);
  r.copy_mem(r_row);
  U.copy_mem(Q);

  Matrix Si(R.rows, R.rows), Vi(R.rows, R.cols);
  rq(R, Si, Vi);
  US.copy_mem(Si);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_project_S(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t S_nrows, S_ncols;
  double *_S;
  int64_t r_nrows;
  double *_r;

  parsec_dtd_unpack_args(this_task,
                         &S_nrows, &S_ncols, &_S,
                         &r_nrows, &_r);

  MatrixWrapper S(_S, S_nrows, S_ncols, S_nrows);
  MatrixWrapper r(_r, r_nrows, r_nrows, r_nrows);

  Matrix rS = matmul(r, S);

  S.copy_mem(rS);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_project_S_left(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t S_nrows, S_ncols;
  double *_S;
  int64_t t_nrows;
  double *_t;

  parsec_dtd_unpack_args(this_task,
                         &S_nrows, &S_ncols, &_S,
                         &t_nrows, &_t);
  MatrixWrapper S(_S, S_nrows, S_ncols, S_nrows);
  MatrixWrapper t(_t, t_nrows, t_nrows, t_nrows);

  Matrix St = matmul(S, t, false, true);
  S.copy_mem(St);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_fill_in_cols_addition(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t F_i_block_nrows,  F_i_block_ncols;
  double *_F_i_block;
  int64_t block_size;
  double *_fill_in;

  parsec_dtd_unpack_args(this_task,
                         &F_i_block_nrows, &F_i_block_ncols, &_F_i_block,
                         &block_size, &_fill_in);

  MatrixWrapper F_i_block(_F_i_block, F_i_block_nrows, F_i_block_ncols, F_i_block_ncols);
  MatrixWrapper fill_in(_fill_in, block_size, block_size, block_size);

  fill_in += matmul(F_i_block, F_i_block, true, false);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_fill_in_cols_QR(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t block_size;
  double *_fill_in_cols;
  int64_t rank;
  double *_US;
  int64_t U_nrows, U_ncols;
  double *_U;
  int64_t t_nrows;
  double *_t;

  parsec_dtd_unpack_args(this_task,
                         &block_size, &_fill_in_cols,
                         &rank, &_US,
                         &U_nrows, &U_ncols, &_U,
                         &t_nrows, &_t);

  MatrixWrapper fill_in_cols(_fill_in_cols, block_size, block_size, block_size);
  MatrixWrapper US(_US, rank, rank, rank);
  MatrixWrapper U(_U, U_nrows, U_ncols, U_nrows);
  MatrixWrapper t(_t, t_nrows, t_nrows, t_nrows);

  fill_in_cols += matmul(U, matmul(US, U, false, true));
  Matrix fill_in_cols_T = transpose(fill_in_cols);

  Matrix Q, R;
  std::tie(Q, R) = pivoted_qr_nopiv_return(fill_in_cols_T, rank);

  Matrix t_row = matmul(Q, U, true, false);
  t.copy_mem(t_row);
  U.copy_mem(Q);

  Matrix Si(R.rows, R.rows), Vi(R.rows, R.cols);
  rq(R, Si, Vi);
  US.copy_mem(Si);

  return PARSEC_HOOK_RETURN_DONE;
}
