#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

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

  CORE_cholesky_full(D_nrows, D_ncols, _D);

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

  CORE_solve_triangular_full(D_dd_nrows, D_dd_ncols, _D_dd,
                         D_id_nrows, D_id_ncols, _D_id);

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

  CORE_syrk_full(D_id_nrows, D_id_ncols, _D_id,
                 D_ij_nrows, D_ij_ncols, _D_ij);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_matmul_full(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_jd_nrows, D_jd_ncols;
  double *_D_jd;
  int64_t D_id_nrows, D_id_ncols;
  double *_D_id;
  int64_t D_ji_nrows, D_ji_ncols;
  double *_D_ji;

  parsec_dtd_unpack_args(this_task,
                         &D_id_nrows, &D_id_ncols, &_D_id,
                         &D_jd_nrows, &D_jd_ncols, &_D_jd,
                         &D_ji_nrows, &D_ji_ncols, &_D_ji);

  CORE_matmul_full(D_jd_nrows, D_jd_ncols, _D_jd,
                   D_id_nrows, D_id_ncols, _D_id,
                   D_ji_nrows, D_ji_ncols, _D_ji);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_factorize_diagonal(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, rank_nrows;
  double *_D;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &rank_nrows, &_D);
  CORE_factorize_diagonal(D_nrows, rank_nrows, _D);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_multiply_complement(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_nrows, D_ncols, D_row_rank, D_col_rank, U_nrows, U_ncols;
  double *_D, *_U;
  char which;

  parsec_dtd_unpack_args(this_task,
                         &D_nrows, &D_ncols, &D_row_rank, &D_col_rank,
                         &U_nrows, &U_ncols,
                         &_D, &_U, &which);

  CORE_multiply_complement(D_nrows, D_ncols, D_row_rank, D_col_rank,
                           U_nrows, U_ncols,
                           _D, _U, which);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_trsm(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_rows, D_cols, D_row_rank, D_col_rank;
  int64_t O_rows, O_cols, O_row_rank, O_col_rank;
  double *_diagonal, *_other;
  char which;

  parsec_dtd_unpack_args(this_task,
                         &D_rows, &D_cols, &D_row_rank, &D_col_rank, &_diagonal,
                         &O_rows, &O_cols, &O_row_rank, &O_col_rank, &_other, &which);

  CORE_trsm(D_rows, D_cols, D_row_rank, D_col_rank, _diagonal,
            O_rows, O_cols, O_row_rank, O_col_rank, _other, which);

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

  CORE_copy_blocks(copy_dense,
                   _D_unelim,
                   D_unelim_rows, D_unelim_cols, D_unelim_row_rank, D_unelim_col_rank,
                   _D_c1c2,
                   D_c1c2_rows, D_c1c2_cols, D_c1c2_row_rank, D_c1c2_col_rank,
                   D_unelim_split_index);

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

  CORE_nb_nb_fill_in(D_i_block_rows, D_i_block_cols, D_i_block_row_rank,
                    D_i_block_col_rank, _D_i_block,
                    D_j_block_rows, D_j_block_cols, D_j_block_row_rank,
                    D_j_block_col_rank, _D_j_block,
                    F_ij_rows, F_ij_cols, F_ij_row_rank,
                    F_ij_col_rank, _F_ij);


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

  CORE_nb_rank_fill_in(D_i_block_rows, D_i_block_cols, D_i_block_row_rank,
                       D_i_block_col_rank, _D_i_block,
                       D_block_j_rows, D_block_j_cols, D_block_j_row_rank,
                       D_block_j_col_rank, _D_block_j,
                       U_j_rows, U_j_cols,
                       _U_j,
                       F_ij_rows, F_ij_cols, F_ij_row_rank,
                       F_ij_col_rank, _F_ij);

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
  int64_t F_nrows,  F_ncols;
  char which;
  double *_F;
  int64_t block_size;
  double *_fill_in;

  parsec_dtd_unpack_args(this_task,
                         &F_nrows, &F_ncols,
                         &which,
                         &_F,
                         &block_size, &_fill_in);

  CORE_fill_in_addition(F_nrows, F_ncols,
                         which,
                         _F,
                         block_size, _fill_in);


  // parsec_data_copy_t* _F_block_j_tile, *_fill_in_tile;
  // parsec_dtd_unpack_refs(this_task, &_F_block_j_tile, &_fill_in_tile);

  // PARSEC_OBJ_RELEASE(_F_block_j_tile->data_copy->original); // release the parsec reference.

  // TODO: free the F(block, j) fill in with a custom desctructor..

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_fill_in_recompression(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t block_size;
  double *_fill_in;
  int64_t rank;
  double *_US;
  int64_t U_nrows, U_ncols;
  double *_U;
  int64_t proj_nrows;
  double *_proj;
  char which;

  parsec_dtd_unpack_args(this_task,
                         &block_size, &_fill_in,
                         &rank, &_US,
                         &U_nrows, &U_ncols, &_U,
                         &proj_nrows, &_proj, &which);

  CORE_fill_in_recompression(block_size, _fill_in,
                         rank, _US,
                         U_nrows, U_ncols, _U,
                         proj_nrows, _proj, which);



  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_project_S(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t S_nrows;
  double *_S;
  int64_t proj_nrows;
  double *_proj;
  char which;

  parsec_dtd_unpack_args(this_task,
                         &S_nrows, &_S,
                         &proj_nrows, &_proj,
                         &which);


  CORE_project_S(S_nrows, _S,
                         proj_nrows, _proj,
                 which);



  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_schurs_complement_1(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_block_block_nrows;
  int64_t D_block_rank;
  double *_D_block_block;
  int64_t D_i_block_nrows;
  int64_t D_i_block_ncols;
  double *_D_i_block;

  parsec_dtd_unpack_args(this_task,
                         &D_block_block_nrows,
                         &D_block_rank,
                         &_D_block_block,
                         &D_i_block_nrows,
                         &D_i_block_ncols,
                         &_D_i_block);

  CORE_schurs_complement_1(D_block_block_nrows,
                           D_block_rank,
                           _D_block_block,
                           D_i_block_nrows,
                           D_i_block_ncols,
                           _D_i_block);


  return PARSEC_HOOK_RETURN_DONE;
}


parsec_hook_return_t
task_schurs_complement_3(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_block_block_nrows;
  int64_t D_block_j_ncols;
  int64_t D_block_rank;
  int64_t D_j_rank;
  double *_D_block_block, *_D_block_j;

  parsec_dtd_unpack_args(this_task,
                         &D_block_block_nrows,
                         &D_block_j_ncols,
                         &D_block_rank,
                         &D_j_rank,
                         &_D_block_block,
                         &_D_block_j);

  CORE_schurs_complement_3(D_block_block_nrows,
                         D_block_j_ncols,
                         D_block_rank,
                         D_j_rank,
                         _D_block_block,
                           _D_block_j);


  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_schurs_complement_2(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_nrows;
  int64_t D_i_block_ncols;
  int64_t D_block_rank;
  double *_D_i_block;
  int64_t D_j_block_nrows;
  int64_t D_j_block_ncols;
  double *_D_j_block;
  int64_t D_i_j_nrows;
  int64_t D_i_j_ncols;
  double *_D_i_j;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_nrows,
                         &D_i_block_ncols,
                         &D_block_rank,
                         &_D_i_block,
                         &D_j_block_nrows,
                         &D_j_block_ncols,
                         &_D_j_block,
                         &D_i_j_nrows,
                         &D_i_j_ncols,
                         &_D_i_j);

  CORE_schurs_complement_2(D_i_block_nrows,
                         D_i_block_ncols,
                         D_block_rank,
                         _D_i_block,
                         D_j_block_nrows,
                         D_j_block_ncols,
                         _D_j_block,
                         D_i_j_nrows,
                         D_i_j_ncols,
                         _D_i_j);


  return PARSEC_HOOK_RETURN_DONE;
}



parsec_hook_return_t
task_syrk_2(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_block_nrows;
  int64_t D_i_block_ncols;
  int64_t D_block_rank;
  double *_D_i_block;
  int64_t D_i_j_nrows;
  int64_t D_i_j_ncols;
  double *_D_i_j;

  parsec_dtd_unpack_args(this_task,
                         &D_i_block_nrows, &D_i_block_ncols, &D_block_rank, &_D_i_block,
                         &D_i_j_nrows, &D_i_j_ncols, &_D_i_j);

  CORE_syrk_2(D_i_block_nrows, D_i_block_ncols, D_block_rank, _D_i_block,
              D_i_j_nrows, D_i_j_ncols, _D_i_j);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_schurs_complement_4(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t D_i_dim;
  int64_t D_j_dim;
  int64_t D_block_dim;
  int64_t A_i_rank;
  int64_t A_j_rank;
  int64_t A_block_rank;
  double *_D_i_j, *_D_block_j, *_D_i_block;

  parsec_dtd_unpack_args(this_task,
                         &D_i_dim,
                         &D_j_dim,
                         &D_block_dim,
                         &A_i_rank,
                         &A_j_rank,
                         &A_block_rank,
                         &_D_i_block,
                         &_D_block_j,
                         &_D_i_j);

  CORE_schurs_complement_4(D_i_dim,
                         D_j_dim,
                         D_block_dim,
                         A_i_rank,
                         A_j_rank,
                         A_block_rank,
                         _D_i_block,
                         _D_block_j,
                         _D_i_j);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_transfer_basis_update(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t U_nrows, U_ncols, rank_c1, rank_c2;
  double *_proj_c1, *_proj_c2, *_U;

  parsec_dtd_unpack_args(this_task,
                         &U_nrows, &U_ncols, &rank_c1, &rank_c2,
                         &_proj_c1, &_proj_c2, &_U);

  CORE_transfer_basis_update(U_nrows, U_ncols, rank_c1, rank_c2,
                         _proj_c1, _proj_c2, _U);

  return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
task_project_fill_in(parsec_execution_stream_t* es, parsec_task_t* this_task) {
  int64_t nrows, ncols, rank_i, rank_j;
  double *_Ui, *_Uj, *_Fij, *_Sij;

  parsec_dtd_unpack_args(this_task,
                         &nrows, &ncols, &rank_i, &rank_j,
                         &_Ui, &_Uj, &_Fij, &_Sij);

  CORE_project_fill_in(nrows, ncols, rank_i, rank_j,
                         _Ui, _Uj, _Fij, _Sij);


  return PARSEC_HOOK_RETURN_DONE;
}
