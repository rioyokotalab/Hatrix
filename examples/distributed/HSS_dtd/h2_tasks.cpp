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

  MatrixWrapper D(_D, D_nrows, D_ncols, D_nrows);

  cholesky(D, Hatrix::Lower);

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

  Matrix UF = make_complement(U);
  Matrix product = matmul(matmul(UF, D, true), UF);

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
  solve_triangular(D_splits[0], D_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  syrk(D_splits[2], D_splits[3], Hatrix::Lower, false, -1, 1);

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
