#ifndef TASKS_C_INTERFACE_H
#define TASKS_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
  void CORE_multiply_complement(int64_t D_nrows, int64_t D_ncols, int64_t D_row_rank, int64_t D_col_rank,
                                int64_t U_nrows, int64_t U_ncols, double* _D, double* _U, char which);

  void CORE_factorize_diagonal(int64_t D_nrows, int64_t rank_nrows, double *_D);

#ifdef __cplusplus
}
#endif

#endif
