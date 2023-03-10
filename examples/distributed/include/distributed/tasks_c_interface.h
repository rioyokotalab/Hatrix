#ifndef TASKS_C_INTERFACE_H
#define TASKS_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
  int64_t
  CORE_get_dim(const int64_t block, const int64_t level);

  void CORE_multiply_complement(int64_t D_nrows, int64_t D_ncols, int64_t D_row_rank, int64_t D_col_rank,
                                int64_t U_nrows, int64_t U_ncols, double* _D, double* _U, char which);

#ifdef __cplusplus
}
#endif

#endif
