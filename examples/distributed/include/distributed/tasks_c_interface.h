#ifndef TASKS_C_INTERFACE_H
#define TASKS_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
  void CORE_multiply_complement(int64_t D_nrows, int64_t D_ncols, int64_t D_row_rank, int64_t D_col_rank,
                                int64_t U_nrows, int64_t U_ncols, double* _D, double* _U, char which);

  void CORE_factorize_diagonal(int64_t D_nrows, int64_t rank_nrows, double *_D);

  void CORE_cholesky_full(int64_t D_nrows, int64_t D_ncols, double* _D);

  void CORE_solve_triangular_full(int64_t D_dd_nrows, int64_t D_dd_ncols, double* _D_dd,
                                  int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id);

  void CORE_syrk_full(int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id,
                      int64_t D_ij_nrows, int64_t D_ij_ncols, double* _D_ij);

  void CORE_matmul_full(int64_t D_id_nrows, int64_t D_id_ncols, double* _D_id,
                        int64_t D_jd_nrows, int64_t D_jd_ncols, double* _D_jd,
                        int64_t D_ij_nrows, int64_t D_ij_ncols, double* _D_ij);

  void CORE_trsm(int64_t D_rows, int64_t D_cols, int64_t D_row_rank, int64_t D_col_rank, double* _diagonal,
                 int64_t O_rows, int64_t O_cols, int64_t O_row_rank, int64_t O_col_rank, double* _other,
                 char which);

#ifdef __cplusplus
}
#endif

#endif
