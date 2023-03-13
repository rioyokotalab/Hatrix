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

  void CORE_copy_blocks(bool copy_dense,
                        double *_D_unelim,
                        int64_t D_unelim_rows, int64_t D_unelim_cols, int64_t D_unelim_row_rank, int64_t D_unelim_col_rank,
                        double *_D_c1c2,
                        int64_t D_c1c2_rows, int64_t D_c1c2_cols, int64_t D_c1c2_row_rank, int64_t D_c1c2_col_rank,
                        int D_unelim_split_index);

  void CORE_nb_nb_fill_in(int64_t D_i_block_rows, int64_t D_i_block_cols,int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                          double *_D_i_block,
                          int64_t D_j_block_rows, int64_t  D_j_block_cols,int64_t D_j_block_row_rank,int64_t D_j_block_col_rank,
                          double *_D_j_block,
                          int64_t F_ij_rows, int64_t F_ij_cols,int64_t  F_ij_row_rank,int64_t  F_ij_col_rank,
                          double *_F_ij);

  void CORE_nb_rank_fill_in(int64_t D_i_block_rows, int64_t D_i_block_cols, int64_t D_i_block_row_rank,
                            int64_t D_i_block_col_rank,
                            double *_D_i_block,
                            int64_t D_block_j_rows, int64_t D_block_j_cols, int64_t D_block_j_row_rank, int64_t D_block_j_col_rank,
                            double *_D_block_j,
                            int64_t U_j_rows, int64_t U_j_cols,
                            double *_U_j,
                            int64_t F_ij_rows, int64_t F_ij_cols, int64_t F_ij_row_rank, int64_t F_ij_col_rank,
                            double *_F_ij);

  void CORE_fill_in_addition(  int64_t F_nrows,  int64_t F_ncols,
                               char which,
                               double *_F,
                               int64_t block_size,
                               double *_fill_in);

  void CORE_fill_in_recompression(  int64_t block_size,
                                    double *_fill_in,
                                    int64_t rank,
                                    double *_US,
                                    int64_t U_nrows, int64_t U_ncols,
                                    double *_U,
                                    int64_t proj_nrows,
                                    double *_proj,
                                    char which);

  void CORE_project_S(int64_t S_nrows,
                        double *_S,
                        int64_t proj_nrows,
                        double *_proj,
                        char which);

  void CORE_schurs_complement_1(int64_t D_block_block_nrows,
                                 int64_t D_block_rank,
                                 double *_D_block_block,
                                 int64_t D_i_block_nrows,
                                 int64_t D_i_block_ncols,
                               double *_D_i_block);

  void CORE_schurs_complement_3(int64_t D_block_block_nrows,
                               int64_t D_block_j_ncols,
                               int64_t D_block_rank,
                               int64_t D_j_rank,
                               double *_D_block_block, double *_D_block_j);

  void CORE_schurs_complement_2(int64_t D_i_block_nrows, int64_t D_i_block_ncols, int64_t D_block_rank,
                                double *_D_i_block,
                                int64_t D_j_block_nrows, int64_t D_j_block_ncols, double *_D_j_block,
                                int64_t D_i_j_nrows, int64_t D_i_j_ncols, double *_D_i_j);

  void CORE_syrk_2(int64_t D_i_block_nrows, int64_t D_i_block_ncols, int64_t D_block_rank,
                   double *_D_i_block,
                   int64_t D_i_j_nrows, int64_t D_i_j_ncols,
                   double *_D_i_j);

  void CORE_schurs_complement_4(int64_t D_i_dim, int64_t D_j_dim, int64_t D_block_dim,
                                int64_t A_i_rank, int64_t A_j_rank, int64_t A_block_rank,
                                double *_D_i_j, double *_D_block_j, double *_D_i_block);

  void CORE_transfer_basis_update(int64_t U_nrows, int64_t U_ncols, int64_t rank_c1, int64_t rank_c2,
                                  double *_proj_c1, double *_proj_c2, double *_U);

  void CORE_project_fill_in(int64_t nrows, int64_t ncols, int64_t rank_i, int64_t rank_j,
                            double *_Ui, double *_Uj, double *_Fij, double *_Sij);

#ifdef __cplusplus
}
#endif

#endif
