#pragma once

#include "franklin/franklin.hpp"

extern Hatrix::RowLevelMap US;

void
factorize_diagonal(Hatrix::SymmetricSharedBasisMatrix& A, int64_t block,
                   int64_t level);

void
solve_forward_level(const Hatrix::SymmetricSharedBasisMatrix& A,
                    Hatrix::Matrix& x_level,
                    const int64_t level);

void
solve_backward_level(const Hatrix::SymmetricSharedBasisMatrix& A,
                     Hatrix::Matrix& x_level,
                     const int64_t level);

void
triangle_reduction(Hatrix::SymmetricSharedBasisMatrix& A, int64_t block, int64_t level);

void
compute_schurs_complement(Hatrix::SymmetricSharedBasisMatrix& A, int64_t block, int64_t level);

void
merge_unfactorized_blocks(Hatrix::SymmetricSharedBasisMatrix& A, int64_t level);

long long int factorize(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts);
Hatrix::Matrix solve(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);

int64_t
permute_backward(const Hatrix::SymmetricSharedBasisMatrix& A,
                 Hatrix::Matrix& x, const int64_t level, int64_t rank_offset);

int64_t
permute_forward(const Hatrix::SymmetricSharedBasisMatrix& A,
                Hatrix::Matrix& x, int64_t level, int64_t permute_offset);

void
update_row_cluster_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                         Hatrix::RowMap<Hatrix::Matrix>& r,
                         const Hatrix::Args& opts);

void
update_col_cluster_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                         Hatrix::RowMap<Hatrix::Matrix>& t,
                         const Hatrix::Args& opts);

void
update_row_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& r);

void
update_col_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& t);

void
update_col_transfer_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t block, const int64_t level,
                          Hatrix::RowMap<Hatrix::Matrix>& t);

void
update_row_transfer_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t block, const int64_t level,
                          Hatrix::RowMap<Hatrix::Matrix>& r);

void
update_row_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& r,
                                      const Hatrix::Args& opts,
                                      const int64_t block,
                                      const int64_t level);

void
update_col_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& t,
                                      const Hatrix::Args& opts,
                                      const int64_t block,
                                      const int64_t level);

// matvec routine
Hatrix::Matrix matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);

void
compute_fill_ins(Hatrix::SymmetricSharedBasisMatrix& A, int64_t block,
                 int64_t level, Hatrix::RowColLevelMap<Hatrix::Matrix>& F);

void
multiply_complements(Hatrix::SymmetricSharedBasisMatrix& A, const int64_t block,
                     const int64_t level);
