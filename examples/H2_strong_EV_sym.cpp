#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"
#include "Domain.hpp"
#include "admissibility.hpp"
#include "functions.hpp"

using namespace Hatrix;
using vec = std::vector<int64_t>;

/*
  Generalized LDL Factorization of H2-Matrix.
  H2-Construction is done using the O(N^2) SVD based technique.
*/

namespace {

void generate_cluster_bases(SymmetricSharedBasisMatrix& A, RowLevelMap& Ubig,
                            const Domain& domain, const Admissibility::CellInteractionLists& interactions,
                            const double err_tol, const int64_t max_rank,
                            const bool is_rel_tol) {
  // Bottom up pass
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    #pragma omp parallel for
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      const auto ii = domain.get_cell_index(i, level);
      if (interactions.far_particles[ii].size() > 0) {  // If row has admissible blocks
        const auto& Ci = domain.cells[ii];
        Matrix far_blocks = generate_p2p_matrix(domain,
                                                Ci.get_bodies(),
                                                interactions.far_particles[ii]);
        if (level == A.max_level) {
          // Leaf level: direct SVD
          Matrix Ui, Si, Vi;
          int64_t rank;
          std::tie(Ui, Si, Vi, rank) = error_svd(far_blocks, err_tol, is_rel_tol, false);
          // Fixed-accuracy with bounded rank
          rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
          // Separate U into original and complement part
          auto Ui_splits = Ui.split(vec{}, vec{rank});
          Matrix Uo(Ui_splits[0], true);  // Deep-copy
          Matrix Uc = rank < Ui.rows ? Matrix(Ui_splits[1], true) : Matrix(Ui.rows, 0);
          // Save actual basis (Ubig) for upper level bases construction
          Matrix Ubig_i(Uo);
          // Insert
          #pragma omp critical
          {
            A.U.insert(i, level, std::move(Uo));
            A.Uc.insert(i, level, std::move(Uc));
            A.US_row.insert(i, level, matmul(Ui, Si)); // Save full basis for ULV update basis operation
            Ubig.insert(i, level, std::move(Ubig_i));
          }
        }
        else {
          // Non-leaf level: project with children's bases then SVD to generate transfer matrix
          // Note: this assumes balanced binary tree of cells
          const auto child_level = level + 1;
          const auto child1 = 2 * i + 0;
          const auto child2 = 2 * i + 1;
          const auto& Ubig_child1 = Ubig(child1, child_level);
          const auto& Ubig_child2 = Ubig(child2, child_level);
          Matrix proj_far_blocks(Ubig_child1.cols + Ubig_child2.cols, far_blocks.cols);
          auto far_blocks_splits = far_blocks.split(vec{Ubig_child1.rows}, {});
          auto proj_far_blocks_splits = proj_far_blocks.split(vec{Ubig_child1.cols}, {});
          matmul(Ubig_child1, far_blocks_splits[0], proj_far_blocks_splits[0], true, false, 1, 0);
          matmul(Ubig_child2, far_blocks_splits[1], proj_far_blocks_splits[1], true, false, 1, 0);
          Matrix Ui, Si, Vi;
          int64_t rank;
          std::tie(Ui, Si, Vi, rank) = error_svd(proj_far_blocks, err_tol, is_rel_tol, false);
          // Fixed-accuracy with bounded rank
          rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
          // Separate U into original and complement part
          auto Ui_splits = Ui.split(vec{}, vec{rank});
          Matrix Uo(Ui_splits[0], true);  // Deep-copy
          Matrix Uc = rank < Ui.rows ? Matrix(Ui_splits[1], true) : Matrix(Ui.rows, 0);
          // Save actual basis (Ubig) for upper level bases construction
          Matrix Ubig_i(Ubig_child1.rows + Ubig_child2.rows, Uo.cols);
          auto Uo_splits = Uo.split(vec{Ubig_child1.cols}, {});
          auto Ubig_i_splits = Ubig_i.split(vec{Ubig_child1.rows}, {});
          matmul(Ubig_child1, Uo_splits[0], Ubig_i_splits[0]);
          matmul(Ubig_child2, Uo_splits[1], Ubig_i_splits[1]);
          // Insert
          #pragma omp critical
          {
            A.U.insert(i, level, std::move(Uo));
            A.Uc.insert(i, level, std::move(Uc));
            A.US_row.insert(i, level, matmul(Ui, Si)); // Save full basis for ULV update basis operation
            Ubig.insert(i, level, std::move(Ubig_i));
          }
        }
      }
    }
  }
}

void generate_far_coupling_matrices(SymmetricSharedBasisMatrix& A, const RowLevelMap& Ubig,
                                    const Domain& domain) {
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    #pragma omp parallel for
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: A.admissible_cols(i, level)) {
        const Matrix Dij = generate_p2p_matrix(domain,
                                               domain.get_cell_index(i, level),
                                               domain.get_cell_index(j, level));
        Matrix Sij = matmul(matmul(Ubig(i, level), Dij, true, false), Ubig(j, level));
        #pragma omp critical
        {
          A.S.insert(i, j, level, std::move(Sij));
        }
      }
    }
  }
}

void generate_near_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                     const Domain& domain) {
  const int64_t level = A.max_level;  // Only generate inadmissible leaf blocks
  #pragma omp parallel for
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    for (int64_t j: A.inadmissible_cols(i, level)) {
      Matrix Dij = generate_p2p_matrix(domain,
                                       domain.get_cell_index(i, level),
                                       domain.get_cell_index(j, level));
      #pragma omp critical
      {
        A.D.insert(i, j, level, std::move(Dij));
      }
    }
  }
}

void construct_H2(SymmetricSharedBasisMatrix& A,
                  const Domain& domain, const double admis,
                  const double err_tol, const int64_t max_rank,
                  const bool is_rel_tol = false) {
  // Initialize cell interactions for admissibility
  Admissibility::CellInteractionLists interactions;
  Admissibility::build_cell_interactions(interactions, domain, admis);
  Admissibility::assemble_farfields(interactions, domain);
  // Initialize matrix block structure and admissibility
  Admissibility::init_block_structure(A, domain);
  Admissibility::init_geometry_admissibility(A, interactions, domain, admis);
  // Generate cluster bases and coupling matrices
  RowLevelMap Ubig;
  generate_cluster_bases(A, Ubig, domain, interactions, err_tol, max_rank, is_rel_tol);
  generate_far_coupling_matrices(A, Ubig, domain);
  generate_near_coupling_matrices(A, domain);
}

Matrix get_Ubig(const SymmetricSharedBasisMatrix& A,
                const int64_t i, const int64_t level) {
  if (level == A.max_level) {
    return A.U(i, level);
  }
  const int64_t child1 = i * 2 + 0;
  const int64_t child2 = i * 2 + 1;
  const Matrix Ubig_child1 = get_Ubig(A, child1, level + 1);
  const Matrix Ubig_child2 = get_Ubig(A, child2, level + 1);

  const int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, A.U(i, level).cols);
  auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
  auto U_splits = A.U(i, level).split(vec{Ubig_child1.cols}, vec{});
  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

double construction_error(const SymmetricSharedBasisMatrix& A,
                          const Domain& domain, const bool relative = false) {
  double dense_norm = 0;
  double diff_norm = 0;
  // Inadmissible blocks (only at leaf level)
  {
    const int64_t level = A.max_level;
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: A.inadmissible_cols(i, level)) {
        const Matrix Dij = generate_p2p_matrix(domain,
                                               domain.get_cell_index(i, level),
                                               domain.get_cell_index(j, level));
        const Matrix& Aij = A.D(i, j, level);
        const auto d_norm = norm(Dij);
        const auto diff = norm(Aij - Dij);
        dense_norm += d_norm * d_norm;
        diff_norm += diff * diff;
      }
    }
  }
  // Admissible blocks
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: A.admissible_cols(i, level)) {
        const Matrix Dij = generate_p2p_matrix(domain,
                                               domain.get_cell_index(i, level),
                                               domain.get_cell_index(j, level));
        const Matrix Ubig = get_Ubig(A, i, level);
        const Matrix Vbig = get_Ubig(A, j, level);
        const Matrix Aij = matmul(matmul(Ubig, A.S(i, j, level)), Vbig, false, true);
        const auto d_norm = norm(Dij);
        const auto diff = norm(Aij - Dij);
        dense_norm += d_norm * d_norm;
        diff_norm += diff * diff;
      }
    }
  }
  return (relative ? std::sqrt(diff_norm / dense_norm) : std::sqrt(diff_norm));
}

int64_t get_basis_min_rank(const SymmetricSharedBasisMatrix& A,
                           int64_t level_begin = 0,
                           int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  int64_t min_rank = std::numeric_limits<int64_t>::max();
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        min_rank = std::min(min_rank, A.U(i, level).cols);
      }
    }
  }
  return (min_rank == std::numeric_limits<int64_t>::max() ? -1 : min_rank);
}

int64_t get_basis_max_rank(const SymmetricSharedBasisMatrix& A,
                           int64_t level_begin = 0,
                           int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  int64_t max_rank = -1;
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        max_rank = std::max(max_rank, A.U(i, level).cols);
      }
    }
  }
  return max_rank;
}

double get_basis_avg_rank(const SymmetricSharedBasisMatrix& A,
                          int64_t level_begin = 0,
                          int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  double sum_rank = 0;
  double num_bases = 0;
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        sum_rank += static_cast<double>(A.U(i, level).cols);
        num_bases += 1.;
      }
    }
  }
  return sum_rank / num_bases;
}

// Return memory usage (in bytes)
int64_t get_memory_usage(const SymmetricSharedBasisMatrix& A) {
  int64_t mem = 0;
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        mem += A.U(i, level).memory_used();
      }
      if (A.US_row.exists(i, level)) {
        mem += A.US_row(i, level).memory_used();
      }
      for (auto j: A.inadmissible_cols(i, level)) {
        if (A.D.exists(i, j, level)) {
          mem += A.D(i, j, level).memory_used();
        }
      }
      for (auto j: A.admissible_cols(i, level)) {
        if (A.S.exists(i, j, level)) {
          mem += A.S(i, j, level).memory_used();
        }
      }
    }
  }
  return mem;
}

// ===== Begin LDL Factorization Functions =====
// Put identity bases when all dense row is encountered in a level
// Note: this is just to ensure that every cluster up to the min_adm_level has U entry
//       this step is not necessary if the condition above is guaranteed during H2-construction
void fill_empty_bases(SymmetricSharedBasisMatrix& A) {
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (!A.U.exists(i, level)) {
        // Use identity matrix as the cluster bases
        if (level == A.max_level) {
          const auto n = A.D(i, i, level).rows;
          A.U.insert(i, level, generate_identity_matrix(n, n));
	  A.Uc.insert(i, level, Matrix(n, 0));
	  A.US_row.insert(i, level, Matrix(n, 0));
        }
        else {
          const auto c1 = 2 * i + 0;
          const auto c2 = 2 * i + 1;
          const auto rank_c1 = A.U(c1, level + 1).cols;
          const auto rank_c2 = A.U(c2, level + 1).cols;
	  const auto n = rank_c1 + rank_c2;
          const auto rank_parent = std::max(rank_c1, rank_c2);

	  Matrix Id = generate_identity_matrix(n, n);
	  auto Id_splits = Id.split(vec{}, vec{rank_parent});
	  Matrix Uo(Id_splits[0], true);
	  Matrix Uc = rank_parent < n ? Matrix(Id_splits[1], true) : Matrix(n, 0);
          A.U.insert(i, level, std::move(Uo));
	  A.Uc.insert(i, level, std::move(Uc));
	  A.US_row.insert(i, level, Matrix(n, 0));
        }
      }
    }
  }
}

Matrix update_cluster_bases(SymmetricSharedBasisMatrix& A,
                            const RowColLevelMap<Matrix>& F,
                            const RowColMap<std::vector<int64_t>>& fill_in_cols,
                            const int64_t i, const int64_t level,
                            const double err_tol, const int64_t max_rank,
                            const bool is_rel_tol = false) {
  const int64_t block_size = A.D(i, i, level).rows;
  // Assemble low-rank blocks along the i-th row
  Matrix& lowrank_blocks = A.US_row(i, level);  // Use US_row saved from SVD-based construction
  assert(lowrank_blocks.rows == block_size);
  // Assemble fill-in blocks along the i-th row
  int64_t ncols = 0;
  std::vector<int64_t> col_splits;
  for (int64_t idx_j = 0; idx_j < fill_in_cols(i, level).size(); idx_j++) {
    const auto j = fill_in_cols(i, level)[idx_j];
    assert(F(i, j, level).rows == block_size);
    ncols += F(i, j, level).cols;
    col_splits.push_back(ncols);
  }
  col_splits.pop_back();  // Last column split index is unused
  Matrix fill_in_blocks(block_size, ncols);
  auto fill_in_blocks_splits = fill_in_blocks.split(vec{}, col_splits);
  for (int64_t idx_j = 0; idx_j < fill_in_cols(i, level).size(); idx_j++) {
    const auto j = fill_in_cols(i, level)[idx_j];
    fill_in_blocks_splits[idx_j] = F(i, j, level);
  }

  // Low-rank approximation of concat(LR, fill-in)
  Matrix Z = concat(lowrank_blocks, fill_in_blocks, 1);
  Matrix Ui, Si, Vi;
  int64_t rank;
  std::tie(Ui, Si, Vi, rank) = error_svd(Z, err_tol, is_rel_tol, false);
  // Fixed-accuracy with bounded rank
  rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
  // Separate U into original and complement part
  auto Ui_splits = Ui.split(vec{}, vec{rank});
  Matrix Uo(Ui_splits[0], true);  // Deep-copy
  Matrix Uc = rank < Ui.rows ? Matrix(Ui_splits[1], true) : Matrix(Ui.rows, 0);
  // Assemble projection matrix from old bases
  Matrix UTxU = matmul(Uo, A.U(i, level), true, false);
  // Erase existing
  A.U.erase(i, level);
  A.Uc.erase(i, level);
  // Insert
  A.U.insert(i, level, std::move(Uo));
  A.Uc.insert(i, level, std::move(Uc));

  return UTxU;
}

void project_far_coupling_matrices(SymmetricSharedBasisMatrix& A, const Matrix& P,
                                   const int64_t i, const int64_t level) {
  #pragma omp parallel for
  for (int64_t idx = 0; idx < A.admissible_cols(i, level).size(); idx++) {
    const auto j = A.admissible_cols(i, level)[idx];
    A.S(i, j, level) = matmul(P, A.S(i, j, level), false, false);
    A.S(j, i, level) = matmul(A.S(j, i, level), P, false, true);  // Symmetric
  }
}

void project_parent_transfer_matrix(SymmetricSharedBasisMatrix& A, const Matrix& P,
                                    const int64_t i, const int64_t level) {
  const int64_t parent = i / 2;
  const int64_t parent_level = level - 1;
  if (A.U.exists(parent, parent_level)) {
    const int64_t child1 = parent * 2 + 0;
    const int64_t child2 = parent * 2 + 1;
    const Matrix& U_child1 = A.U(child1, level);
    const Matrix& U_child2 = A.U(child2, level);
    const Matrix& U = A.U(parent, parent_level);
    const Matrix& US = A.US_row(parent, parent_level);

    Matrix U_new(U_child1.cols + U_child2.cols, U.cols);
    Matrix US_new(U_child1.cols + U_child2.cols, US.cols);
    auto U_new_splits  = U_new.split(vec{U_child1.cols},  {});
    auto US_new_splits = US_new.split(vec{U_child1.cols}, {});
    if (i == child1) {
      // Project transfer matrix of first child
      const auto U_splits = U.split(vec{P.cols}, {});
      matmul(P, U_splits[0], U_new_splits[0], false, false, 1, 0);
      U_new_splits[1] = U_splits[1];
      // Project saved US_row
      const auto US_splits = US.split(vec{P.cols}, {});
      matmul(P, US_splits[0], US_new_splits[0], false, false, 1, 0);
      US_new_splits[1] = US_splits[1];
    }
    else {  // i == child2
      // Project transfer matrix of second child
      const auto U_splits = U.split(vec{U_child1.cols}, {});
      U_new_splits[0] = U_splits[0];
      matmul(P, U_splits[1], U_new_splits[1], false, false, 1, 0);
      // Project saved US_row
      const auto US_splits = US.split(vec{U_child1.cols}, {});
      US_new_splits[0] = US_splits[0];
      matmul(P, US_splits[1], US_new_splits[1], false, false, 1, 0);
    }
    // Construct complement of projected basis
    const int64_t rank = U_new.cols;
    Matrix Q(U_new.rows, U_new.rows);
    Matrix R(U_new.rows, rank);
    Matrix U_copy(U_new);
    qr(U_copy, Q, R);
    auto Q_splits = Q.split(vec{}, vec{rank});
    Matrix Uc_new = rank < U_new.rows ? Matrix(Q_splits[1], true) : Matrix(U_new.rows, 0);
    // Erase existing
    A.U.erase(parent, parent_level);
    A.Uc.erase(parent, parent_level);
    A.US_row.erase(parent, parent_level);
    // Insert new
    A.U.insert(parent, parent_level, std::move(U_new));
    A.Uc.insert(parent, parent_level, std::move(Uc_new));
    A.US_row.insert(parent, parent_level, std::move(US_new));
  }
}

void apply_UF(SymmetricSharedBasisMatrix& A,
              const int64_t k, const int64_t level) {
  Matrix U_F = concat(A.Uc(k, level), A.U(k, level), 1);
  // Multiply to dense blocks along the row
  #pragma omp parallel for
  for (int64_t idx_j = 0; idx_j < A.inadmissible_cols(k, level).size(); idx_j++) {
    const auto j = A.inadmissible_cols(k, level)[idx_j];
    if (j < k) {
      // Do not touch the eliminated part (cc and oc)
      const auto left_col_split = A.D(k, j, level).cols - A.U(j, level).cols;
      auto D_splits = A.D(k, j, level).split(vec{}, vec{left_col_split});
      D_splits[1] = matmul(U_F, D_splits[1], true);
    }
    else {
      A.D(k, j, level) = matmul(U_F, A.D(k, j, level), true);
    }
  }
  // Multiply to dense blocks along the column
  #pragma omp parallel for
  for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
    const auto i = A.inadmissible_cols(k, level)[idx_i];
    if (i < k) {
      // Do not touch the eliminated part (cc and co)
      const auto top_row_split = A.D(i, k, level).rows - A.U(i, level).cols;
      auto D_splits = A.D(i, k, level).split(vec{top_row_split}, vec{});
      D_splits[1] = matmul(D_splits[1], U_F);
    }
    else {
      A.D(i, k, level) = matmul(A.D(i, k, level), U_F);
    }
  }
}

void partial_factorize_diagonal(SymmetricSharedBasisMatrix& A,
                                RowColLevelMap<Matrix>& F,
                                RowColMap<std::vector<int64_t>>& fill_in_cols,
                                const int64_t k, const int64_t level) {
  // Split diagonal block along the row and column
  Matrix& D_diag = A.D(k, k, level);
  const auto diag_c_size = D_diag.rows - A.U(k, level).cols;
  if (diag_c_size > 0) {
    auto D_diag_splits = D_diag.split(vec{diag_c_size}, vec{diag_c_size});
    Matrix& D_diag_cc = D_diag_splits[0];
    ldl(D_diag_cc);

    // Lower elimination
    #pragma omp parallel for
    for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
      const auto i = A.inadmissible_cols(k, level)[idx_i];
      Matrix& D_i = A.D(i, k, level);
      const auto lower_o_size =
          (i <= k || level == A.max_level) ? A.U(i, level).cols : A.U(i * 2, level + 1).cols;
      const auto lower_c_size = D_i.rows - lower_o_size;
      auto D_i_splits = D_i.split(vec{lower_c_size}, vec{diag_c_size});
      if (i > k && lower_c_size > 0) {
        Matrix& D_i_cc = D_i_splits[0];
        solve_triangular(D_diag_cc, D_i_cc, Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(D_diag_cc, D_i_cc, Hatrix::Right);
      }
      Matrix& D_i_oc = D_i_splits[2];
      solve_triangular(D_diag_cc, D_i_oc, Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(D_diag_cc, D_i_oc, Hatrix::Right);
    }

    // Right elimination
    #pragma omp parallel for
    for (int64_t idx_j = 0; idx_j < A.inadmissible_cols(k, level).size(); idx_j++) {
      const auto j = A.inadmissible_cols(k, level)[idx_j];
      Matrix& D_j = A.D(k, j, level);
      const auto right_o_size =
          (j <= k || level == A.max_level) ? A.U(j, level).cols : A.U(j * 2, level + 1).cols;
      const auto right_c_size = D_j.cols - right_o_size;
      auto D_j_splits  = D_j.split(vec{diag_c_size}, vec{right_c_size});
      if (j > k && right_c_size > 0) {
        Matrix& D_j_cc = D_j_splits[0];
        solve_triangular(D_diag_cc, D_j_cc, Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(D_diag_cc, D_j_cc, Hatrix::Left);
      }
      Matrix& D_j_co = D_j_splits[1];
      solve_triangular(D_diag_cc, D_j_co, Hatrix::Left, Hatrix::Lower, true, false);
      solve_diagonal(D_diag_cc, D_j_co, Hatrix::Left);
    }

    // Schur's complement into inadmissible block
    #pragma omp parallel for collapse(2)
    for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
      for (int64_t idx_j = 0; idx_j < A.inadmissible_cols(k, level).size(); idx_j++) {
        const auto i = A.inadmissible_cols(k, level)[idx_i];
        const auto j = A.inadmissible_cols(k, level)[idx_j];
        if (A.is_admissible.exists(i, j, level) && !A.is_admissible(i, j, level)) {
          const Matrix& D_i = A.D(i, k, level);
          const Matrix& D_j = A.D(k, j, level);
          const auto lower_o_size =
              (i <= k || level == A.max_level) ? A.U(i, level).cols : A.U(i * 2, level + 1).cols;
          const auto right_o_size =
              (j <= k || level == A.max_level) ? A.U(j, level).cols : A.U(j * 2, level + 1).cols;
          const auto lower_c_size = D_i.rows - lower_o_size;
          const auto right_c_size = D_j.cols - right_o_size;
          const auto D_i_splits  = D_i.split(vec{lower_c_size}, vec{diag_c_size});
          const auto D_j_splits  = D_j.split(vec{diag_c_size}, vec{right_c_size});
          auto D_ij_splits = A.D(i, j, level).split(vec{lower_c_size}, vec{right_c_size});

          const Matrix& D_j_cc = D_j_splits[0];
          const Matrix& D_j_co = D_j_splits[1];
          if (i > k && j > k && lower_c_size > 0 && right_c_size > 0) {
            // cc x cc -> cc
            Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
            Matrix& D_ij_cc = D_ij_splits[0];
            column_scale(D_i_cc, D_diag_cc);  // LD
            matmul(D_i_cc, D_j_cc, D_ij_cc, false, false, -1, 1);  // LDL^T
          }
          if (i > k && lower_c_size > 0) {
            // cc x co -> co
            Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
            Matrix& D_ij_co = D_ij_splits[1];
            column_scale(D_i_cc, D_diag_cc);  // LD
            matmul(D_i_cc, D_j_co, D_ij_co, false, false, -1, 1);  // LDL^T
          }
          if (j > k && right_c_size > 0) {
            // oc x cc -> oc
            Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
            Matrix& D_ij_oc = D_ij_splits[2];
            column_scale(D_i_oc, D_diag_cc);  // LD
            matmul(D_i_oc, D_j_cc, D_ij_oc, false, false, -1, 1);  // LDL^T
          }
          {
            // oc x co -> oo
            Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
            Matrix& D_ij_oo = D_ij_splits[3];
            column_scale(D_i_oc, D_diag_cc);  // LD
            matmul(D_i_oc, D_j_co, D_ij_oo, false, false, -1, 1);  // LDL^T
          }
        }
      }
    }

    // Schur's complement into admissible block (fill-in)
    #pragma omp parallel for collapse(2)
    for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
      for (int64_t idx_j = 0; idx_j < A.inadmissible_cols(k, level).size(); idx_j++) {
        const auto i = A.inadmissible_cols(k, level)[idx_i];
        const auto j = A.inadmissible_cols(k, level)[idx_j];
        const bool is_admissible_ij =
            !A.is_admissible.exists(i, j, level) ||
            (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level));
        const bool fill_ij =
            (i > k && j > k) ||  // b*b       fill-in block
            (i > k && j < k) ||  // b*rank    fill-in block
            (i < k && j > k) ||  // rank*b    fill-in block
            (i < k && j < k);    // rank*rank fill-in block
        if (is_admissible_ij && fill_ij) {
          const Matrix& D_i = A.D(i, k, level);
          const Matrix& D_j = A.D(k, j, level);
          const auto lower_o_size =
              (i <= k || level == A.max_level) ? A.U(i, level).cols : A.U(i * 2, level + 1).cols;
          const auto right_o_size =
              (j <= k || level == A.max_level) ? A.U(j, level).cols : A.U(j * 2, level + 1).cols;
          const auto lower_c_size = D_i.rows - lower_o_size;
          const auto right_c_size = D_j.cols - right_o_size;
          const auto D_i_splits  = D_i.split(vec{lower_c_size}, vec{diag_c_size});
          const auto D_j_splits  = D_j.split(vec{diag_c_size}, vec{right_c_size});

          Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
          Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
          column_scale(D_i_cc, D_diag_cc);
          column_scale(D_i_oc, D_diag_cc);
          const Matrix& D_j_cc = D_j_splits[0];
          const Matrix& D_j_co = D_j_splits[1];

          Matrix F_ij(D_i.rows, D_j.cols);
          if (i > k && j > k && lower_c_size > 0 && right_c_size > 0) {
            // Create b*b fill-in block
            Matrix fill_in(D_i.rows, D_j.cols);
            auto fill_in_splits = fill_in.split(vec{lower_c_size}, vec{right_c_size});
            matmul(D_i_cc, D_j_cc, fill_in_splits[0], false, false, -1, 1);  // Fill cc part
            matmul(D_i_cc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill co part
            matmul(D_i_oc, D_j_cc, fill_in_splits[2], false, false, -1, 1);  // Fill oc part
            matmul(D_i_oc, D_j_co, fill_in_splits[3], false, false, -1, 1);  // Fill oo part
            F_ij += fill_in;
          }
          if (i > k && j < k && lower_c_size > 0) {
            // Create b*rank fill-in block
            Matrix fill_in(D_i.rows, right_o_size);
            auto fill_in_splits = fill_in.split(vec{lower_c_size}, vec{});
            matmul(D_i_cc, D_j_co, fill_in_splits[0], false, false, -1, 1);  // Fill co part
            matmul(D_i_oc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill oo part
            // b*rank fill-in always has a form of Aik*Vk_c * inv(Akk_cc) x (Uk_c)^T*Akj*Vj_o
            // Convert to b*b block by applying (Vj_o)^T from right
            // Which is safe from bases update since j has been eliminated before (j < k)
            F_ij += matmul(fill_in, A.U(j, level), false, true);
          }
          if (i < k && j > k && right_c_size > 0) {
            // Create rank*b fill-in block
            Matrix fill_in(lower_o_size, D_j.cols);
            auto fill_in_splits = fill_in.split(vec{}, vec{right_c_size});
            matmul(D_i_oc, D_j_cc, fill_in_splits[0], false, false, -1, 1);  // Fill oc part
            matmul(D_i_oc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill oo part
            // rank*b fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj
            // Convert to b*b block by applying Ui_o from left
            // Which is safe from bases update since i has been eliminated before (i < k)
            F_ij += matmul(A.U(i, level), fill_in, false, false);
          }
          if (i < k && j < k) {
            // Create rank*rank fill-in block
            Matrix fill_in(lower_o_size, right_o_size);
            matmul(D_i_oc, D_j_co, fill_in, false, false, -1, 1);  // Fill oo part
            // rank*rank fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj*Vj_o
            // Convert to b*b block by applying Ui_o from left and (Vj_o)^T from right
            // Which is safe from bases update since i and j have been eliminated before (i,j < k)
            F_ij += matmul(matmul(A.U(i, level), fill_in),
                           A.U(j, level), false, true);
          }
          // Save or accumulate with existing fill-in block that has been propagated from lower level
          #pragma omp critical
          {
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(F_ij));
              fill_in_cols(i, level).push_back(j);
            }
            else {
              F(i, j, level) += F_ij;
            }
          }
        }
      }
    }
  } // if (diag_c_size > 0)
}

void add_fill_ins(SymmetricSharedBasisMatrix& A,
                  const RowColLevelMap<Matrix>& F,
                  const RowColMap<std::vector<int64_t>>& fill_in_cols,
                  const int64_t level) {
  // Add fill-in contribution to its corresponding far coupling matrix
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    for (int64_t j: fill_in_cols(i, level)) {
      assert(F.exists(i, j, level));
      A.S(i, j, level) += matmul(matmul(A.U(i, level), F(i, j, level), true),
                                 A.U(j, level));
    }
  }
}

void propagate_fill_ins(SymmetricSharedBasisMatrix& A,
                        RowColLevelMap<Matrix>& F,
                        RowColMap<std::vector<int64_t>>& fill_in_cols,
                        const int64_t level) {
  const int64_t parent_level = level - 1;
  // Propagate fill-in to upper level admissible blocks (if any)
  if (parent_level >= A.min_adm_level) {
    // Mark parent node that has fill-in coming from the current level
    // Use set instead of vector here to handle duplicates (i.e. fill-ins coming from both children)
    RowMap<std::set<int64_t>> parent_fill_in_cols;
    for (int64_t i = 0; i < A.level_nblocks[parent_level]; i++) {
      parent_fill_in_cols.insert(i, std::set<int64_t>());
    }
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: fill_in_cols(i, level)) {
        const int64_t ip = i / 2;
        const int64_t jp = j / 2;
        if ((!A.is_admissible.exists(ip, jp, parent_level)) ||
            (A.is_admissible.exists(ip, jp, parent_level) && A.is_admissible(ip, jp, parent_level))) {
          parent_fill_in_cols(ip).insert(jp);
        }
      }
    }
    for (int64_t i = 0; i < A.level_nblocks[parent_level]; i++) {
      for (int64_t j: parent_fill_in_cols(i)) {
        fill_in_cols(i, parent_level).push_back(j);
      }
    }
    // Propagate fill-ins to parent level
    for (int64_t i = 0; i < A.level_nblocks[parent_level]; ++i) {
      for (int64_t j: fill_in_cols(i, parent_level)) {
        const auto i1 = i * 2;
        const auto i2 = i * 2 + 1;
        const auto j1 = j * 2;
        const auto j2 = j * 2 + 1;
        const auto nrows = A.U(i1, level).cols + A.U(i2, level).cols;
        const auto ncols = A.U(j1, level).cols + A.U(j2, level).cols;
        Matrix fill_in(nrows, ncols);
        auto fill_in_splits = fill_in.split(vec{A.U(i1, level).cols},
                                            vec{A.U(j1, level).cols});
        if (F.exists(i1, j1, level)) {
          matmul(matmul(A.U(i1, level), F(i1, j1, level), true, false),
                 A.U(j1, level), fill_in_splits[0], false, false, 1, 0);
        }
        if (F.exists(i1, j2, level)) {
          matmul(matmul(A.U(i1, level), F(i1, j2, level), true, false),
                 A.U(j2, level), fill_in_splits[1], false, false, 1, 0);
        }
        if (F.exists(i2, j1, level)) {
          matmul(matmul(A.U(i2, level), F(i2, j1, level), true, false),
                 A.U(j1, level), fill_in_splits[2], false, false, 1, 0);
        }
        if (F.exists(i2, j2, level)) {
          matmul(matmul(A.U(i2, level), F(i2, j2, level), true, false),
                 A.U(j2, level), fill_in_splits[3], false, false, 1, 0);
        }
        F.insert(i, j, parent_level, std::move(fill_in));
      }
    }
  }
}

Matrix get_oo_part(const SymmetricSharedBasisMatrix& A,
                   const int64_t i, const int64_t j,
                   const int64_t level) {
  if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
    // Admissible block, use S block
    return A.S(i, j, level);
  }
  else {
    // Inadmissible block, use oo part of dense block
    const Matrix& Dij = A.D(i, j, level);
    const Matrix& Ui = A.U(i, level);
    const Matrix& Uj = A.U(j, level);
    auto Dij_splits = Dij.split(vec{Dij.rows - Ui.cols},
                                vec{Dij.cols - Uj.cols});
    return Dij_splits[3];
  }
}

void permute_and_merge(SymmetricSharedBasisMatrix& A,
                       const int64_t level) {
  const auto parent_level = level - 1;
  for (int64_t i = 0; i < A.level_nblocks[parent_level]; i++) {
    for (int64_t j: A.inadmissible_cols(i, parent_level)) {
      const auto i_c1 = i * 2 + 0;
      const auto i_c2 = i * 2 + 1;
      const auto j_c1 = j * 2 + 0;
      const auto j_c2 = j * 2 + 1;
      const auto nrows = A.U(i_c1, level).cols + A.U(i_c2, level).cols;
      const auto ncols = A.U(j_c1, level).cols + A.U(j_c2, level).cols;
      Matrix Dij(nrows, ncols);
      auto Dij_splits = Dij.split(vec{A.U(i_c1, level).cols},
                                  vec{A.U(j_c1, level).cols});
      Dij_splits[0] = get_oo_part(A, i_c1, j_c1, level);  // Dij_cc
      Dij_splits[1] = get_oo_part(A, i_c1, j_c2, level);  // Dij_co
      Dij_splits[2] = get_oo_part(A, i_c2, j_c1, level);  // Dij_oc
      Dij_splits[3] = get_oo_part(A, i_c2, j_c2, level);  // Dij_oo
      A.D.insert(i, j, parent_level, std::move(Dij));
    }
  }
}

void factorize_level(SymmetricSharedBasisMatrix& A,
                     RowColLevelMap<Matrix>& F,
                     RowColMap<std::vector<int64_t>>& fill_in_cols,
                     const int64_t level,
                     const double err_tol, const int64_t max_rank,
                     const bool is_rel_tol = false) {
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    // Update cluster basis if fill-in is found along the row/column
    if (fill_in_cols(k, level).size() > 0) {
      const Matrix PU = update_cluster_bases(A, F, fill_in_cols, k, level, err_tol, max_rank, is_rel_tol);
      project_far_coupling_matrices(A, PU, k, level);
      project_parent_transfer_matrix(A, PU, k, level);
    }
    apply_UF(A, k, level);
    partial_factorize_diagonal(A, F, fill_in_cols, k, level);
  }
}

void factorize_remaining_as_dense(SymmetricSharedBasisMatrix& A) {
  // Factorize remaining blocks (on min_adm_level-1) as block dense
  const auto level = A.min_adm_level - 1;
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    ldl(A.D(k, k, level));
    // Lower elimination
    #pragma omp parallel for
    for (int64_t i = k + 1; i < A.level_nblocks[level]; i++) {
      solve_triangular(A.D(k, k, level), A.D(i, k, level), Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(A.D(k, k, level), A.D(i, k, level), Hatrix::Right);
    }
    // Right elimination
    #pragma omp parallel for
    for (int64_t j = k + 1; j < A.level_nblocks[level]; j++) {
      solve_triangular(A.D(k, k, level), A.D(k, j, level), Hatrix::Left, Hatrix::Lower, true, false);
      solve_diagonal(A.D(k, k, level), A.D(k, j, level), Hatrix::Left);
    }
    // Schur's complement
    #pragma omp parallel for collapse(2)
    for (int64_t i = k + 1; i < A.level_nblocks[level]; i++) {
      for (int64_t j = k + 1; j < A.level_nblocks[level]; j++) {
        Matrix Dik(A.D(i, k, level), true);  // Deep-copy
        column_scale(Dik, A.D(k, k, level));  // LD
        matmul(Dik, A.D(k, j, level), A.D(i, j, level), false, false, -1, 1);
      }
    }
  }
}

void factorize(SymmetricSharedBasisMatrix& A,
               const double err_tol, const int64_t max_rank,
               const bool is_rel_tol = false) {
  // Preprocess: make sure all cluster bases exist and none of them is full-rank
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (!A.U.exists(i, level)) {
        throw std::logic_error("Cluster bases not found at A.U(" + std::to_string(i) +
                               "," + std::to_string(level) + ")");
	return;
      }
    }
  }

  // Initialize variables to handle fill-ins
  RowColLevelMap<Matrix> F;
  RowColMap<std::vector<int64_t>> fill_in_cols;
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      fill_in_cols.insert(i, level, std::vector<int64_t>());
    }
  }
  // LDL Factorize
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    factorize_level(A, F, fill_in_cols, level, err_tol, max_rank, is_rel_tol);
    add_fill_ins(A, F, fill_in_cols, level);
    propagate_fill_ins(A, F, fill_in_cols, level);
    permute_and_merge(A, level);
  }
  factorize_remaining_as_dense(A);
}
// ===== End   LDL Factorization Functions =====

// ===== Begin Slicing The Spectrum Functions =====
void shift_diagonal(SymmetricSharedBasisMatrix& A, const double value) {
  const int64_t level = A.max_level;
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    Matrix& Dkk = A.D(k, k, level);
    for (int64_t i = 0; i < Dkk.min_dim(); i++) {
      Dkk(i, i) += value;
    }
  }
}

int64_t count_ldl_negative_diagonal_entries(const SymmetricSharedBasisMatrix& LDL,
                                            bool& is_singular) {
  constexpr static double EPS = std::numeric_limits<double>::epsilon();
  int64_t count = 0;
  for (int64_t level = LDL.max_level; level >= LDL.min_adm_level; level--) {
    for (int64_t k = 0; k < LDL.level_nblocks[level]; k++) {
      const Matrix& Dkk = LDL.D(k, k, level);
      const auto rank = LDL.U(k, level).cols;
      const auto Dkk_splits = Dkk.split(vec{Dkk.rows - rank}, vec{Dkk.cols - rank});
      const Matrix& Dkk_cc = Dkk_splits[0];
      for(int64_t i = 0; i < Dkk_cc.min_dim(); i++) {
        count += (Dkk_cc(i, i) < 0 ? 1 : 0);
        if(std::isnan(Dkk_cc(i, i)) || std::abs(Dkk_cc(i, i)) < EPS) is_singular = true;
      }
    }
  }
  // Remaining blocks that are factorized as block-dense matrix
  {
    const auto level = LDL.min_adm_level - 1;
    for (int64_t k = 0; k < LDL.level_nblocks[level]; k++) {
      const Matrix& Dkk = LDL.D(k, k, level);
      for(int64_t i = 0; i < Dkk.min_dim(); i++) {
        count += (Dkk(i, i) < 0 ? 1 : 0);
        if(std::isnan(Dkk(i, i)) || std::abs(Dkk(i, i)) < EPS) is_singular = true;
      }
    }
  }
  return count;
}

int64_t inertia(const SymmetricSharedBasisMatrix& A, const double err_tol,
                const double lambda, bool& is_singular,
                int64_t& min_rank, int64_t& max_rank,
                int64_t& max_mem) {
  SymmetricSharedBasisMatrix M(A);
  shift_diagonal(M, -lambda);  // Shift diagonal entries
  factorize(M, err_tol, 0, false);  // LDL factorize
  // Get statistics after LDL factorization of shifted matrix
  const auto factor_min_rank = get_basis_min_rank(M);
  const auto factor_max_rank = get_basis_max_rank(M);
  const auto factor_mem = get_memory_usage(M);
  if (factor_max_rank > max_rank) {
    min_rank = factor_min_rank;
    max_rank = factor_max_rank;
    max_mem  = factor_mem;
  }
  return count_ldl_negative_diagonal_entries(M, is_singular);
}

double get_kth_eigenvalue(const SymmetricSharedBasisMatrix& A, const double err_tol,
                          const double ev_tol, const int64_t k,
                          const double a, const double b,
                          bool& is_singular,
                          int64_t& min_rank, int64_t& max_rank,
                          int64_t& max_mem) {
  double left = a;
  double right = b;
  while((right - left) >= ev_tol) {
    const auto mid = (left + right) / 2.;
    const auto v = inertia(A, err_tol, mid,
                           is_singular, min_rank, max_rank, max_mem);
    if (is_singular) {
      printf("Warning: Bisection is terminated because shifted matrix becomes singular (shift=%lf).\n", mid);
      break;
    }
    if(v >= k) right = mid;
    else left = mid;
  }
  return (left + right) / 2.;
}
// ===== End   Slicing The Spectrum Functions =====

}  // namespace

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  // err_tol == 0 means fixed rank
  const double err_tol = argc > 3 ? atof(argv[3]) : 1.e-8;
  // Use relative or absolute error threshold for LRA
  const bool is_rel_tol = argc > 4 ? (atol(argv[4]) == 1) : false;
  // Fixed accuracy with bounded rank
  const int64_t max_rank = argc > 5 ? atol(argv[5]) : 20;
  const double admis = argc > 6 ? atof(argv[6]) : 2;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: ELSES Dense Matrix
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  // 4: Random Uniform Grid
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim  = argc > 9 ? atol(argv[9]) : 1;

  // Eigenvalue computation parameters
  const double ev_tol = argc > 10 ? atof(argv[10]) : 1.e-3;
  int64_t k_begin = argc > 11 ? atol(argv[11]) : 1;
  int64_t k_end = argc > 12 ? atol(argv[12]) : k_begin;
  double a = argc > 13 ? atof(argv[13]) : 0.;
  double b = argc > 14 ? atof(argv[14]) : 1.;

  // ELSES Input Files
  const std::string file_name = argc > 15 ? std::string(argv[15]) : "";

  Hatrix::set_kernel_constants(1.e-3, 1.);
  std::string kernel_name = "";
  switch (kernel_type) {
    case 0: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
      break;
    }
    case 1: {
      Hatrix::set_kernel_function(Hatrix::yukawa_kernel);
      kernel_name = "yukawa";
      break;
    }
    case 2: {
      Hatrix::set_kernel_function(Hatrix::ELSES_dense_input);
      kernel_name = "ELSES-dense-file";
      break;
    }
    default: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
    }
  }

  Hatrix::Domain domain(N, ndim);
  std::string geom_name = std::to_string(ndim) + "d-";
  switch (geom_type) {
    case 0: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
      break;
    }
    case 1: {
      domain.initialize_unit_cubical_mesh();
      geom_name += "cubical_mesh";
      break;
    }
    case 2: {
      domain.initialize_starsh_uniform_grid();
      geom_name += "starsh_uniform_grid";
      break;
    }
    case 3: {
      domain.ndim = 3;
      const auto prefix_end = file_name.find_last_of("/\\");
      geom_name = file_name.substr(prefix_end + 1);
      break;
    }
    case 4: {
      domain.initialize_random_uniform_grid();
      geom_name += "random_uniform_grid";
      break;
    }
    default: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
    }
  }
  // Pre-processing step for ELSES geometry
  if (geom_type == 3) {
    const int64_t num_atoms_per_molecule = 60;
    const int64_t num_electrons_per_atom = kernel_type == 2 ? 4 : 1;
    const int64_t molecule_size = num_atoms_per_molecule * num_electrons_per_atom;
    assert(file_name.length() > 0);
    domain.read_bodies_ELSES(file_name + ".xyz", num_electrons_per_atom);
    assert(N == domain.N);

    domain.sort_bodies_ELSES(molecule_size);
    domain.build_tree_from_sorted_bodies(leaf_size, std::vector<int64_t>(N / leaf_size, leaf_size));
    if (kernel_type == 2) {
      domain.read_p2p_matrix_ELSES(file_name + ".dat");
    }
  }
  else {
    domain.build_tree(leaf_size);
  }

  SymmetricSharedBasisMatrix A;
  const auto start_construct = std::chrono::system_clock::now();
  construct_H2(A, domain, admis, err_tol, max_rank, is_rel_tol);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();

  const auto construct_error = construction_error(A, domain, is_rel_tol);
  const auto construct_min_rank = get_basis_min_rank(A);
  const auto construct_max_rank = get_basis_max_rank(A);
  const auto construct_avg_rank = get_basis_avg_rank(A);
  const auto construct_mem_bytes = get_memory_usage(A);

  const std::string err_prefix = (is_rel_tol ? "rel" : "abs");
  printf("N=%" PRId64 " leaf_size=%d %s_err_tol=%.1e max_rank=%d admis=%.2lf kernel=%s geometry=%s\n"
         "h2_height=%d construct_min_rank=%d construct_max_rank=%d construct_avg_rank=%.2lf "
         "construct_time=%e[ms] construct_mem=%e[GB] construct_%s_err=%e\n",
         N, (int)leaf_size, err_prefix.c_str(), err_tol, (int)max_rank, admis,
         kernel_name.c_str(), geom_name.c_str(),
         (int)A.max_level, (int)construct_min_rank, (int)construct_max_rank, construct_avg_rank,
         construct_time, static_cast<double>(construct_mem_bytes) * 1e-9, err_prefix.c_str(), construct_error);

  // Compute reference eigenvalues using LAPACK dsyev
  std::vector<double> ref_ev;
  double ref_ev_time = 0;
  const bool compute_ref_ev = (N <= 40000);
  if (compute_ref_ev) {
    Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
    const auto ref_ev_start = std::chrono::system_clock::now();
    ref_ev = Hatrix::get_eigenvalues(Adense);
    const auto ref_ev_stop = std::chrono::system_clock::now();
    ref_ev_time = std::chrono::duration_cast<std::chrono::milliseconds>
                  (ref_ev_stop - ref_ev_start).count();
  }

  // Check whether the given starting interval [a,b] contains the target eigenvalue(s)
  printf("ev_tol=%.1e k_begin=%d k_end=%d a=%lf b=%lf ref_ev_time=%e[ms]\n",
         ev_tol, (int)k_begin, (int)k_end, a, b, ref_ev_time);
  const auto ev_err_tol = err_tol;
  bool is_singular;
  int64_t factor_min_rank, factor_max_rank, factor_mem_bytes;
  const auto va = inertia(A, ev_err_tol, a, is_singular,
                          factor_min_rank, factor_max_rank, factor_mem_bytes);
  const auto vb = inertia(A, ev_err_tol, b, is_singular,
                           factor_min_rank, factor_max_rank, factor_mem_bytes);
  if (k_begin <= va || k_end > vb) {
    printf("Warning: starting interval [%lf,%lf] does not contain all target eigenvalues [%d,%d]\n",
           a, b, (int)k_begin, (int)k_end);
  }
  // Find target eigenvalue(s) with slicing-the-spectrum
  double h2_ev_total_time = 0;
  for (int64_t k = k_begin; k <= k_end; k++) {
    is_singular = false;
    factor_min_rank = construct_min_rank;
    factor_max_rank = construct_max_rank;
    factor_mem_bytes = construct_mem_bytes;
    const auto h2_ev_start = std::chrono::system_clock::now();
    const auto h2_ev_k = get_kth_eigenvalue(A, ev_err_tol, ev_tol, k, a, b,
                                            is_singular, factor_min_rank, factor_max_rank, factor_mem_bytes);
    const auto h2_ev_stop = std::chrono::system_clock::now();
    const double h2_ev_time = std::chrono::duration_cast<std::chrono::milliseconds>
                              (h2_ev_stop - h2_ev_start).count();
    h2_ev_total_time += h2_ev_time;
    const auto h2_ev_mem_bytes = construct_mem_bytes + factor_mem_bytes;
    const auto ref_ev_k = compute_ref_ev ? ref_ev[k-1] : 0;
    const auto h2_ev_abs_err = compute_ref_ev ? std::abs(h2_ev_k - ref_ev_k) : 0;
    const auto h2_ev_rel_err = compute_ref_ev ? (h2_ev_abs_err / ref_ev_k) : 0;
    const bool err_ok = h2_ev_abs_err < (0.5 * ev_tol);
    printf("k=%d factor_max_rank=%d h2_ev_k=%lf ref_ev_k=%lf "
           "h2_ev_time=%e[ms] h2_ev_mem=%e[GB] h2_ev_abs_err=%e[%s] h2_ev_rel_err=%e\n",
           (int)k, (int)factor_max_rank, h2_ev_k, ref_ev_k,
           h2_ev_time, static_cast<double>(h2_ev_mem_bytes) * 1e-9,
           h2_ev_abs_err, (err_ok ? "OK" : "NG"), h2_ev_rel_err);
  }
  printf("h2_ev_total_time=%e[ms]\n", h2_ev_total_time);

  return 0;
}

