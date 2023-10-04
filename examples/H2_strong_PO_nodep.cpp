#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
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
  Generalized Cholesky Factorization of H2-Matrix without trailing dependencies.
  H2-Construction is done using the O(N) scheme with ID and hierarchical sample points.
*/

namespace {

std::vector<int64_t> get_skeleton_particles(const SymmetricSharedBasisMatrix& A,
                                            const Domain& domain,
                                            const int64_t i, const int64_t level) {
  std::vector<int64_t> skeleton;
  if (level == A.max_level) {
    // Leaf level: use all bodies as skeleton
    const auto& Ci = domain.cells[domain.get_cell_index(i, level)];
    skeleton = Ci.get_bodies();
  }
  else {
    // Non-leaf level: gather children's multipoles
    const auto child_level = level + 1;
    const auto child1 = 2 * i + 0;
    const auto child2 = 2 * i + 1;
    const auto& child1_multipoles = A.multipoles(child1, child_level);
    const auto& child2_multipoles = A.multipoles(child2, child_level);
    skeleton.insert(skeleton.end(), child1_multipoles.begin(), child1_multipoles.end());
    skeleton.insert(skeleton.end(), child2_multipoles.begin(), child2_multipoles.end());
  }
  return skeleton;
}

void generate_cluster_bases(SymmetricSharedBasisMatrix& A,
                            const Domain& domain,
                            const Admissibility::CellInteractionLists& interactions,
                            const double err_tol, const int64_t max_rank,
                            const bool is_rel_tol) {
  // Bottom up pass
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    #pragma omp parallel for
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      const auto ii = domain.get_cell_index(i, level);
      if (interactions.far_particles[ii].size() > 0) {  // If row has admissible blocks
        const auto skeleton = get_skeleton_particles(A, domain, i, level);
        // Key to order N complexity is here:
        // The size of far_blocks is always constant: rank x sample_size
        Matrix far_blocks = generate_p2p_matrix(domain, skeleton, interactions.far_particles[ii]);
        // LRA with ID
        Matrix Ui;
        std::vector<int64_t> ipiv_rows;
        std::tie(Ui, ipiv_rows) = error_id_row(far_blocks, err_tol, is_rel_tol);
        int64_t rank = Ui.cols;
        // Fixed-accuracy with bounded rank
        rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
        Ui.shrink(Ui.rows, rank);
        // Construct right factor (skeleton rows) from Interpolative Decomposition (ID)
        Matrix SV(rank, far_blocks.cols);
        for (int64_t r = 0; r < rank; r++) {
          for (int64_t c = 0; c < far_blocks.cols; c++) {
            SV(r, c) = far_blocks(ipiv_rows[r], c);
          }
        }
        Matrix Si(rank, rank);
        Matrix Vi(rank, SV.cols);
        rq(SV, Si, Vi);
        // Convert ipiv to multipoles to be used by parent
        std::vector<int64_t> multipoles_i;
        multipoles_i.reserve(rank);
        for (int64_t j = 0; j < rank; j++) {
          multipoles_i.push_back(skeleton[ipiv_rows[j]]);
        }
        // Multiply U with child R
        if (level < A.max_level) {
          const auto child_level = level + 1;
          const auto child1 = 2 * i + 0;
          const auto child2 = 2 * i + 1;
          const auto& child1_multipoles = A.multipoles(child1, child_level);
          auto Ui_splits = Ui.split(vec{(int64_t)child1_multipoles.size()}, {});
          triangular_matmul(A.R_row(child1, child_level), Ui_splits[0],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(A.R_row(child2, child_level), Ui_splits[1],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
        }
        // Orthogonalize basis with QR
        Matrix Q(Ui.rows, Ui.rows);
        Matrix R(Ui.rows, rank);
        Matrix Ui_copy(Ui);
        qr(Ui_copy, Q, R);
        // Separate Q into original and complement part
        auto Q_splits = Q.split(vec{}, vec{rank});
        Matrix Qo(Q_splits[0], true);
        Matrix Qc = rank < Ui.rows ? Matrix(Q_splits[1], true) : Matrix(Ui.rows, 0);
        R.shrink(rank, rank);
        // Insert
        #pragma omp critical
        {
          A.U.insert(i, level, std::move(Qo));
          A.Uc.insert(i, level, std::move(Qc));
          A.US_row.insert(i, level, matmul(Ui, Si));  // for ULV update basis operation
          A.R_row.insert(i, level, std::move(R));
          A.multipoles.insert(i, level, std::move(multipoles_i));
        }
      }
    }
  }
}

void generate_far_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                    const Domain& domain, const int64_t level) {
  #pragma omp parallel for
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    for (int64_t j: A.admissible_cols(i, level)) {
      Matrix Sij = generate_p2p_matrix(domain, A.multipoles(i, level), A.multipoles(j, level));
      // Multiply with R from left and right
      triangular_matmul(A.R_row(i, level), Sij,
                        Hatrix::Left, Hatrix::Upper, false, false, 1);
      triangular_matmul(A.R_row(j, level), Sij,
                        Hatrix::Right, Hatrix::Upper, true, false, 1);
      #pragma omp critical
      {
        A.S.insert(i, j, level, std::move(Sij));
      }
    }
  }
}

void generate_far_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                    const Domain& domain) {
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    generate_far_coupling_matrices(A, domain, level);
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
                  Admissibility::CellInteractionLists& interactions,
                  const Domain& domain, const double admis,
                  const double err_tol, const int64_t max_rank,
                  const int64_t sampling_algo,
                  const int64_t sample_local_size, const int64_t sample_far_size,
                  const bool is_rel_tol = false) {
  // Initialize cell interactions for admissibility
  Admissibility::build_cell_interactions(interactions, domain, admis);
  Admissibility::assemble_farfields_sample(interactions, domain,
                                           sampling_algo, sample_local_size, sample_far_size);
  // Initialize matrix block structure and admissibility
  Admissibility::init_block_structure(A, domain);
  Admissibility::init_geometry_admissibility(A, interactions, domain, admis);
  // Generate cluster bases and coupling matrices
  generate_cluster_bases(A, domain, interactions, err_tol, max_rank, is_rel_tol);
  generate_far_coupling_matrices(A, domain);
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

// ===== Begin Cholesky Factorization Functions =====
Matrix get_skeleton_matrix(const SymmetricSharedBasisMatrix& A,
                           const Domain& domain,
                           const int64_t i, const int64_t j, const int64_t level) {
  const auto skeleton_i = get_skeleton_particles(A, domain, i, level);
  const auto skeleton_j = get_skeleton_particles(A, domain, j, level);
  return generate_p2p_matrix(domain, skeleton_i, skeleton_j);
}

void precompute_fill_in(const SymmetricSharedBasisMatrix& A,
                        RowColLevelMap<Matrix>& F,
                        RowColMap<std::vector<int64_t>>& fill_in_cols,
                        const Domain& domain, const int64_t level) {
  #pragma omp parallel for
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    Matrix Dkk = get_skeleton_matrix(A, domain, k, k, level);
    cholesky(Dkk, Hatrix::Lower);
    for (int64_t j: A.inadmissible_cols(k, level)) {
      if (j != k) {
        // Compute fill_in = Djk * inv(Dkk)
        // Equivalent to: (fill_in)^T = inv(Dkk) * Djk^T = inv(Dkk) * Dkj
        Matrix Dkj = get_skeleton_matrix(A, domain, k, j, level);
        solve_triangular(Dkk, Dkj, Hatrix::Left, Hatrix::Lower, false, false, 1);
        solve_triangular(Dkk, Dkj, Hatrix::Left, Hatrix::Lower, false, true,  1);
        #pragma omp critical
        {
          F.insert(j, k, level, transpose(Dkj));
          fill_in_cols(j, level).push_back(k);
        }
      }
    }
  }
}

void generate_composite_bases(SymmetricSharedBasisMatrix& A,
                              const RowColLevelMap<Matrix>& F,
                              const RowColMap<std::vector<int64_t>>& fill_in_cols,
                              const Domain& domain,
                              const Admissibility::CellInteractionLists& interactions,
                              const int64_t level,
                              const double err_tol, const int64_t max_rank,
                              const bool is_rel_tol) {
  #pragma omp parallel for
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    const bool has_fill_in = (fill_in_cols(i, level).size() > 0);
    if (has_fill_in) {
      const auto skeleton = get_skeleton_particles(A, domain, i, level);
      const int64_t skeleton_size = skeleton.size();
      // Assemble low-rank blocks along the i-th row
      const auto ii = domain.get_cell_index(i, level);
      const Matrix lowrank_blocks = generate_p2p_matrix(domain, skeleton, interactions.far_particles[ii]);
      // Assemble fill-in blocks along the i-th row
      int64_t ncols = 0;
      std::vector<int64_t> col_splits;
      for (int64_t j: fill_in_cols(i, level)) {
        assert(F(i, j, level).rows == skeleton_size);
        ncols += F(i, j, level).cols;
        col_splits.push_back(ncols);
      }
      col_splits.pop_back();  // Last column split index is unused
      Matrix fill_in_blocks(skeleton_size, ncols);
      auto fill_in_blocks_splits = fill_in_blocks.split(vec{}, col_splits);
      int64_t idx = 0;
      for (int64_t j: fill_in_cols(i, level)) {
        fill_in_blocks_splits[idx++] = F(i, j, level);
      }
      // Low-rank approximation of concat(LR, fill-in)
      Matrix Z = concat(lowrank_blocks, fill_in_blocks, 1);
      // LRA with ID
      Matrix Ui;
      std::vector<int64_t> ipiv_rows;
      std::tie(Ui, ipiv_rows) = error_id_row(Z, err_tol, is_rel_tol);
      int64_t rank = Ui.cols;
      // Fixed-accuracy with bounded rank
      rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
      Ui.shrink(Ui.rows, rank);
      // Convert ipiv to multipoles to be used by parent
      std::vector<int64_t> multipoles_i;
      multipoles_i.reserve(rank);
      for (int64_t j = 0; j < rank; j++) {
        multipoles_i.push_back(skeleton[ipiv_rows[j]]);
      }
      // Multiply U with child R
      if (level < A.max_level) {
        const auto child_level = level + 1;
        const auto child1 = 2 * i + 0;
        const auto child2 = 2 * i + 1;
        const auto& child1_multipoles = A.multipoles(child1, child_level);
        auto Ui_splits = Ui.split(vec{(int64_t)child1_multipoles.size()}, {});
        triangular_matmul(A.R_row(child1, child_level), Ui_splits[0],
                          Hatrix::Left, Hatrix::Upper, false, false, 1);
        triangular_matmul(A.R_row(child2, child_level), Ui_splits[1],
                          Hatrix::Left, Hatrix::Upper, false, false, 1);
      }
      // Orthogonalize basis with QR
      Matrix Q(Ui.rows, Ui.rows);
      Matrix R(Ui.rows, rank);
      qr(Ui, Q, R);
      // Separate Q into original and complement part
      auto Q_splits = Q.split(vec{}, vec{rank});
      Matrix Qo(Q_splits[0], true);
      Matrix Qc = rank < Ui.rows ? Matrix(Q_splits[1], true) : Matrix(Ui.rows, 0);
      R.shrink(rank, rank);
      #pragma omp critical
      {
        // Erase existing
        A.U.erase(i, level);
        A.Uc.erase(i, level);
        A.R_row.erase(i, level);
        A.multipoles.erase(i, level);
        // Insert new
        A.U.insert(i, level, std::move(Qo));
        A.Uc.insert(i, level, std::move(Qc));
        A.R_row.insert(i, level, std::move(R));
        A.multipoles.insert(i, level, std::move(multipoles_i));
      }
    }
  }
}

void apply_UF(SymmetricSharedBasisMatrix& A,
              const int64_t k, const int64_t level) {
  const Matrix UF_k = concat(A.Uc(k, level), A.U(k, level), 1);
  // Multiply from both sides to dense blocks along the column
  for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
    const auto i = A.inadmissible_cols(k, level)[idx_i];
    const Matrix UF_i = concat(A.Uc(i, level), A.U(i, level), 1);
    A.D(i, k, level) = matmul(UF_i, matmul(A.D(i, k, level), UF_k), true, false);
  }
}

void partial_factorize_diagonal(SymmetricSharedBasisMatrix& A,
                                const int64_t k, const int64_t level) {
  // Split diagonal block along the row and column
  Matrix& D_diag = A.D(k, k, level);
  const auto diag_c_size = D_diag.rows - A.U(k, level).cols;
  if (diag_c_size > 0) {
    // Diagonal factorization
    auto D_diag_splits = D_diag.split(vec{diag_c_size}, vec{diag_c_size});
    Matrix& D_diag_cc = D_diag_splits[0];
    cholesky(D_diag_cc, Hatrix::Lower);
    // Lower elimination
    for (int64_t idx_i = 0; idx_i < A.inadmissible_cols(k, level).size(); idx_i++) {
      const auto i = A.inadmissible_cols(k, level)[idx_i];
      Matrix& D_i = A.D(i, k, level);
      const auto lower_o_size = A.U(i, level).cols;
      const auto lower_c_size = D_i.rows - lower_o_size;
      auto D_i_splits = D_i.split(vec{lower_c_size}, vec{diag_c_size});
      if (i > k && lower_c_size > 0) {
        Matrix& D_i_cc = D_i_splits[0];
        solve_triangular(D_diag_cc, D_i_cc, Hatrix::Right, Hatrix::Lower, false, true);
      }
      Matrix& D_i_oc = D_i_splits[2];
      solve_triangular(D_diag_cc, D_i_oc, Hatrix::Right, Hatrix::Lower, false, true);
    }
    // Schur's complement into into diagonal block
    Matrix& D_diag_oc = D_diag_splits[2];
    Matrix& D_diag_oo = D_diag_splits[3];
    matmul(D_diag_oc, D_diag_oc, D_diag_oo, false, true, -1, 1);
  } // if (diag_c_size > 0)
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
                     const int64_t level) {
  #pragma omp parallel for
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    apply_UF(A, k, level);
    partial_factorize_diagonal(A, k, level);
  }
}

void factorize(SymmetricSharedBasisMatrix& A,
               const Domain& domain,
               const Admissibility::CellInteractionLists& interactions,
               const double err_tol, const int64_t max_rank,
               const bool is_rel_tol = false) {
  // Preprocess
  A.S.erase_all();  // Clear far coupling matrices
  // Initialize variables to handle fill-ins
  RowColLevelMap<Matrix> F;
  RowColMap<std::vector<int64_t>> fill_in_cols;
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      fill_in_cols.insert(i, level, std::vector<int64_t>());
    }
  }
  // LDL Factorize
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    precompute_fill_in(A, F, fill_in_cols, domain, level);
    generate_composite_bases(A, F, fill_in_cols, domain, interactions, level,
                             err_tol, max_rank, is_rel_tol);
    generate_far_coupling_matrices(A, domain, level);
    factorize_level(A, level);
    permute_and_merge(A, level);
  }
  // Factorize remaining root level
  cholesky(A.D(0, 0, A.min_level-1), Hatrix::Lower);
}
// ===== End   Cholesky Factorization Functions =====

// ===== Begin Cholesky Solve Functions =====
void solve_forward(const SymmetricSharedBasisMatrix& A,
                   const int64_t level, const RowLevelMap& X,
                   RowLevelMap& Xc, RowLevelMap& Xo) {
  for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
    const auto use_k_c = (A.Uc(k, level).cols > 0);
    // Left Multiplication with (U_F)^T
    matmul(A.U(k, level) , X(k, level), Xo(k, level), true, false, 1, 1);
    if (use_k_c) {
      matmul(A.Uc(k, level), X(k, level), Xc(k, level), true, false, 1, 1);
      // Solve with diagonal block cc
      const auto D_k_splits = A.D(k, k, level).split(vec{A.Uc(k, level).cols},
                                                     vec{A.Uc(k, level).cols});
      const Matrix& D_k_cc = D_k_splits[0];
      const Matrix& D_k_oc = D_k_splits[2];
      solve_triangular(D_k_cc, Xc(k, level), Hatrix::Left, Hatrix::Lower, false, false);

      for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
        if (A.is_admissible.exists(i, k, level) && !A.is_admissible(i, k, level)) {
          const auto use_i_c = (A.Uc(i, level).cols > 0);
          const auto D_i_splits = A.D(i, k, level).split(vec{A.Uc(i, level).cols},
                                                         vec{A.Uc(k, level).cols});
          const Matrix& D_i_cc = D_i_splits[0];
          const Matrix& D_i_oc = D_i_splits[2];
          if (i > k && use_i_c) {
            matmul(D_i_cc, Xc(k, level), Xc(i, level), false, false, -1, 1);
          }
          matmul(D_i_oc, Xc(k, level), Xo(i, level), false, false, -1, 1);
        }
      }
    }
  }
}

void solve_backward(const SymmetricSharedBasisMatrix& A,
                    const int64_t level, RowLevelMap& X,
                    RowLevelMap& Xc, const RowLevelMap& Xo) {
  for (int64_t k = A.level_nblocks[level]-1; k >= 0; k--) {
    const auto use_k_c = (A.Uc(k, level).cols > 0);
    if (use_k_c) {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
        if (A.is_admissible.exists(i, k, level) && !A.is_admissible(i, k, level)) {
          const auto use_i_c = (A.Uc(i, level).cols > 0);
          const auto D_i_splits = A.D(i, k, level).split(vec{A.Uc(i, level).cols},
                                                         vec{A.Uc(k, level).cols});
          const Matrix& D_i_cc = D_i_splits[0];
          const Matrix& D_i_oc = D_i_splits[2];
          matmul(D_i_oc, Xo(i, level), Xc(k, level), true, false, -1, 1);
          if (i > k && use_i_c) {
            matmul(D_i_cc, Xc(i, level), Xc(k, level), true, false, -1, 1);
          }
        }
      }
      // Solve with diagonal block cc
      const auto D_k_splits = A.D(k, k, level).split(vec{A.Uc(k, level).cols},
                                                     vec{A.Uc(k, level).cols});
      const Matrix& D_k_cc = D_k_splits[0];
      const Matrix& D_k_oc = D_k_splits[2];
      solve_triangular(D_k_cc, Xc(k, level), Hatrix::Left, Hatrix::Lower, false, true);
      // Left Multiplication with U_F
      matmul(A.Uc(k, level), Xc(k, level), X(k, level), false, false, 1, 0);
    }
    matmul(A.U(k, level) , Xo(k, level), X(k, level), false, false, 1, 1);
  }
}

void permute_rhs(const SymmetricSharedBasisMatrix& A,
                 const char fwbk, const int64_t level,
                 RowLevelMap& Xo, RowLevelMap& X) {
  const auto parent_level = level - 1;
  for (int64_t k = 0; k < A.level_nblocks[parent_level]; k++) {
    const auto c1 = 2 * k + 0;
    const auto c2 = 2 * k + 1;
    auto X_k_splits = X(k, parent_level).split(vec{Xo(c1, level).rows}, vec{});
    if (fwbk == 'F' || fwbk == 'f') {
      X_k_splits[0] = Xo(c1, level);
      X_k_splits[1] = Xo(c2, level);
    }
    else if (fwbk == 'B' || fwbk == 'b') {
      Xo(c1, level) = X_k_splits[0];
      Xo(c2, level) = X_k_splits[1];
    }
  }
}

void solve(const SymmetricSharedBasisMatrix& A, Matrix& b) {
  RowLevelMap X, Xc, Xo, B;
  std::vector<int64_t> B_split_indices;

  // Allocate RHS
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    int64_t level_size = 0;
    for (int64_t k = 0; k < A.level_nblocks[level]; k++) {
      const auto c_size = A.Uc(k, level).cols;
      const auto o_size = A.U(k, level).cols;
      Xc.insert(k, level, Matrix(c_size, 1));
      Xo.insert(k, level, Matrix(o_size, 1));
      X.insert(k, level, Matrix(c_size + o_size, 1));
      B.insert(k, level, Matrix(c_size + o_size, 1));

      level_size += c_size + o_size;
      if (level == A.max_level &&
          k < (A.level_nblocks[level] - 1)) {
        B_split_indices.push_back(level_size);
      }
    }
  }
  // Allocate for root level
  {
    const int64_t level = A.min_level - 1;
    int64_t o_size = 0;
    for(int64_t k = 0; k < A.level_nblocks[A.min_level]; k++) {
      o_size += A.U(k, A.min_level).cols;
    }
    X.insert(0, level, Matrix(o_size, 1));
    B.insert(0, level, Matrix(o_size, 1));
  }

  // Initialize leaf level X with b
  auto b_splits = b.split(B_split_indices, vec{});
  for (int64_t k = 0; k < A.level_nblocks[A.max_level]; k++) {
    X(k, A.max_level) = b_splits[k];
  }
  // Solve Forward (Bottom-Up)
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    solve_forward(A, level, X, Xc, Xo);
    permute_rhs(A, 'F', level, Xo, X);
  }
  B(0, 0) = X(0, 0);
  solve_triangular(A.D(0, 0, 0), B(0, 0), Hatrix::Left, Hatrix::Lower, false, false);
  solve_triangular(A.D(0, 0, 0), B(0, 0), Hatrix::Left, Hatrix::Lower, false, true);
  // Solve Backward (Top-Down)
  for (int64_t level = A.min_level; level <= A.max_level; level++) {
    permute_rhs(A, 'B', level, Xo, B);
    solve_backward(A, level, B, Xc, Xo);
  }
  // Overwrite b with result
  for (int64_t k = 0; k < A.level_nblocks[A.max_level]; k++) {
    b_splits[k] = B(k, A.max_level);
  }
}

double solve_error(const Matrix& x, const Matrix& ref,
                   const bool is_rel_tol = false) {
  double diff_norm = 0.;
  double ref_norm = 0.;
  for (int64_t j = 0; j < x.cols; j++) {
    for (int64_t i = 0; i < x.rows; i++) {
      const auto diff = x(i, j) - ref(i, j);
      diff_norm += diff * diff;
      ref_norm += ref(i, j) * ref(i, j);
    }
  }
  return std::sqrt(diff_norm / (is_rel_tol ? ref_norm : 1.));
}
// ===== End   Cholesky Solve Functions =====

Matrix body_neutral_charge(const Domain& domain,
                           const double cmax, const int64_t seed) {
  if (seed > 0)
    srand(seed);

  const auto nbodies = domain.N;
  Matrix X(nbodies, 1);
  double avg = 0.;
  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++) {
    double c = ((double)rand() / RAND_MAX) * cmax2 - cmax;
    X(i, 0) = c;
    avg = avg + c;
  }
  avg = avg / nbodies;

  if (avg != 0.)
    for (int64_t i = 0; i < nbodies; i++)
      X(i, 0) -= avg;
  return X;
}

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

  // Specify sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net method
  const int64_t sampling_algo = argc > 10 ? atol(argv[10]) : 3;
  // Specify maximum number of sample points that represents a cluster
  // For anchor-net method: size of grid
  int64_t sample_local_size = argc > 11 ? atol(argv[11]) :
                              (ndim == 1 ? 10 * leaf_size : 30);
  // Specify maximum number of sample points that represents a cluster's far-field
  // For anchor-net method: size of grid
  int64_t sample_far_size = argc > 12 ? atol(argv[12]) :
                            (ndim == 1 ? 10 * leaf_size + 10: sample_local_size + 5);

  // ELSES Input Files
  const std::string file_name = argc > 13 ? std::string(argv[13]) : "";

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
  std::string sampling_algo_name = "";
  switch (sampling_algo) {
    case 0: {
      sampling_algo_name = "equally_spaced_indices";
      break;
    }
    case 1: {
      sampling_algo_name = "random_indices";
      break;
    }
    case 2: {
      sampling_algo_name = "farthest_point_sampling";
      break;
    }
    case 3: {
      sampling_algo_name = "anchor_net";
      break;
    }
    default: {
      sampling_algo_name = "no_sample";
      break;
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
  Admissibility::CellInteractionLists interactions;
  const auto start_construct = std::chrono::system_clock::now();
  construct_H2(A, interactions, domain, admis, err_tol, max_rank,
               sampling_algo, sample_local_size, sample_far_size, is_rel_tol);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  // Count maximum farfield size in interaction lists
  int64_t max_farfield_size = -1;
  for (const auto& farfield: interactions.far_particles) {
    max_farfield_size = std::max(max_farfield_size, (int64_t)farfield.size());
  }
  const auto construct_error = construction_error(A, domain, is_rel_tol);
  const auto construct_min_rank = get_basis_min_rank(A);
  const auto construct_max_rank = get_basis_max_rank(A);
  const auto construct_avg_rank = get_basis_avg_rank(A);
  const auto construct_mem = static_cast<double>(get_memory_usage(A)) * 1e-9;

  const std::string err_prefix = (is_rel_tol ? "rel" : "abs");
  printf("N=%" PRId64 " leaf_size=%d %s_err_tol=%.1e max_rank=%d admis=%.2lf kernel=%s geometry=%s\n"
         "sampling_algo=%s sample_local_size=%d sample_far_size=%d max_farfield_size=%" PRId64 "\n"
         "h2_height=%d construct_min_rank=%d construct_max_rank=%d construct_avg_rank=%.2lf "
         "construct_time=%e[ms] construct_mem=%e[GB] construct_%s_err=%e\n",
         N, (int)leaf_size, err_prefix.c_str(), err_tol, (int)max_rank, admis,
         kernel_name.c_str(), geom_name.c_str(),
         sampling_algo_name.c_str(), (int)sample_local_size, (int)sample_far_size, max_farfield_size,
         (int)A.max_level, (int)construct_min_rank, (int)construct_max_rank, construct_avg_rank,
         construct_time, construct_mem, err_prefix.c_str(), construct_error);

  const auto start_factor = std::chrono::system_clock::now();
  factorize(A, domain, interactions, err_tol, max_rank, is_rel_tol);
  const auto stop_factor = std::chrono::system_clock::now();
  const double factor_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_factor - start_factor).count();
  const auto factor_min_rank = get_basis_min_rank(A);
  const auto factor_max_rank = get_basis_max_rank(A);
  const auto factor_avg_rank = get_basis_avg_rank(A);
  const auto factor_mem = static_cast<double>(get_memory_usage(A)) * 1e-9;
  printf("factor_min_rank=%d factor_max_rank=%d factor_avg_rank=%.2lf "
         "factor_time=%e[ms] factor_mem=%e[GB]\n",
         (int)factor_min_rank, (int)factor_max_rank, factor_avg_rank,
         factor_time, factor_mem);

  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  Hatrix::Matrix x = body_neutral_charge(domain, 1, 0);
  Hatrix::Matrix b = Hatrix::matmul(Adense, x);
  const auto solve_start = std::chrono::system_clock::now();
  solve(A, b);
  const auto solve_stop = std::chrono::system_clock::now();
  const double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (solve_stop - solve_start).count();
  const auto solve_err = solve_error(b, x, is_rel_tol);
  printf("solve_time=%e[ms] solve_%s_error=%e\n",
         solve_time, err_prefix.c_str(), solve_err);

  return 0;
}
