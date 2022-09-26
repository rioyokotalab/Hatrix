#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <string>
#include <iomanip>
#include <functional>
#include <fstream>
#include <chrono>
#include <stdexcept>

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

using vec = std::vector<int64_t>;

/*
 * Note: the current Domain class is not designed for BLR2 since it assumes a balanced binary tree partition
 * where every cell has two children. However, we can enforce BLR2 structure by a simple workaround
 * that use only the leaf level cells. One thing to keep in mind is that
 * the leaf level in H2-matrix structure (leaf_level = height = 1) is different to
 * the actual leaf level of the domain partition tree (leaf_level = domain.tree_height).
 * This means that we have to adjust the level in some tasks that require cell information, such as:
 * - Getting cell index from (block_index, level)
 * - Generating p2p_matrix using block_index and level
 * See parts that involve matrix_type below
 */
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};

namespace Hatrix {

class SymmetricH2 {
 public:
  int64_t N, leaf_size;
  double accuracy;
  bool use_rel_acc;
  double ID_tolerance;
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U, R_row;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  std::vector<int64_t> level_blocks;
  RowColMap<std::vector<int64_t>> skeleton_rows;
  RowColMap<std::vector<int64_t>> ipiv_rows;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;

  void generate_row_cluster_basis(const Domain& domain);
  void generate_coupling_matrices(const Domain& domain);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;

  void pre_compute_fill_in(const int64_t level, RowColLevelMap<Matrix>& F) const;
  void reconstruct_row_cluster_basis(const Domain& domain,
                                     const int64_t level,
                                     const RowColLevelMap<Matrix>& F);
  void reconstruct_S_coupling_matrices(const Domain& domain, const int64_t level);
  Matrix get_skeleton(const Matrix& A,
                      const std::vector<int64_t>& skel_rows,
                      const std::vector<int64_t>& skel_cols) const;
  void propagate_upper_level_fill_in(const int64_t level, RowColLevelMap<Matrix>& F) const;
  void factorize_level(const int64_t level);

  int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  void solve_forward_level(Matrix& x_level, const int64_t level) const;
  void solve_diagonal_level(Matrix& x_level, const int64_t level) const;
  void solve_backward_level(Matrix& x_level, const int64_t level) const;

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const bool use_rel_acc,
              const int64_t max_rank, const double admis,
              const int64_t matrix_type);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double construction_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;

  void factorize(const Domain& domain);
  Matrix solve(const Matrix& b) const;
};

void SymmetricH2::initialize_geometry_admissibility(const Domain& domain) {
  if (matrix_type == H2_MATRIX) {
    height = domain.tree_height;
    level_blocks.assign(height + 1, 0);
    for (const auto& cell: domain.cells) {
      const auto level = cell.level;
      const auto i = cell.block_index;
      level_blocks[level]++;
      // Near interaction list: inadmissible dense blocks
      for (const auto near_idx: cell.near_list) {
        const auto j_near = domain.cells[near_idx].block_index;
        is_admissible.insert(i, j_near, level, false);
      }
      // Far interaction list: admissible low-rank blocks
      for (const auto far_idx: cell.far_list) {
        const auto j_far = domain.cells[far_idx].block_index;
        is_admissible.insert(i, j_far, level, true);
      }
    }
  }
  else if (matrix_type == BLR2_MATRIX) {
    height = 1;
    level_blocks.assign(height + 1, 0);
    level_blocks[0] = 1;
    level_blocks[1] = (int64_t)1 << domain.tree_height;
    is_admissible.insert(0, 0, 0, false);
    for (int64_t i = 0; i < level_blocks[height]; i++) {
      for (int64_t j = 0; j < level_blocks[height]; j++) {
        const auto level = domain.tree_height;
        const auto& source = domain.cells[domain.get_cell_idx(i, level)];
        const auto& target = domain.cells[domain.get_cell_idx(j, level)];
        is_admissible.insert(i, j, height, domain.is_well_separated(source, target, admis));
      }
    }
  }
}

int64_t SymmetricH2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  return domain.cells[idx].nbodies;
}

bool SymmetricH2::row_has_admissible_blocks(const int64_t row, const int64_t level) const {
  bool has_admis = false;
  for (int64_t j = 0; j < level_blocks[level]; j++) {
    if ((!is_admissible.exists(row, j, level)) || // part of upper level admissible block
        (is_admissible.exists(row, j, level) && is_admissible(row, j, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

void SymmetricH2::generate_row_cluster_basis(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (row_has_admissible_blocks(node, level)) {
        const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
        const auto idx = domain.get_cell_idx(node, node_level);
        const auto& cell = domain.cells[idx];
        std::vector<int64_t> skeleton;
        if (level == height) {
          // Leaf level: use all bodies as skeleton
          skeleton = cell.get_bodies();
        }
        else {
          // Non-leaf level: gather children's skeleton
          const auto& child1 = domain.cells[cell.child];
          const auto& child2 = domain.cells[cell.child + 1];
          const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
          const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
          skeleton.insert(skeleton.end(), child1_skeleton.begin(), child1_skeleton.end());
          skeleton.insert(skeleton.end(), child2_skeleton.begin(), child2_skeleton.end());
        }
        // Key to order N complexity is here:
        // The size of adm_block_row is always constant: ID_skeleton_size x sample_far_size
        Matrix adm_block_row = generate_p2p_matrix(domain, skeleton, cell.sample_farfield);
        // ID compress
        Matrix U_node;
        std::vector<int64_t> ipiv_row;
        std::tie(U_node, ipiv_row) = error_id_row(adm_block_row, ID_tolerance, use_rel_acc);
        int64_t rank = U_node.cols;
        if (max_rank > 0 && rank > max_rank) {
          // Truncate to max_rank
          U_node.shrink(U_node.rows, max_rank);
          rank = max_rank;
        }
        // Convert ipiv to node skeleton rows to be used by parent
        std::vector<int64_t> skel_rows;
        skel_rows.reserve(rank);
        for (int64_t i = 0; i < rank; i++) {
          skel_rows.push_back(skeleton[ipiv_row[i]]);
        }
        // Multiply U with child R
        if (level < height) {
          const auto& child1 = domain.cells[cell.child];
          const auto& child2 = domain.cells[cell.child + 1];
          const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
          const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
          auto U_node_splits = U_node.split(vec{(int64_t)child1_skeleton.size()}, vec{});
          triangular_matmul(R_row(child1.block_index, child1.level), U_node_splits[0],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(R_row(child2.block_index, child2.level), U_node_splits[1],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
        }
        // Orthogonalize basis with QR
        Matrix Q(U_node.rows, U_node.cols);
        Matrix R(U_node.cols, U_node.cols);
        qr(U_node, Q, R);
        U.insert(node, level, std::move(Q));
        R_row.insert(node, level, std::move(R));
        skeleton_rows.insert(node, level, std::move(skel_rows));
      }
    }
  }
}

void SymmetricH2::generate_coupling_matrices(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t i = 0; i < num_nodes; i++) {
      for (int64_t j = 0; j < num_nodes; j++) {
        // Inadmissible leaf blocks
        if (level == height &&
            is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
          D.insert(i, j, level, generate_p2p_matrix(domain, i, j, node_level));
        }
        // Admissible blocks
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const auto& skeleton_i = skeleton_rows(i, level);
          const auto& skeleton_j = skeleton_rows(j, level);
          Matrix skeleton_matrix = generate_p2p_matrix(domain, skeleton_i, skeleton_j);
          // Multiply with R from left and right
          triangular_matmul(R_row(i, level), skeleton_matrix,
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(R_row(j, level), skeleton_matrix,
                            Hatrix::Right, Hatrix::Upper, true, false, 1);
          S.insert(i, j, level, std::move(skeleton_matrix));
        }
      }
    }
  }
}

Matrix SymmetricH2::get_Ubig(const int64_t node, const int64_t level) const {
  if (level == height) {
    return U(node, level);
  }

  const int64_t child1 = node * 2;
  const int64_t child2 = node * 2 + 1;
  const Matrix Ubig_child1 = get_Ubig(child1, level + 1);
  const Matrix Ubig_child2 = get_Ubig(child2, level + 1);

  const int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, U(node, level).cols);
  auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
  auto U_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

SymmetricH2::SymmetricH2(const Domain& domain,
                         const int64_t N, const int64_t leaf_size,
                         const double accuracy, const bool use_rel_acc,
                         const int64_t max_rank, const double admis,
                         const int64_t matrix_type)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis), matrix_type(matrix_type) {
  // Set ID tolerance to be smaller than desired accuracy, based on HiDR paper source code
  // https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c#L859
  ID_tolerance = accuracy * 1e-1;
  initialize_geometry_admissibility(domain);
  generate_row_cluster_basis(domain);
  generate_coupling_matrices(domain);
}

int64_t SymmetricH2::get_basis_min_rank() const {
  int64_t rank_min = N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    for (int64_t node = 0; node < nblocks; node++) {
      if (U.exists(node, level)) {
        rank_min = std::min(rank_min, U(node, level).cols);
      }
    }
  }
  return rank_min;
}

int64_t SymmetricH2::get_basis_max_rank() const {
  int64_t rank_max = -N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    for (int64_t node = 0; node < nblocks; node++) {
      if (U.exists(node, level)) {
        rank_max = std::max(rank_max, U(node, level).cols);
      }
    }
  }
  return rank_max;
}

double SymmetricH2::construction_error(const Domain& domain) const {
  double dense_norm = 0;
  double diff_norm = 0;
  // Inadmissible blocks (only at leaf level)
  for (int64_t i = 0; i < level_blocks[height]; i++) {
    for (int64_t j = 0; j < level_blocks[height]; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
        const Matrix D_ij = Hatrix::generate_p2p_matrix(domain, i, j, node_level);
        const Matrix A_ij = D(i, j, height);
        const auto dnorm = norm(D_ij);
        const auto diff = norm(A_ij - D_ij);
        dense_norm += dnorm * dnorm;
        diff_norm += diff * diff;
      }
    }
  }
  // Admissible blocks
  for (int64_t level = height; level > 0; level--) {
    for (int64_t i = 0; i < level_blocks[level]; i++) {
      for (int64_t j = 0; j < level_blocks[level]; j++) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
          const Matrix D_ij = Hatrix::generate_p2p_matrix(domain, i, j, node_level);
          const Matrix Ubig = get_Ubig(i, level);
          const Matrix Vbig = get_Ubig(j, level);
          const Matrix A_ij = matmul(matmul(Ubig, S(i, j, level)), Vbig, false, true);
          const auto dnorm = norm(D_ij);
          const auto diff = norm(A_ij - D_ij);
          dense_norm += dnorm * dnorm;
          diff_norm += diff * diff;
        }
      }
    }
  }
  return (use_rel_acc ? std::sqrt(diff_norm / dense_norm) : std::sqrt(diff_norm));
}

void SymmetricH2::print_structure(const int64_t level) const {
  if (level == 0) { return; }
  const int64_t nblocks = level_blocks[level];
  std::cout << "LEVEL: " << level << " NBLOCKS: " << nblocks << std::endl;
  for (int64_t i = 0; i < nblocks; i++) {
    if (level == height && D.exists(i, i, height)) {
      std::cout << D(i, i, height).rows << " ";
    }
    std::cout << "| ";
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, level)) {
        std::cout << is_admissible(i, j, level) << " | " ;
      }
      else {
        std::cout << "  | ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  print_structure(level - 1);
}

void SymmetricH2::print_ranks() const {
  for(int64_t level = height; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    for(int64_t node = 0; node < nblocks; node++) {
      std::cout << "node=" << node << "," << "level=" << level << ":\t"
                << "diag= ";
      if(D.exists(node, node, level)) {
        std::cout << D(node, node, level).rows << "x" << D(node, node, level).cols;
      }
      else {
        std::cout << "empty";
      }
      std::cout << ", row_rank=" << (U.exists(node, level) ?
                                     U(node, level).cols : -1)
                << std::endl;
    }
  }
}

double SymmetricH2::low_rank_block_ratio() const {
  double total = 0, low_rank = 0;
  const int64_t nblocks = level_blocks[height];
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if ((is_admissible.exists(i, j, height) && is_admissible(i, j, height)) ||
          !is_admissible.exists(i, j, height)) {
        low_rank += 1;
      }
      total += 1;
    }
  }
  return low_rank / total;
}

void SymmetricH2::pre_compute_fill_in(const int64_t level,
                                      RowColLevelMap<Matrix>& F) const {
  const int64_t nblocks = level_blocks[level];
  for (int64_t k = 0; k < nblocks; k++) {
    Matrix Dkk = D(k, k, level);
    ldl(Dkk);
    for (int64_t i = 0; i < nblocks; i++) {
      if (i != k && is_admissible.exists(i, k, level) && !is_admissible(i, k, level)) {
        Matrix Dik = D(i, k, level);
        solve_triangular(Dkk, Dik, Hatrix::Right, Hatrix::Lower, true, true);
        for (int64_t j = 0; j < nblocks; j++) {
          if (j != k && is_admissible.exists(k, j, level) && !is_admissible(k, j, level)) {
            Matrix Dkj = D(k, j, level);
            solve_triangular(Dkk, Dkj, Hatrix::Left, Hatrix::Lower, true, false);
            solve_diagonal(Dkk, Dkj, Hatrix::Left);

            Matrix fill_in = matmul(Dik, Dkj, false, false, -1.0);
            if (F.exists(i, j, level)) {
              assert(F(i, j, level).rows == fill_in.rows);
              assert(F(i, j, level).cols == fill_in.cols);
              F(i, j, level) += fill_in;
            }
            else {
              F.insert(i, j, level, std::move(fill_in));
            }
          }
        }
      }
    }
  }
}

void SymmetricH2::reconstruct_row_cluster_basis(const Domain& domain,
                                                const int64_t level,
                                                const RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  RowMap<Matrix> r;
  for (int64_t i = 0; i < nblocks; i++) {
    const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
    const auto idx = domain.get_cell_idx(i, node_level);
    const auto& cell = domain.cells[idx];
    std::vector<int64_t> skeleton;
    if (level == height) {
      // Leaf level: use all bodies as skeleton
      skeleton = cell.get_bodies();
    }
    else {
      // Non-leaf level: gather children's skeleton
      const auto& child1 = domain.cells[cell.child];
      const auto& child2 = domain.cells[cell.child + 1];
      const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
      const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
      skeleton.insert(skeleton.end(), child1_skeleton.begin(), child1_skeleton.end());
      skeleton.insert(skeleton.end(), child2_skeleton.begin(), child2_skeleton.end());
    }
    const int64_t block_size = skeleton.size();
    Matrix block_row(block_size, 0);
    // Append low-rank part
    if (row_has_admissible_blocks(i, level)) {
      block_row = concat(block_row,
                         generate_p2p_matrix(domain, skeleton, cell.sample_farfield), 1);
    }
    // Append fill-in part
    for (int64_t j = 0; j < nblocks; j++) {
      if (F.exists(i, j, level)) {
        block_row = concat(block_row, F(i, j, level), 1);
      }
    }
    // ID compress
    Matrix Ui;
    std::vector<int64_t> ipiv_row;
    std::tie(Ui, ipiv_row) = error_id_row(block_row, ID_tolerance, use_rel_acc);
    int64_t rank = Ui.cols;
    if (max_rank > 0 && rank > max_rank) {
      // Truncate to max_rank
      Ui.shrink(Ui.rows, max_rank);
      rank = max_rank;
    }
    // Convert ipiv to node skeleton rows to be used by parent
    std::vector<int64_t> skel_rows;
    skel_rows.reserve(rank);
    for (int64_t i = 0; i < rank; i++) {
      skel_rows.push_back(skeleton[ipiv_row[i]]);
    }
    // Multiply U with child R
    if (level < height) {
      const auto& child1 = domain.cells[cell.child];
      const auto& child2 = domain.cells[cell.child + 1];
      const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
      const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
      auto Ui_splits = Ui.split(vec{(int64_t)child1_skeleton.size()}, vec{});
      triangular_matmul(R_row(child1.block_index, child1.level), Ui_splits[0],
                        Hatrix::Left, Hatrix::Upper, false, false, 1);
      triangular_matmul(R_row(child2.block_index, child2.level), Ui_splits[1],
                        Hatrix::Left, Hatrix::Upper, false, false, 1);
    }
    // Orthogonalize basis with QR
    Matrix Q(Ui.rows, Ui.cols);
    Matrix R(Ui.cols, Ui.cols);
    qr(Ui, Q, R);
    U.insert(i, level, std::move(Q));
    R_row.insert(i, level, std::move(R));
    skeleton_rows.insert(i, level, std::move(skel_rows));
    // Save ipiv_row for propagating fill-ins to upper level
    ipiv_row.resize(rank);
    ipiv_rows.insert(i, level, std::move(ipiv_row));
  }
}

void SymmetricH2::reconstruct_S_coupling_matrices(const Domain& domain, const int64_t level) {
  const auto nblocks = level_blocks[level];
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      // Admissible blocks
      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
        const auto& skeleton_i = skeleton_rows(i, level);
        const auto& skeleton_j = skeleton_rows(j, level);
        Matrix skeleton_matrix = generate_p2p_matrix(domain, skeleton_i, skeleton_j);
        // Multiply with R from left and right
        triangular_matmul(R_row(i, level), skeleton_matrix,
                          Hatrix::Left, Hatrix::Upper, false, false, 1);
        triangular_matmul(R_row(j, level), skeleton_matrix,
                          Hatrix::Right, Hatrix::Upper, true, false, 1);
        S.insert(i, j, level, std::move(skeleton_matrix));
      }
    }
  }
}

Matrix SymmetricH2::get_skeleton(const Matrix& A,
                                 const std::vector<int64_t>& skel_rows,
                                 const std::vector<int64_t>& skel_cols) const {
  Matrix out(skel_rows.size(), skel_cols.size());
  for (int64_t i = 0; i < out.rows; i++) {
    for (int64_t j = 0; j < out.cols; j++) {
      out(i, j) = A(skel_rows[i], skel_cols[j]);
    }
  }
  return out;
}

void SymmetricH2::propagate_upper_level_fill_in(const int64_t level,
                                                RowColLevelMap<Matrix>& F) const {
  const int64_t parent_level = level - 1;
  if (parent_level == 0) return;

  const int64_t parent_nblocks = level_blocks[parent_level];
  for (int64_t i = 0; i < parent_nblocks; i++) {
    for (int64_t j = 0; j < parent_nblocks; j++) {
      if ((!is_admissible.exists(i, j, parent_level)) ||
          (is_admissible.exists(i, j, parent_level) && is_admissible(i, j, parent_level))) {
        const int64_t i_c1 = i * 2 + 0;
        const int64_t i_c2 = i * 2 + 1;
        const int64_t j_c1 = j * 2 + 0;
        const int64_t j_c2 = j * 2 + 1;
        if (F.exists(i_c1, j_c1, level) || F.exists(i_c1, j_c2, level) ||
            F.exists(i_c2, j_c1, level) || F.exists(i_c2, j_c2, level)) {
          const int64_t nrows = U(i_c1, level).cols + U(i_c2, level).cols;
          const int64_t ncols = U(j_c1, level).cols + U(j_c2, level).cols;
          Matrix fill_in(nrows, ncols);
          auto fill_in_splits = fill_in.split(vec{U(i_c1, level).cols},
                                              vec{U(j_c1, level).cols});
          if (F.exists(i_c1, j_c1, level)) {
            fill_in_splits[0] = get_skeleton(F(i_c1, j_c1, level),
                                             ipiv_rows(i_c1, level),
                                             ipiv_rows(j_c1, level));
          }
          if (F.exists(i_c1, j_c2, level)) {
            fill_in_splits[1] = get_skeleton(F(i_c1, j_c2, level),
                                             ipiv_rows(i_c1, level),
                                             ipiv_rows(j_c2, level));
          }
          if (F.exists(i_c2, j_c1, level)) {
            fill_in_splits[2] = get_skeleton(F(i_c2, j_c1, level),
                                             ipiv_rows(i_c2, level),
                                             ipiv_rows(j_c1, level));
          }
          if (F.exists(i_c2, j_c2, level)) {
            fill_in_splits[3] = get_skeleton(F(i_c2, j_c2, level),
                                             ipiv_rows(i_c2, level),
                                             ipiv_rows(j_c2, level));
          }
          F.insert(i, j, parent_level, std::move(fill_in));
        }
      }
    }
  }
}

void SymmetricH2::factorize_level(const int64_t level) {
  const int64_t parent_level = level - 1;
  const int64_t nblocks = level_blocks[level];
  for (int64_t block = 0; block < nblocks; block++) {
    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;

    auto diagonal_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = block+1; i < nblocks; i++) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; i++) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = block+1; j < nblocks; j++) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        auto D_splits = D(block, j, level).split(vec{diag_row_split},
                                                 vec{D(block, j, level).cols - U(j, level).cols});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }
    // TRSM with co blocks on the row
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        auto D_splits = D(block, j, level).split(vec{diag_row_split},
                                                 vec{D(block, j, level).cols - U(j, level).cols});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[1], Hatrix::Left);
      }
    }

    // Schur's complement into own oo part
    // oc x co -> oo
    Matrix Doc(diagonal_splits[2], true);  // Deep-copy of view
    column_scale(Doc, Dcc);
    matmul(Doc, diagonal_splits[1], diagonal_splits[3], false, false, -1.0, 1.0);
  }  // for (int64_t block = 0; block < nblocks; block++)
}

void SymmetricH2::factorize(const Domain& domain) {
  int64_t level = height;
  RowColLevelMap<Matrix> F;

  // Clear basis, skeleton_rows, and coupling matrices
  U.erase_all();
  R_row.erase_all();
  skeleton_rows.erase_all();
  S.erase_all();

  for (; level > 0; level--) {
    pre_compute_fill_in(level, F);
    reconstruct_row_cluster_basis(domain, level, F);
    reconstruct_S_coupling_matrices(domain, level);
    propagate_upper_level_fill_in(level, F);

    const int64_t nblocks = level_blocks[level];
    // Multiply with (U_F)^T from left
    for (int64_t i = 0; i < nblocks; i++) {
      Matrix U_F = prepend_complement_basis(U(i, level));
      for (int64_t j = 0; j < nblocks; j++) {
        if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          D(i, j, level) = matmul(U_F, D(i, j, level), true);
        }
      }
    }
    // Multiply with U_F from right
    for (int64_t j = 0; j < nblocks; j++) {
      Matrix U_F = prepend_complement_basis(U(j, level));
      for (int64_t i = 0; i < nblocks; i++) {
        if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          D(i, j, level) = matmul(D(i, j, level), U_F);
        }
      }
    }
    factorize_level(level);

    const int64_t parent_level = level - 1;
    const int64_t parent_nblocks = level_blocks[parent_level];
    // Merge the unfactorized parts.
    for (int64_t i = 0; i < parent_nblocks; i++) {
      for (int64_t j = 0; j < parent_nblocks; j++) {
        if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
          // TODO: need to switch to morton indexing so finding the parent is straightforward.
          std::vector<int64_t> i_children, j_children;
          std::vector<int64_t> row_split, col_split;
          int64_t nrows = 0, ncols = 0;
          if (matrix_type == BLR2_MATRIX) {
            for (int64_t n = 0; n < level_blocks[level]; n++) {
              i_children.push_back(n);
              j_children.push_back(n);

              nrows += U(n, level).cols;
              ncols += U(n, level).cols;
              if(n < (level_blocks[level] - 1)) {
                row_split.push_back(nrows);
                col_split.push_back(ncols);
              }
            }
          }
          else if (matrix_type == H2_MATRIX) {
            for (int64_t n = 0; n < 2; n++) {
              int64_t ic = i * 2 + n;
              int64_t jc = j * 2 + n;
              i_children.push_back(ic);
              j_children.push_back(jc);

              nrows += U(ic, level).cols;
              ncols += U(jc, level).cols;
              if(n < 1) {
                row_split.push_back(nrows);
                col_split.push_back(ncols);
              }
            }
          }
          Matrix D_unelim(nrows, ncols);
          auto D_unelim_splits = D_unelim.split(row_split, col_split);

          for (int64_t ic1 = 0; ic1 < i_children.size(); ic1++) {
            for (int64_t jc2 = 0; jc2 < j_children.size(); jc2++) {
              int64_t c1 = i_children[ic1], c2 = j_children[jc2];

              if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {
                auto D_splits = D(c1, c2, level).split(
                    vec{D(c1, c2, level).rows - U(c1, level).cols},
                    vec{D(c1, c2, level).cols - U(c2, level).cols});
                D_unelim_splits[ic1 * j_children.size() + jc2] = D_splits[3];
              }
              else if (is_admissible.exists(c1, c2, level) && is_admissible(c1, c2, level)) {
                D_unelim_splits[ic1 * j_children.size() + jc2] = S(c1, c2, level);
              }
            }
          }

          D.insert(i, j, parent_level, std::move(D_unelim));
        }
      }
    }
  } // for (; level > 0; level--)

  // Factorize remaining root level
  ldl(D(0, 0, level));
}

// Permute the vector forward and return the offset at which the new vector begins.
int64_t SymmetricH2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const {
  Matrix copy(x);
  const int64_t nblocks = level_blocks[level];
  const int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < nblocks; block++) {
    rank_offset += D(block, block, level).rows - U(block, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < nblocks; block++) {
    const int64_t rows = D(block, block, level).rows;
    const int64_t rank = U(block, level).cols;
    const int64_t c_size = rows - rank;
    // Copy the complement part of the vector into the temporary vector
    for (int64_t i = 0; i < c_size; i++) {
      copy(c_offset + csize_offset + i, 0) = x(c_offset + bsize_offset + i, 0);
    }
    // Copy the rank part of the vector into the temporary vector
    for (int64_t i = 0; i < rank; i++) {
      copy(rank_offset + rsize_offset + i, 0) = x(c_offset + bsize_offset + c_size + i, 0);
    }

    csize_offset += c_size;
    bsize_offset += rows;
    rsize_offset += rank;
  }
  x = copy;
  return rank_offset;
}

// Permute the vector backward and return the offset at which the new vector begins
int64_t SymmetricH2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) const {
  Matrix copy(x);
  const int64_t nblocks = level_blocks[level];
  int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < nblocks; block++) {
    c_offset -= D(block, block, level).cols - U(block, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < nblocks; block++) {
    const int64_t cols = D(block, block, level).cols;
    const int64_t rank = U(block, level).cols;
    const int64_t c_size = cols - rank;

    for (int64_t i = 0; i < c_size; i++) {
      copy(c_offset + bsize_offset + i, 0) = x(c_offset + csize_offset + i, 0);
    }
    for (int64_t i = 0; i < rank; i++) {
      copy(c_offset + bsize_offset + c_size + i, 0) = x(rank_offset + rsize_offset + i, 0);
    }

    csize_offset += c_size;
    bsize_offset += cols;
    rsize_offset += rank;
  }
  x = copy;
  return c_offset;
}

void SymmetricH2::solve_forward_level(Matrix& x_level, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> row_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; i++) {
    row_offsets.push_back(nrows + D(i, i, level).rows);
    nrows += D(i, i, level).rows;
  }
  auto x_level_split = x_level.split(row_offsets, vec{});

  // Multiply with (U_F)^T at the beginning
  for (int64_t block = 0; block < nblocks; block++) {
    Matrix U_F = prepend_complement_basis(U(block, level));
    x_level_split[block] = matmul(U_F, x_level_split[block], true);
  }

  for (int64_t block = 0; block < nblocks; block++) {
    const int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    const int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;

    Matrix x_block(x_level_split[block], true);  // Deep-copy of a view
    auto x_block_splits = x_block.split(vec{diag_col_split}, vec{});
    // Solve forward with diagonal L
    auto L_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_triangular(L_block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
    // Forward substitution with oc block on the diagonal
    matmul(L_block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
    // Forward substitution with cc and oc blocks below the diagonal
    for (int64_t irow = block+1; irow < nblocks; irow++) {
      if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
        auto lower_splits = D(irow, block, level).split(vec{}, vec{diag_col_split});
        matmul(lower_splits[0], x_block_splits[0], x_level_split[irow], false, false, -1.0, 1.0);
      }
    }
    // Forward substitution with oc blocks above the diagonal
    for (int64_t irow = 0; irow < block; irow++) {
      if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
        const int64_t top_row_split = D(irow, block, level).rows - U(irow, level).cols;
        const int64_t top_col_split = diag_col_split;
        auto top_splits = D(irow, block, level).split(vec{top_row_split}, vec{top_col_split});

        Matrix x_irow(x_level_split[irow], true);  // Deep-copy of view
        auto x_irow_splits = x_irow.split(vec{top_row_split}, vec{});
        matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);
        x_level_split[irow] = x_irow;
      }
    }
    // Write x_block
    x_level_split[block] = x_block;
  }
}

void SymmetricH2::solve_diagonal_level(Matrix& x_level, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; i++) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  // Solve diagonal using cc blocks
  for (int64_t block = nblocks-1; block >= 0; block--) {
    const int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    const int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;

    Matrix x_block(x_level_split[block], true);  // Deep-copy of view
    auto x_block_splits = x_block.split(vec{diag_col_split}, {});
    // Solve with cc block on the diagonal
    auto D_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_diagonal(D_block_splits[0], x_block_splits[0], Hatrix::Left);
    // Write x_block
    x_level_split[block] = x_block;
  }
}

void SymmetricH2::solve_backward_level(Matrix& x_level, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; i++) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  for (int64_t block = nblocks-1; block >= 0; block--) {
    const int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    const int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(vec{diag_row_split}, vec{});
    // Backward substitution with co blocks in the left of diagonal
    for (int64_t jcol = block-1; jcol >= 0; jcol--) {
      if (is_admissible.exists(block, jcol, level) && !is_admissible(block, jcol, level)) {
        const int64_t left_row_split = diag_row_split;
        const int64_t left_col_split = D(block, jcol, level).cols - U(jcol, level).cols;
        auto left_splits = D(block, jcol, level).split(vec{left_row_split}, vec{left_col_split});

        Matrix x_jcol(x_level_split[jcol], true);  // Deep-copy of view
        auto x_jcol_splits = x_jcol.split(vec{left_col_split}, vec{});
        matmul(left_splits[1], x_jcol_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
      }
    }
    // Backward substitution with cc and co blocks in the right of diagonal
    for (int64_t jcol = nblocks-1; jcol > block; jcol--) {
      if (is_admissible.exists(block, jcol, level) && !is_admissible(block, jcol, level)) {
        auto right_splits = D(block, jcol, level).split(vec{diag_row_split}, vec{});
        matmul(right_splits[0], x_level_split[jcol], x_block_splits[0], false, false, -1.0, 1.0);
      }
    }
    // Solve backward with diagonal L^T
    auto L_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    matmul(L_block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
    solve_triangular(L_block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true, true);
    // Write x_block
    x_level_split[block] = x_block;
  }
  // Multiply with U_F at the end
  for (int64_t block = nblocks-1; block >= 0; block--) {
    Matrix U_F = prepend_complement_basis(U(block, level));
    x_level_split[block] = matmul(U_F, x_level_split[block]);
  }
}

Matrix SymmetricH2::solve(const Matrix& b) const {
  Matrix x(b);
  int64_t level = height;
  int64_t rhs_offset = 0;

  // Forward
  for (; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; i++) {
      nrows += D(i, i, level).rows;
    }

    Matrix x_level(nrows, 1);
    for (int64_t i = 0; i < x_level.rows; i++) {
      x_level(i, 0) = x(rhs_offset + i, 0);
    }
    solve_forward_level(x_level, level);
    for (int64_t i = 0; i < x_level.rows; i++) {
      x(rhs_offset + i, 0) = x_level(i, 0);
    }

    rhs_offset = permute_forward(x, level, rhs_offset);
  }

  // Solve with root level LDL
  auto x_splits = x.split(vec{rhs_offset}, vec{});
  const int64_t last_nodes = level_blocks[level];
  assert(level == 0);
  assert(last_nodes == 1);
  solve_triangular(D(0, 0, level), x_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
  solve_diagonal(D(0, 0, level), x_splits[1], Hatrix::Left);
  solve_triangular(D(0, 0, level), x_splits[1], Hatrix::Left, Hatrix::Lower, true, true);
  level++;

  // Backward
  for (; level <= height; level++) {
    const int64_t nblocks = level_blocks[level];

    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; i++) {
      nrows += D(i, i, level).cols;
    }
    Matrix x_level(nrows, 1);

    rhs_offset = permute_backward(x, level, rhs_offset);

    for (int64_t i = 0; i < x_level.rows; i++) {
      x_level(i, 0) = x(rhs_offset + i, 0);
    }
    solve_diagonal_level(x_level, level);
    solve_backward_level(x_level, level);
    for (int64_t i = 0; i < x_level.rows; i++) {
      x(rhs_offset + i, 0) = x_level(i, 0);
    }
  }

  return x;
}

} // namespace Hatrix

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-8;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const double admis = argc > 5 ? atof(argv[5]) : 3;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 6 ? atol(argv[6]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 7 ? atol(argv[7]) : 0;
  const int64_t ndim  = argc > 8 ? atol(argv[8]) : 1;

  // Specify sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net method
  const int64_t sampling_algo = argc > 9 ? atol(argv[9]) : 3;
  // Specify maximum number of sample points that represents a cluster
  // For anchor-net method: size of grid
  int64_t sample_self_size = argc > 10 ? atol(argv[10]) :
                             (ndim == 1 ? 10 * leaf_size : 30);
  // Specify maximum number of sample points that represents a cluster's far-field
  // For anchor-net method: size of grid
  int64_t sample_far_size = argc > 11 ? atol(argv[11]) :
                            (ndim == 1 ? 10 * leaf_size + 10: sample_self_size + 5);

  // Use relative or absolute error threshold
  const bool use_rel_acc = argc > 12 ? (atol(argv[12]) == 1) : false;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 13 ? atol(argv[13]) : 1;

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1e-3 / (double)N, 1.);
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
  }

  domain.build_tree(leaf_size);
  domain.build_interactions(admis);
  const auto start_sample = std::chrono::system_clock::now();
  domain.build_sample_bodies(sample_self_size, sample_far_size, sampling_algo);
  const auto stop_sample = std::chrono::system_clock::now();
  const double sample_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_sample - start_sample).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error = A.construction_error(domain);
  double lr_ratio = A.low_rank_block_ratio();
  A.print_structure(A.height);

  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " acc_type=" << (use_rel_acc ? "rel_err" : "abs_err")
            << " max_rank=" << max_rank
            << " admis=" << admis << std::setw(3)
            << " sampling_algo=" << sampling_algo_name
            << " sample_self_size=" << sample_self_size
            << " sample_far_size=" << sample_far_size
            << " sample_farfield_max_size=" << domain.get_max_farfield_size()
            << " compress_alg=" << "ID"
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " height=" << A.height
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " sample_time=" << sample_time
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error
            << std::defaultfloat << std::endl;

  const auto start_factor = std::chrono::system_clock::now();
  A.factorize(domain);
  const auto stop_factor = std::chrono::system_clock::now();
  const double factor_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_factor - start_factor).count();

  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix b = Hatrix::matmul(Adense, x);
  const auto solve_start = std::chrono::system_clock::now();
  Hatrix::Matrix x_solve = A.solve(b);
  const auto solve_stop = std::chrono::system_clock::now();
  const double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (solve_stop - solve_start).count();
  const auto diff_norm = Hatrix::norm(x_solve - x);
  const auto dense_norm = Hatrix::norm(x);
  const auto solve_error = use_rel_acc ? diff_norm / dense_norm : diff_norm;

  std::cout << "factor_min_rank=" << A.get_basis_min_rank()
            << " factor_max_rank=" << A.get_basis_max_rank()
            << " factor_time=" << factor_time
            << " solve_time=" << solve_time
            << " solve_error=" << std::scientific << solve_error
            << std::defaultfloat << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
