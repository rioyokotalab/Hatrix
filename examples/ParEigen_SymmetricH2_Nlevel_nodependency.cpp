#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

constexpr double EPS = std::numeric_limits<double>::epsilon();
using vec = std::vector<int64_t>;

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV
// Uncomment the following line to enable debug
// #define DEBUG_OUTPUT
// Uncomment the following line to enable timer
// #define USE_TIMER
// Comment the following line to use ID (without SVD) for basis construction
// #define USE_SVD_COMPRESSION
// Uncomment the following to ignore L factor of overall LDL (since only D is needed for H2-Bisection)
#define IGNORE_L_FACTOR

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

void shift_diag(Hatrix::Matrix& A, const double shift) {
  for(int64_t i = 0; i < A.min_dim(); i++) {
    A(i, i) += shift;
  }
}

namespace Hatrix {

class SymmetricH2 {
 public:
  int64_t N, leaf_size;
  double accuracy;
  bool use_rel_acc;
  double err_tol;
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
  int64_t min_adm_level;
  RowLevelMap U, Uc, R_row;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowColMap<std::vector<int64_t>> near_neighbors, far_neighbors;  // This is actually RowLevelMap
  std::vector<int64_t> level_blocks;
  RowColMap<std::vector<int64_t>> multipoles;
  RowColLevelMap<Matrix> F;  // Fill-in blocks

 private:
  void initialize_geometry_admissibility(const Domain& domain);
  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  std::vector<int64_t> get_skeleton(const Domain& domain,
                                    const int64_t node, const int64_t level) const;
  void generate_row_cluster_basis(const Domain& domain, const int64_t level,
                                  const bool include_fill_in);
  void generate_near_coupling_matrices(const Domain& domain);
  void generate_far_coupling_matrices(const Domain& domain, const int64_t level);
  Matrix get_Ubig(const int64_t node, const int64_t level) const;

  Matrix get_dense_skeleton(const Domain& domain,
                            const int64_t i, const int64_t j, const int64_t level) const;
  Matrix get_oo(const int64_t i, const int64_t j, const int64_t level) const;
  void compute_fill_in(const Domain& domain, const int64_t level, const double diag_shift);
  void factorize_level(const int64_t level);
  void permute_and_merge(const int64_t level);

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const bool use_rel_acc,
              const int64_t max_rank, const double admis,
              const int64_t matrix_type, const bool build_basis);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double get_basis_avg_rank() const;
  double construction_error(const Domain& domain) const;
  int64_t memory_usage() const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;

  void factorize(const Domain& domain, const double diag_shift = 0);
};

void SymmetricH2::initialize_geometry_admissibility(const Domain& domain) {
  min_adm_level = -1;
  if (matrix_type == H2_MATRIX) {
    height = domain.tree_height;
    level_blocks.assign(height + 1, 0);
    for (const auto& cell: domain.cells) {
      const auto level = cell.level;
      const auto i = cell.block_index;
      level_blocks[level]++;
      // Near interaction list: inadmissible dense blocks
      near_neighbors.insert(i, level, std::vector<int64_t>());
      for (const auto near_idx: cell.near_list) {
        const auto j_near = domain.cells[near_idx].block_index;
        is_admissible.insert(i, j_near, level, false);
        near_neighbors(i, level).push_back(j_near);
      }
      // Far interaction list: admissible low-rank blocks
      far_neighbors.insert(i, level, std::vector<int64_t>());
      for (const auto far_idx: cell.far_list) {
        const auto j_far = domain.cells[far_idx].block_index;
        is_admissible.insert(i, j_far, level, true);
        far_neighbors(i, level).push_back(j_far);
      }
      if ((min_adm_level == -1) && (far_neighbors(i, level).size() > 0)) {
        min_adm_level = level;
      }
    }
  }
  else if (matrix_type == BLR2_MATRIX) {
    height = 1;
    level_blocks.assign(height + 1, 0);
     // Root level
    level_blocks[0] = 1;
    is_admissible.insert(0, 0, 0, false);
    near_neighbors.insert(0, 0, std::vector<int64_t>(1, 0));
    far_neighbors.insert(0, 0, std::vector<int64_t>());
    // Subdivide into BLR
    level_blocks[1] = (int64_t)1 << domain.tree_height;
    for (int64_t i = 0; i < level_blocks[height]; i++) {
      near_neighbors.insert(i, height, std::vector<int64_t>());
      far_neighbors.insert(i, height, std::vector<int64_t>());
      for (int64_t j = 0; j < level_blocks[height]; j++) {
        const auto level = domain.tree_height;
        const auto& source = domain.cells[domain.get_cell_idx(i, level)];
        const auto& target = domain.cells[domain.get_cell_idx(j, level)];
        is_admissible.insert(i, j, height, domain.is_well_separated(source, target, admis));
        if (is_admissible(i, j, height)) {
          far_neighbors(i, height).push_back(j);
        }
        else {
          near_neighbors(i, height).push_back(j);
        }
      }
    }
    min_adm_level = height;
  }
}

int64_t SymmetricH2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  return domain.cells[idx].nbodies;
}

std::vector<int64_t> SymmetricH2::get_skeleton(const Domain& domain,
                                               const int64_t node, const int64_t level) const {
  const auto node_level = (matrix_type == BLR2_MATRIX && level == height) ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  const auto& cell = domain.cells[idx];
  std::vector<int64_t> skeleton;
  if (level == height) {
    // Leaf level: use all bodies
    skeleton = cell.get_bodies();
  }
  else {
    if (matrix_type == BLR2_MATRIX) {
      // BLR2 Root: Gather multipoles of all leaf level nodes
      const auto nleaf_nodes = level_blocks[height];
      for (int64_t child = 0; child < nleaf_nodes; child++) {
        const auto& child_multipoles = multipoles(child, height);
        skeleton.insert(skeleton.end(), child_multipoles.begin(), child_multipoles.end());
      }
    }
    else {
      // H2 Non-Leaf: Gather children's multipoles
      const auto& child1 = domain.cells[cell.child];
      const auto& child2 = domain.cells[cell.child + 1];
      const auto& child1_multipoles = multipoles(child1.block_index, child1.level);
      const auto& child2_multipoles = multipoles(child2.block_index, child2.level);
      skeleton.insert(skeleton.end(), child1_multipoles.begin(), child1_multipoles.end());
      skeleton.insert(skeleton.end(), child2_multipoles.begin(), child2_multipoles.end());
    }
  }
  return skeleton;
}

void SymmetricH2::generate_row_cluster_basis(const Domain& domain,
                                             const int64_t level,
                                             const bool include_fill_in) {
  const int64_t num_nodes = level_blocks[level];
  // Allocate spaces for all nodes
  for (int64_t i = 0; i < num_nodes; i++) {
    const auto node_level = (matrix_type == BLR2_MATRIX && level == height) ? domain.tree_height : level;
    const auto idx = domain.get_cell_idx(i, node_level);
    const auto& cell = domain.cells[idx];
    const auto skeleton = get_skeleton(domain, i, level);
    const auto skeleton_size = skeleton.size();
    U.insert(i, level, Matrix(skeleton_size, skeleton_size));
    Uc.insert(i, level, generate_identity_matrix(skeleton_size, skeleton_size));
    R_row.insert(i, level, Matrix(skeleton_size, skeleton_size));
    multipoles.insert(i, level, std::vector<int64_t>(skeleton_size));
  }
  // Construct bases
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    const auto node_level = (matrix_type == BLR2_MATRIX && level == height) ? domain.tree_height : level;
    const auto idx = domain.get_cell_idx(i, node_level);
    const auto& cell = domain.cells[idx];
    const auto skeleton = get_skeleton(domain, i, level);
    const int64_t skeleton_size = skeleton.size();
    const int64_t near_size = include_fill_in ? cell.near_list.size() - 1 : 0;
    const int64_t far_size  = cell.sample_farfield.size();
    auto& U_i = U(i, level);
    auto& Uc_i = Uc(i, level);
    auto& R_i = R_row(i, level);
    auto& multipoles_i = multipoles(i, level);
    if (near_size + far_size > 0) {
      // Allocate skeleton_row
      std::vector<int64_t> col_splits;
      int64_t ncols = 0;
      if (far_size > 0) {
        ncols += far_size;
        col_splits.push_back(ncols);
      }
      if (near_size > 0) {
        for (int64_t j: near_neighbors(i, level)) {
          if (i != j) {
            if (!F.exists(i, j, level)) {
              throw std::logic_error("Could not find F(" + std::to_string(i) + "," + std::to_string(j) +
                                     "," + std::to_string(level) + ") when constructing composite basis");
            }
            ncols += F(i, j, level).cols;
            col_splits.push_back(ncols);
          }
        }
      }
      col_splits.pop_back();
      Matrix skeleton_row(skeleton_size, ncols);
      auto skeleton_row_splits = skeleton_row.split({}, col_splits);
      int64_t k = 0;
      // Append low-rank part
      if (far_size > 0) {
        skeleton_row_splits[k++] = generate_p2p_matrix(domain, skeleton, cell.sample_farfield);
      }
      // Append fill-in part
      if (near_size > 0) {
        for (int64_t j: near_neighbors(i, level)) {
          if (i != j) {
            skeleton_row_splits[k++] = F(i, j, level);
          }
        }
      }

      Matrix Ui;
      int64_t rank;
      std::vector<int64_t> ipiv_row;
#ifdef USE_SVD_COMPRESSION
      // SVD followed by ID
      Matrix Utemp, Stemp, Vtemp;
      std::tie(Utemp, Stemp, Vtemp, rank) = error_svd(skeleton_row, err_tol, use_rel_acc, true);
      // Truncate to max_rank if exceeded
      if (max_rank > 0 && rank > max_rank) {
        rank = max_rank;
        Utemp.shrink(Utemp.rows, rank);
        Stemp.shrink(rank, rank);
      }
      // ID to get skeleton rows
      column_scale(Utemp, Stemp);
      std::tie(Ui, ipiv_row) = truncated_id_row(Utemp, rank);
#else
      // ID Only
      std::tie(Ui, ipiv_row) = error_id_row(skeleton_row, err_tol, use_rel_acc);
      rank = Ui.cols;
      // Truncate to max_rank if exceeded
      if (max_rank > 0 && rank > max_rank) {
        rank = max_rank;
        Ui.shrink(Ui.rows, rank);
      }
#endif
      // Multiply U with child R
      if (level < height) {
        const auto& child1 = domain.cells[cell.child];
        const auto& child2 = domain.cells[cell.child + 1];
        const auto& child1_skeleton = multipoles(child1.block_index, child1.level);
        const auto& child2_skeleton = multipoles(child2.block_index, child2.level);
        auto Ui_splits = Ui.split(vec{(int64_t)child1_skeleton.size()}, vec{});
        triangular_matmul(R_row(child1.block_index, child1.level), Ui_splits[0],
                          Hatrix::Left, Hatrix::Upper, false, false, 1);
        triangular_matmul(R_row(child2.block_index, child2.level), Ui_splits[1],
                          Hatrix::Left, Hatrix::Upper, false, false, 1);
      }
      // Orthogonalize basis with QR
      Matrix Q(skeleton_size, skeleton_size);
      Matrix R(skeleton_size, rank);
      qr(Ui, Q, R);
      auto Q_splits = Q.split(vec{}, vec{rank});
      Matrix Qo(Q_splits[0], true);
      Matrix Qc = rank < skeleton_size ? Matrix(Q_splits[1], true) : Matrix(skeleton_size, 0);
      R.shrink(rank, rank);
      // Insert to pre-allocated space
      U_i.shrink(Qo.rows, Qo.cols);
      Uc_i.shrink(Qc.rows, Qc.cols);
      R_i.shrink(rank, rank);
      U_i = Qo;
      Uc_i = Qc;
      R_i = R;
      // Convert ipiv to multipoles
      multipoles_i.resize(rank);
      for (int64_t k = 0; k < rank; k++) {
        multipoles_i[k] = skeleton[ipiv_row[k]];
      }
    }
    else {
      // Insert Dummies
      const int64_t rank = 0;
      U_i.shrink(skeleton_size, rank);
      Uc_i.shrink(skeleton_size, skeleton_size);
      R_i.shrink(rank, rank);
      multipoles_i.resize(rank);
    }
  }
}

void SymmetricH2::generate_near_coupling_matrices(const Domain& domain) {
  const auto level = height;
  const auto num_nodes = level_blocks[level];
  // Inadmissible leaf blocks
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j: near_neighbors(i, level)) {
      const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
      D.insert(i, j, level, generate_p2p_matrix(domain, i, j, node_level));
    }
  }
}

void SymmetricH2::generate_far_coupling_matrices(const Domain& domain, const int64_t level) {
  const auto num_nodes = level_blocks[level];
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j: far_neighbors(i, level)) {
      const auto& multipoles_i = multipoles(i, level);
      const auto& multipoles_j = multipoles(j, level);
      Matrix Sij = generate_p2p_matrix(domain, multipoles_i, multipoles_j);
      // Multiply with R from left and right
      triangular_matmul(R_row(i, level), Sij,
                        Hatrix::Left, Hatrix::Upper, false, false, 1);
      triangular_matmul(R_row(j, level), Sij,
                        Hatrix::Right, Hatrix::Upper, true, false, 1);
      #pragma omp critical
      {
        S.insert(i, j, level, std::move(Sij));
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
                         const int64_t matrix_type, const bool build_basis)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis), matrix_type(matrix_type) {
  // Consider setting error tolerance to be smaller than desired accuracy, based on HiDR paper source code
  // https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c#L859
  err_tol = accuracy;
  initialize_geometry_admissibility(domain);
  generate_near_coupling_matrices(domain);
  if (build_basis) {
    for (int64_t level = height; level >= 0; level--) {
      generate_row_cluster_basis(domain, level, false);
      generate_far_coupling_matrices(domain, level);
    }
  }
}

int64_t SymmetricH2::get_basis_min_rank() const {
  int64_t rank_min = N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (U.exists(node, level) && U(node, level).cols > 0) {
        rank_min = std::min(rank_min, U(node, level).cols);
      }
    }
  }
  return rank_min;
}

int64_t SymmetricH2::get_basis_max_rank() const {
  int64_t rank_max = -N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (U.exists(node, level) && U(node, level).cols > 0) {
        rank_max = std::max(rank_max, U(node, level).cols);
      }
    }
  }
  return rank_max;
}

double SymmetricH2::get_basis_avg_rank() const {
  int64_t rank_sum = 0;
  int64_t num_bases = 0;
  for (int64_t level = height; level > 0; level--) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (U.exists(node, level) && U(node, level).cols > 0) {
        num_bases++;
        rank_sum += U(node, level).cols;
      }
    }
  }
  return (double)rank_sum / (double)num_bases;
}


double SymmetricH2::construction_error(const Domain& domain) const {
  double dense_norm = 0;
  double diff_norm = 0;
  // Inadmissible blocks (only at leaf level)
  for (int64_t i = 0; i < level_blocks[height]; i++) {
    for (int64_t j: near_neighbors(i, height)) {
      const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
      const Matrix D_ij = Hatrix::generate_p2p_matrix(domain, i, j, node_level);
      const Matrix A_ij = D(i, j, height);
      const auto dnorm = norm(D_ij);
      const auto diff = norm(A_ij - D_ij);
      dense_norm += dnorm * dnorm;
      diff_norm += diff * diff;
    }
  }
  // Admissible blocks
  for (int64_t level = height; level > 0; level--) {
    for (int64_t i = 0; i < level_blocks[level]; i++) {
      for (int64_t j: far_neighbors(i, level)) {
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
  return (use_rel_acc ? std::sqrt(diff_norm / dense_norm) : std::sqrt(diff_norm));
}

int64_t SymmetricH2::memory_usage() const {
  int64_t mem = 0;
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t i = 0; i < num_nodes; i++) {
      if (U.exists(i, level)) {
        mem += U(i, level).memory_used();
      }
      if (Uc.exists(i, level)) {
        mem += Uc(i, level).memory_used();
      }
      if (R_row.exists(i, level)) {
        mem += R_row(i, level).memory_used();
      }
      for (auto j: near_neighbors(i, level)) {
        if (D.exists(i, j, level)) {
          mem += D(i, j, level).memory_used();
        }
      }
      for (auto j: far_neighbors(i, level)) {
        if (S.exists(i, j, level)) {
          mem += S(i, j, level).memory_used();
        }
      }
    }
  }
  return mem;
}

void SymmetricH2::print_structure(const int64_t level) const {
  if (level == 0) { return; }
  const int64_t num_nodes = level_blocks[level];
  std::cout << "LEVEL: " << level << " NUM_NODES: " << num_nodes << std::endl;
  for (int64_t i = 0; i < num_nodes; i++) {
    if (level == height && D.exists(i, i, height)) {
      std::cout << D(i, i, height).rows << " ";
    }
    std::cout << "| ";
    for (int64_t j = 0; j < num_nodes; j++) {
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
    const int64_t num_nodes = level_blocks[level];
    printf("LEVEL:%d\n", (int)level);
    for(int64_t node = 0; node < num_nodes; node++) {
      printf("\tNode-%d: Rank=%d\n", (int)node,
             (U.exists(node, level) ? (int)U(node, level).cols : -1));
    }
  }
}

double SymmetricH2::low_rank_block_ratio() const {
  double total = 0, low_rank = 0;
  const int64_t num_nodes = level_blocks[height];
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j = 0; j < num_nodes; j++) {
      if ((is_admissible.exists(i, j, height) && is_admissible(i, j, height)) ||
          !is_admissible.exists(i, j, height)) {
        low_rank += 1;
      }
      total += 1;
    }
  }
  return low_rank / total;
}

#ifdef USE_JSON
void SymmetricH2::fill_JSON(const Domain& domain,
                            const int64_t i, const int64_t j,
                            const int64_t level,
                            nlohmann::json& json) const {
  json["abs_pos"] = {i, j};
  json["level"] = level;
  json["dim"] = {get_block_size(domain, i, level), get_block_size(domain, j, level)};
  if (is_admissible.exists(i, j, level)) {
    if (is_admissible(i, j, level)) {
      json["type"] = "LowRank";
      json["rank"] = U(i, level).cols;
      const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
      Matrix Dij = generate_p2p_matrix(domain, i, j, node_level);
      json["svalues"] = get_singular_values(Dij);
    }
    else {
      if (level == height) {
        json["type"] = "Dense";
        const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
        Matrix Dij = generate_p2p_matrix(domain, i, j, node_level);
        json["svalues"] = get_singular_values(Dij);
      }
      else {
        json["type"] = "Hierarchical";
        json["children"] = {};
        if (matrix_type == BLR2_MATRIX) {
          for (int64_t i_child = 0; i_child < level_blocks[height]; i_child++) {
            std::vector<nlohmann::json> row(level_blocks[height]);
            int64_t j_pos = 0;
            for (int64_t j_child = 0; j_child < level_blocks[height]; j_child++) {
              fill_JSON(domain, i_child, j_child, height, row[j_pos]);
              j_pos++;
            }
            json["children"].push_back(row);
          }
        }
        else {
          for (int64_t i_child = 2 * i; i_child <= (2 * i + 1); i_child++) {
            std::vector<nlohmann::json> row(2);
            int64_t j_pos = 0;
            for (int64_t j_child = 2 * j; j_child <= (2 * j + 1); j_child++) {
              fill_JSON(domain, i_child, j_child, level + 1, row[j_pos]);
              j_pos++;
            }
            json["children"].push_back(row);
          }
        }
      }
    }
  }
}

void SymmetricH2::write_JSON(const Domain& domain,
                             const std::string filename) const {
  nlohmann::json json;
  fill_JSON(domain, 0, 0, 0, json);
  std::ofstream out_file(filename);
  out_file << json << std::endl;
}
#endif

Matrix SymmetricH2::get_dense_skeleton(const Domain& domain,
                                       const int64_t i, const int64_t j, const int64_t level) const {
  const auto skeleton_i = get_skeleton(domain, i, level);
  const auto skeleton_j = get_skeleton(domain, j, level);
  return generate_p2p_matrix(domain, skeleton_i, skeleton_j);
}

Matrix SymmetricH2::get_oo(const int64_t i, const int64_t j, const int64_t level) const {
  if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
    // Admissible block, use S block
    return S(i, j, level);
  }
  else {
    // Inadmissible block, use oo part of dense block
    const Matrix& Dij = D(i, j, level);
    const Matrix& Ui = U(i, level);
    const Matrix& Uj = U(j, level);
    const auto Dij_splits = Dij.split(vec{Dij.rows - Ui.cols},
                                      vec{Dij.cols - Uj.cols});
    return Dij_splits[3];
  }
}

void SymmetricH2::compute_fill_in(const Domain& domain, const int64_t level, const double diag_shift) {
  const int64_t num_nodes = level_blocks[level];
  #pragma omp parallel for
  for (int64_t k = 0; k < num_nodes; k++) {
    Matrix Dkk = get_dense_skeleton(domain, k, k, level);
    shift_diag(Dkk, diag_shift);
    std::vector<int> ipiv;
    pivoted_ldl(Dkk, ipiv);
    for (int64_t i: near_neighbors(k, level)) {
      if (i != k) {
        // Compute fill_in = Dik * inv(Dkk)
        Matrix fill_in_T = transpose(get_dense_skeleton(domain, i, k, level)); // TODO change to D_ki
        pivoted_ldl_solve(Dkk, ipiv, fill_in_T);
        #pragma omp critical
        {
          F.insert(i, k, level, transpose(fill_in_T));
        }
      }
    }
  }
}

void SymmetricH2::factorize_level(const int64_t level) {
  if (level == 0) return;
  const int64_t parent_level = level - 1;
  const int64_t num_nodes = level_blocks[level];

  #pragma omp parallel for
  for (int64_t k = 0; k < num_nodes; k++) {
    // Skeleton (o) and Redundancy (c) decomposition
    Matrix UF_k = concat(Uc(k, level), U(k, level), 1);
    for (int64_t i: near_neighbors(k, level)) {
#ifdef IGNORE_L_FACTOR
      if (i == k) {
        Matrix UF_i = concat(Uc(i, level), U(i, level), 1);
        D(i, k, level) = matmul(UF_i, matmul(D(i, k, level), UF_k), true, false);
      }
      else {
        // Ignore redundancy parts
        Matrix Dik(D(i, k, level));
        auto Dik_splits = D(i, k, level).split(vec{Uc(i, level).cols}, vec{Uc(k, level).cols});
        Dik_splits[3] = matmul(U(i, level), matmul(Dik, U(k, level)), true, false);
      }
#else
      Matrix UF_i = concat(Uc(i, level), U(i, level), 1);
      D(i, k, level) = matmul(UF_i, matmul(D(i, k, level), UF_k), true, false);
#endif
    }
    // Factorization
    const int64_t diag_row_split = D(k, k, level).rows - U(k, level).cols;
    const int64_t diag_col_split = D(k, k, level).cols - U(k, level).cols;
    auto Dkk_splits = D(k, k, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dkk_cc = Dkk_splits[0];
    Matrix& Dkk_oc = Dkk_splits[2];
    Matrix& Dkk_oo = Dkk_splits[3];
    ldl(Dkk_cc);
    // Lower Elimination
    for (int64_t i: near_neighbors(k, level)) {
#ifdef IGNORE_L_FACTOR
      // Ignore triangular solves with other than diagonal block
      if (i != k) continue;
#endif
      auto Dik_splits = D(i, k, level).split(vec{D(i, k, level).rows - U(i, level).cols},
                                             vec{diag_col_split});
      Matrix& Dik_cc = Dik_splits[0];
      Matrix& Dik_oc = Dik_splits[2];
      solve_triangular(Dkk_cc, Dik_oc, Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(Dkk_cc, Dik_oc, Hatrix::Right);
      if (i > k) {
        solve_triangular(Dkk_cc, Dik_cc, Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dkk_cc, Dik_cc, Hatrix::Right);
      }
    }
    // Schur Complement
    Matrix Dkk_oc_copy(Dkk_oc, true);
    column_scale(Dkk_oc_copy, Dkk_cc);
    matmul(Dkk_oc_copy, Dkk_oc, Dkk_oo, false, true, -1, 1);
  }
}

void SymmetricH2::permute_and_merge(const int64_t level) {
  if (level == 0) return;
  // Merge oo parts of children as parent level near coupling matrices
  if (matrix_type == BLR2_MATRIX) {
    const int64_t num_nodes = level_blocks[level];
    int64_t nrows = 0;
    std::vector<int64_t> parent_row_splits;
    for (int64_t i = 0; i < num_nodes; i++) {
      nrows += U(i, level).cols;
      if (i < (num_nodes - 1)) {
        parent_row_splits.push_back(nrows);
      }
    }
    Matrix Dij(nrows, nrows);
    auto Dij_splits = Dij.split(parent_row_splits, parent_row_splits);
    for (int64_t ic = 0; ic < num_nodes; ic++) {
      for (int64_t jc = 0; jc < num_nodes; jc++) {
        Dij_splits[ic * num_nodes + jc] = get_oo(ic, jc, level);
      }
    }
    D.insert(0, 0, 0, std::move(Dij));
  }
  else {
    const auto parent_level = level - 1;
    const auto parent_num_nodes = level_blocks[parent_level];
    for (int64_t i = 0; i < parent_num_nodes; i++) {
      for (int64_t j: near_neighbors(i, parent_level)) {
        const auto i_c1 = i * 2 + 0;
        const auto i_c2 = i * 2 + 1;
        const auto j_c1 = j * 2 + 0;
        const auto j_c2 = j * 2 + 1;
        const auto nrows = U(i_c1, level).cols + U(i_c2, level).cols;
        const auto ncols = U(j_c1, level).cols + U(j_c2, level).cols;
        Matrix Dij(nrows, ncols);
        auto Dij_splits = Dij.split(vec{U(i_c1, level).cols},
                                    vec{U(j_c1, level).cols});
        Dij_splits[0] = get_oo(i_c1, j_c1, level);  // Dij_cc
        Dij_splits[1] = get_oo(i_c1, j_c2, level);  // Dij_co
        Dij_splits[2] = get_oo(i_c2, j_c1, level);  // Dij_oc
        Dij_splits[3] = get_oo(i_c2, j_c2, level);  // Dij_oo
        D.insert(i, j, parent_level, std::move(Dij));
      }
    }
  }
}

void SymmetricH2::factorize(const Domain& domain, const double diag_shift) {
  for (int64_t level = height; level >= 0; level--) {
    compute_fill_in(domain, level, diag_shift);
    generate_row_cluster_basis(domain, level, true);
    generate_far_coupling_matrices(domain, level);
    factorize_level(level);
    permute_and_merge(level);
  }
  // Factorize remaining root level
  ldl(D(0, 0, 0));
}

int64_t inertia(const SymmetricH2& A, const Domain& domain,
                const double lambda, bool &singular) {
  SymmetricH2 A_shifted(A);
  // Shift leaf level diagonal blocks
  const int64_t leaf_num_nodes = A.level_blocks[A.height];
  for(int64_t node = 0; node < leaf_num_nodes; node++) {
    shift_diag(A_shifted.D(node, node, A.height), -lambda);
  }
  // LDL Factorize
  A_shifted.factorize(domain, -lambda);
  // Count negative entries in D
  int64_t negative_elements_count = 0;
  for(int64_t level = A.height; level >= 0; level--) {
    int64_t num_nodes = A.level_blocks[level];
    for(int64_t node = 0; node < num_nodes; node++) {
      if(level == 0) {
        const Matrix& D_lambda = A_shifted.D(node, node, level);
        for(int64_t i = 0; i < D_lambda.min_dim(); i++) {
          negative_elements_count += (D_lambda(i, i) < 0 ? 1 : 0);
          if(std::isnan(D_lambda(i, i)) || std::abs(D_lambda(i, i)) < EPS) singular = true;
        }
      }
      else {
        const Matrix& D_node = A_shifted.D(node, node, level);
        const auto rank = A_shifted.U(node, level).cols;
        const auto D_node_splits = D_node.split(vec{D_node.rows - rank},
                                                vec{D_node.cols - rank});
        const Matrix& D_lambda = D_node_splits[0];
        for(int64_t i = 0; i < D_lambda.min_dim(); i++) {
          negative_elements_count += (D_lambda(i, i) < 0 ? 1 : 0);
          if(std::isnan(D_lambda(i, i)) || std::abs(D_lambda(i, i)) < EPS) singular = true;
        }
      }
    }
  }
  return negative_elements_count;
}

double get_kth_eigenvalue(const SymmetricH2& A, const Domain& domain,
                          const double ev_tol, const int64_t idx_k,
                          const std::vector<int64_t>& k_list,
                          std::vector<double>& a,
                          std::vector<double>& b) {
  bool singular = false;
  while((b[idx_k] - a[idx_k]) >= ev_tol) {
    const auto mid = (a[idx_k] + b[idx_k]) / 2;
    const auto v_mid = inertia(A, domain, mid, singular);
    if(singular) {
      printf("Shifted matrix becomes singular (shift=%.5lf)\n", mid);
      break;
    }
    // Update intervals accordingly
    for (int64_t idx = 0; idx < k_list.size(); idx++) {
      const auto ki = k_list[idx];
      if (ki <= v_mid && mid < b[idx]) {
        b[idx] = mid;
      }
      if (ki > v_mid && mid > a[idx]) {
        a[idx] = mid;
      }
    }
  }
  return (a[idx_k] + b[idx_k]) / 2.;
}

} // namespace Hatrix

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Parse Inputs
  int64_t N = argc > 1 ? atol(argv[1]) : 256;
  int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-8;
  // Use relative or absolute error threshold for LRA
  const bool use_rel_acc = argc > 4 ? (atol(argv[4]) == 1) : false;
  const int64_t max_rank = argc > 5 ? atol(argv[5]) : 30;
  const double admis = argc > 6 ? atof(argv[6]) : 3;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 7 ? atol(argv[7]) : 1;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: ELSES Dense Matrix
  const int64_t kernel_type = argc > 8 ? atol(argv[8]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  const int64_t geom_type = argc > 9 ? atol(argv[9]) : 0;
  int64_t ndim  = argc > 10 ? atol(argv[10]) : 1;

  // Specify sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net method
  const int64_t sampling_algo = argc > 11 ? atol(argv[11]) : 3;
  // Specify maximum number of sample points that represents a cluster
  // For anchor-net method: size of grid
  int64_t sample_self_size = argc > 12 ? atol(argv[12]) :
                             (ndim == 1 ? 10 * leaf_size : 30);
  // Specify maximum number of sample points that represents a cluster's far-field
  // For anchor-net method: size of grid
  int64_t sample_far_size = argc > 13 ? atol(argv[13]) :
                            (ndim == 1 ? 10 * leaf_size + 10: sample_self_size + 5);
  // Eigenvalue computation parameters
  const double ev_tol = argc > 14 ? atof(argv[14]) : 1.e-3;
  int64_t k_begin = argc > 15 ? atol(argv[15]) : 1;
  int64_t k_end = argc > 16 ? atol(argv[16]) : k_begin;
  double a = argc > 17 ? atof(argv[17]) : 0;
  double b = argc > 18 ? atof(argv[18]) : 0;
  const bool compute_eig_acc = argc > 19 ? (atol(argv[19]) == 1) : true;
  const int64_t print_csv_header = argc > 20 ? atol(argv[20]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 21 ? std::string(argv[21]) : "";
  const int64_t sort_bodies = argc > 22 ? atol(argv[22]) : 0;

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1e-3, 1.);
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
      kernel_name = "ELSES-kernel";
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
  }

  // Construct H2-Matrix
  // At the moment all processes redundantly construct the same instance of H2-Matrix
  const bool is_non_synthetic = (geom_type == 3);
  if (is_non_synthetic) {
    const int64_t num_atoms_per_molecule = 60;
    const int64_t num_electrons_per_atom = kernel_type == 2 ? 4 : 1;
    const int64_t molecule_size = num_atoms_per_molecule * num_electrons_per_atom;
    assert(file_name.length() > 0);
    domain.read_bodies_ELSES(file_name + ".xyz", num_electrons_per_atom);
    assert(N == domain.N);

    if (sort_bodies) {
      domain.sort_bodies_ELSES(molecule_size);
      geom_name = geom_name + "_sorted";
    }
    domain.build_tree_from_sorted_bodies(leaf_size, std::vector<int64_t>(N / leaf_size, leaf_size));
    if (kernel_type == 2) {
      domain.read_p2p_matrix_ELSES(file_name + ".dat");
    }
  }
  else {
    domain.build_tree(leaf_size);
  }
  domain.build_interactions(admis);

  const auto start_sample = MPI_Wtime();
  domain.build_sample_bodies(sample_self_size, 0, sample_far_size, sampling_algo, geom_type == 3);
  const auto stop_sample = MPI_Wtime();
  const double sample_time = stop_sample - start_sample;

  const auto start_construct = MPI_Wtime();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type, true);
  const auto stop_construct = MPI_Wtime();
  const auto construct_min_rank = A.get_basis_min_rank();
  const auto construct_max_rank = A.get_basis_max_rank();
  const auto construct_avg_rank = A.get_basis_avg_rank();
  const auto construct_time = stop_construct - start_construct;
  const auto construct_error = A.construction_error(domain);
  const auto construct_mem = A.memory_usage();
  // Construct without basis
  Hatrix::SymmetricH2 M(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type, false);

  // All processes finished construction
  MPI_Barrier(MPI_COMM_WORLD);

#ifndef OUTPUT_CSV
  if (mpi_rank == 0) {
      printf("mpi_nprocs=%d nthreads=%d N=%d leaf_size=%d accuracy=%.1e acc_type=%d max_rank=%d"
             " admis=%.1lf matrix_type=%d kernel=%s geometry=%s"
             " sampling_algo=%s sample_self_size=%d sample_far_size=%d sample_farfield_max_size=%d"
             " height=%d construct_min_rank=%d construct_max_rank=%d construct_avg_rank=%.3lf"
             " construct_mem=%d construct_time=%.3lf construct_error=%.5e\n",
             mpi_nprocs, omp_get_max_threads(), (int)N, (int)leaf_size, accuracy,
             (int)use_rel_acc, (int)max_rank, admis, (int)matrix_type,
             kernel_name.c_str(), geom_name.c_str(), sampling_algo_name.c_str(),
             (int)sample_self_size, (int)sample_far_size,
             (int)domain.get_max_farfield_size(), (int)A.height, (int)construct_min_rank,
             (int)construct_max_rank, construct_avg_rank,
             (int)construct_mem, construct_time, construct_error);
  }
#endif

  // Compute dense eigenvalue with LAPACK dsyev (if necessary)
  std::vector<double> dense_ev;
  double dense_eig_time = 0;
  if (mpi_rank == 0) {
    if (compute_eig_acc) {
      Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
      const auto dense_eig_start = MPI_Wtime();
      dense_ev = Hatrix::get_eigenvalues(Adense);
      const auto dense_eig_stop = MPI_Wtime();
      dense_eig_time = dense_eig_stop - dense_eig_start;
    }
#ifndef OUTPUT_CSV
    printf("dense_eig_time_all=%.3lf\n", dense_eig_time);
#endif
  }

  // Check whether starting interval contains all desired eigenvalues
  int v_a, v_b;
  if (mpi_rank == 0) {
    bool s;
    v_a = inertia(M, domain, a, s);
    v_b = inertia(M, domain, b, s);
    if (!(k_begin > v_a && k_end <= v_b)) {
      printf("Error: starting interval does not contain all target eigenvalues:"
             " [%.3lf, %.3lf] contain [%d, %d] eigenvalues, while the target is [%d, %d]\n",
             a, b, v_a + 1, v_b, (int)k_begin, (int)k_end);
      // Abort by making initial task of size 0
      k_begin = 0;
      k_end = -1;
    }
  }

#ifdef OUTPUT_CSV
  if (mpi_rank == 0 && print_csv_header == 1) {
    // Print CSV header
    printf("mpi_nprocs,nthreads,N,leaf_size,accuracy,acc_type,max_rank,admis,matrix_type,kernel,geometry"
           ",sampling_algo,sample_self_size,sample_far_size,sample_farfield_max_size"
           ",height,construct_min_rank,construct_max_rank,construct_avg_rank"
           ",construct_mem,construct_time,construct_error"
           ",dense_eig_time_all,h2_eig_time_all"
           ",m,k,a,b,v_a,v_b,ev_tol,dense_ev,h2_ev,eig_abs_err,success\n");
  }
#endif

  // Initialize variables
  double h2_ev_time = 0;
  std::vector<double> h2_ev;
  // Begin computing target eigenvalues
  MPI_Barrier(MPI_COMM_WORLD);
  h2_ev_time -= MPI_Wtime();
  // Get task for current process
  const int64_t num_ev = k_end - k_begin + 1;
  const int64_t num_working_procs = mpi_nprocs > num_ev ? num_ev : mpi_nprocs;
#ifdef DEBUG_OUTPUT
  if (mpi_rank == 0) {
    printf("\nProcess-%d: num_ev=%d, num_working_procs=%d\n",
           mpi_rank, (int)num_ev, (int)num_working_procs);
  }
#endif
  // Create communicator for working processes
  MPI_Comm comm;
  int color = (mpi_rank < num_working_procs);
  int key = mpi_rank;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm);

  std::vector<int> offset(num_working_procs + 1, 0), count(num_working_procs, 0);
  for (int i = 0; i < (int)offset.size(); i++)
    offset[i] = (i * num_ev) / num_working_procs;
  for (int i = 0; i < (int)count.size(); i++)
    count[i] = offset[i + 1] - offset[i];

  int64_t local_num_ev;
  // Compute eigenvalues
  if (mpi_rank < num_working_procs) {
    local_num_ev = count[mpi_rank];
    const int64_t k0 = k_begin + offset[mpi_rank];
    const int64_t k1 = k0 + local_num_ev - 1;
#ifdef DEBUG_OUTPUT
    printf("Process-%d: local_num_ev=%d, k0=%d, k1=%d\n",
           mpi_rank, (int)local_num_ev, (int)k0, (int)k1);
#endif
    std::vector<int64_t> local_k(local_num_ev);  // Local target eigenvalue indices
    std::vector<double> local_a(local_num_ev, a), local_b(local_num_ev, b);  // Local starting intervals
    std::vector<double> local_ev(local_num_ev);
    for (int64_t idx_k = 0; idx_k < local_num_ev; idx_k++) {
      local_k[idx_k] = k0 + idx_k;
    }
    int64_t idx_k_start = 0;
    int64_t idx_k_finish = local_num_ev-1;
    // Compute largest and smallest eigenvalues in the set
#ifdef DEBUG_OUTPUT
    printf("Process-%d: Computing %d-th eigenvalue\n",
           mpi_rank, (int)local_k[idx_k_start]);
#endif
    local_ev[idx_k_start] = get_kth_eigenvalue(M, domain, ev_tol, idx_k_start,
                                               local_k, local_a, local_b);
    if (idx_k_start < idx_k_finish) {
#ifdef DEBUG_OUTPUT
      printf("Process-%d: Computing %d-th eigenvalue\n",
             mpi_rank, (int)local_k[idx_k_finish]);
#endif
      local_ev[idx_k_finish] = get_kth_eigenvalue(M, domain, ev_tol, idx_k_finish,
                                                  local_k, local_a, local_b);
    }
    idx_k_start++;
    idx_k_finish--;
#ifdef DEBUG_OUTPUT
    printf("Process-%d: Computing %d inner eigenvalues between [%d, %d]-th eigenvalue\n",
           mpi_rank, (int)std::max((int64_t)0, idx_k_finish - idx_k_start + 1),
           (int)local_k[idx_k_start-1], (int)local_k[idx_k_finish+1]);
#endif
    // Compute the rest of eigenvalues
    for (int64_t idx_k = idx_k_start; idx_k <= idx_k_finish; idx_k++) {
      local_ev[idx_k] = get_kth_eigenvalue(M, domain, ev_tol, idx_k,
                                           local_k, local_a, local_b);
    }
    // Gather results at process 0
    if (mpi_rank == 0) {
      h2_ev.assign(num_ev, 0);
      MPI_Gatherv(local_ev.data(), local_num_ev, MPI_DOUBLE,
                  h2_ev.data(), count.data(), offset.data(), MPI_DOUBLE, 0, comm);
    }
    else {
      MPI_Gatherv(local_ev.data(), local_num_ev, MPI_DOUBLE,
                  nullptr, nullptr, nullptr, MPI_DOUBLE, 0, comm);
    }
  }
  else {
#ifdef DEBUG_OUTPUT
    printf("Process-%d: Not working\n", mpi_rank);
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  h2_ev_time += MPI_Wtime();

  if (mpi_rank == 0) {
    const int m = local_num_ev;
    for (int k = k_begin; k <= k_end; k++) {
      const double dense_ev_k = compute_eig_acc ? dense_ev[k-1] : -1;
      const double h2_ev_k = h2_ev[k - k_begin];
      const double eig_abs_err = compute_eig_acc ? std::abs(dense_ev_k - h2_ev_k) : -1;
      const std::string success = eig_abs_err < (0.5 * ev_tol) ? "TRUE" : "FALSE";
#ifndef OUTPUT_CSV
      printf("h2_eig_time_all=%.3lf m=%d k=%d a=%.2lf b=%.2lf v_a=%d v_b=%d ev_tol=%.1e"
             " dense_ev=%.8lf h2_ev=%.8lf eig_abs_err=%.2e success=%s\n",
             h2_ev_time, m, k, a, b, v_a, v_b, ev_tol,
             dense_ev_k, h2_ev_k, eig_abs_err, success.c_str());
#else
      printf("%d,%d,%d,%d,%.1e,%d,%d,%.1lf,%d,%s,%s,%s,%d,%d,%d,%d,%d,%d,%.3lf,%d,%.3lf,%.5e"
             ",%.3lf,%.3lf,%d,%d,%.2lf,%.2lf,%d,%d,%.1e,%.8lf,%.8lf,%.2e,%s\n",
             mpi_nprocs, omp_get_max_threads(), (int)N, (int)leaf_size,
             accuracy, (int)use_rel_acc, (int)max_rank,
             admis, (int)matrix_type, kernel_name.c_str(), geom_name.c_str(),
             sampling_algo_name.c_str(), (int)sample_self_size, (int)sample_far_size,
             (int)domain.get_max_farfield_size(), (int)A.height,
             (int)construct_min_rank, (int)construct_max_rank, construct_avg_rank,
             (int)construct_mem, construct_time, construct_error, dense_eig_time,
             h2_ev_time, m, k, a, b, v_a, v_b, ev_tol,
             dense_ev_k, h2_ev_k, eig_abs_err, success.c_str());
#endif
    }
  }

  Hatrix::Context::finalize();
  MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}

