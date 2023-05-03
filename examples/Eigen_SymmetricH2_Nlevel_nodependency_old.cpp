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
#include <cstdio>

#ifdef USE_JSON
#include "nlohmann/json.hpp"
#endif

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

#ifdef USE_TIMER
#define START_TIMER(name) Hatrix::timing::start(name)
#define STOP_TIMER(name)  Hatrix::timing::stop(name)
#define PRINT_TIME(name, depth) Hatrix::timing::printTime(name, depth)
#else
#define START_TIMER(name)
#define STOP_TIMER(name)
#define PRINT_TIME(name, depth)
#endif

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
#ifdef USE_JSON
  void fill_JSON(const Domain& domain,
                 const int64_t i, const int64_t j,
                 const int64_t level, nlohmann::json& json) const;
#endif

  void factorize_level(const int64_t level);
  void permute_and_merge(const int64_t level);

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const bool use_rel_acc,
              const int64_t max_rank, const double admis,
              const int64_t matrix_type, const bool include_fill_in);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double get_basis_avg_rank() const;
  double construction_error(const Domain& domain) const;
  int64_t memory_usage() const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;
#ifdef USE_JSON
  void write_JSON(const Domain& domain, const std::string filename) const;
#endif

  void factorize(const Domain& domain);
  std::tuple<int64_t, int64_t> inertia(const Domain& domain,
                                       const double lambda, bool &singular) const;
  std::tuple<double, double, int64_t>
  get_kth_eigenvalue(const Domain& domain,
                     const double ev_tol, const int64_t idx_k,
                     const std::vector<int64_t>& k_list,
                     std::vector<double>& a, std::vector<double>& b) const;
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
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    const auto node_level = (matrix_type == BLR2_MATRIX && level == height) ? domain.tree_height : level;
    const auto idx = domain.get_cell_idx(i, node_level);
    const auto& cell = domain.cells[idx];
    const auto skeleton = get_skeleton(domain, i, level);
    const int64_t skeleton_size = skeleton.size();
    const int64_t near_size = include_fill_in ? cell.sample_nearfield.size() : 0;
    const int64_t far_size  = cell.sample_farfield.size();
    if (near_size + far_size > 0) {
      Matrix skeleton_dn(skeleton_size, 0);
      Matrix skeleton_lr(skeleton_size, 0);
      double norm_dn = 0.;
      double norm_lr = 0.;
      // Fill-in (dense) part
      if (near_size > 0) {
        // Use sample of nearfield blocks within the same level
        Matrix nearblocks = generate_p2p_matrix(domain, skeleton, cell.sample_nearfield);
        // skeleton_dn = concat(skeleton_dn, matmul(nearblocks, nearblocks, false, true), 1);
        skeleton_dn = concat(skeleton_dn, nearblocks, 1);  // Concat nearblocks instead of its gram matrix
        // norm_dn = Hatrix::norm(skeleton_dn);
      }
      // Low-rank part
      if (far_size > 0) {
        skeleton_lr = concat(skeleton_lr, generate_p2p_matrix(domain, skeleton, cell.sample_farfield), 1);
        // norm_lr = Hatrix::norm(skeleton_lr);
      }
      // const double scale = (norm_dn == 0. || norm_lr == 0.) ? 1. : norm_lr / norm_dn;
      // Hatrix::scale(skeleton_dn, scale);
      Matrix skeleton_row = concat(skeleton_dn, skeleton_lr, 1);

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
      // Convert ipiv to multipoles
      std::vector<int64_t> node_multipoles;
      node_multipoles.reserve(rank);
      for (int64_t k = 0; k < rank; k++) {
        node_multipoles.push_back(skeleton[ipiv_row[k]]);
      }
      #pragma omp critical
      {
        // Insert
        U.insert(i, level, std::move(Qo));
        Uc.insert(i, level, std::move(Qc));
        R_row.insert(i, level, std::move(R));
        multipoles.insert(i, level, std::move(node_multipoles));
      }
    }
    else {
      #pragma omp critical
      {
        // Insert Dummies
        const int64_t rank = 0;
        U.insert(i, level, Matrix(skeleton_size, rank));
        Uc.insert(i, level, generate_identity_matrix(skeleton_size, skeleton_size));
        R_row.insert(i, level, Matrix(rank, rank));
        multipoles.insert(i, level, std::vector<int64_t>());
      }
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
                         const int64_t matrix_type, const bool include_fill_in)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis), matrix_type(matrix_type) {
  // Consider setting error tolerance to be smaller than desired accuracy, based on HiDR paper source code
  // https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c#L859
  err_tol = accuracy;
#ifndef USE_SVD_COMPRESSION
  // err_tol *= 1e-2;  // Use smaller threshold to match SVD accuracy
#endif
  initialize_geometry_admissibility(domain);
  generate_near_coupling_matrices(domain);
  for (int64_t level = height; level >= 0; level--) {
    generate_row_cluster_basis(domain, level, include_fill_in);
    generate_far_coupling_matrices(domain, level);
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

void SymmetricH2::factorize_level(const int64_t level) {
  if (level == 0) return;
  const int64_t parent_level = level - 1;
  const int64_t num_nodes = level_blocks[level];

  #pragma omp parallel for
  for (int64_t k = 0; k < num_nodes; k++) {
    // Skeleton (o) and Redundancy (c) decomposition
    Matrix UF_k = concat(Uc(k, level), U(k, level), 1);
    for (int64_t i: near_neighbors(k, level)) {
      Matrix UF_i = concat(Uc(i, level), U(i, level), 1);
      D(i, k, level) = matmul(UF_i, matmul(D(i, k, level), UF_k), true, false);
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
  auto Dchild_oo = [this](const int64_t ic, const int64_t jc, const int64_t child_level) {
    if (is_admissible.exists(ic, jc, child_level) && is_admissible(ic, jc, child_level)) {
      // Admissible block, use S block
      return S(ic, jc, child_level);
    }
    else {
      // Inadmissible block, use oo part of dense block
      Matrix& Dchild = D(ic, jc, child_level);
      Matrix& Ui = U(ic, child_level);
      Matrix& Uj = U(jc, child_level);
      auto Dchild_splits = Dchild.split(vec{Dchild.rows - Ui.cols},
                                        vec{Dchild.cols - Uj.cols});
      return Dchild_splits[3];
    }
  };
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
        Dij_splits[ic * num_nodes + jc] = Dchild_oo(ic, jc, level);
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
        Dij_splits[0] = Dchild_oo(i_c1, j_c1, level);  // Dij_cc
        Dij_splits[1] = Dchild_oo(i_c1, j_c2, level);  // Dij_co
        Dij_splits[2] = Dchild_oo(i_c2, j_c1, level);  // Dij_oc
        Dij_splits[3] = Dchild_oo(i_c2, j_c2, level);  // Dij_oo
        D.insert(i, j, parent_level, std::move(Dij));
      }
    }
  }
}

void SymmetricH2::factorize(const Domain& domain) {
  for (int64_t level = height; level >= 0; level--) {
    factorize_level(level);
    permute_and_merge(level);
  }
  // Factorize remaining root level
  ldl(D(0, 0, 0));
}

std::tuple<int64_t, int64_t>
SymmetricH2::inertia(const Domain& domain,
                     const double lambda, bool &singular) const {
  SymmetricH2 A_shifted(*this);
  // Shift leaf level diagonal blocks
  int64_t leaf_num_nodes = level_blocks[height];
  for(int64_t node = 0; node < leaf_num_nodes; node++) {
    shift_diag(A_shifted.D(node, node, height), -lambda);
  }
  // LDL Factorize
  A_shifted.factorize(domain);
  // Count negative entries in D
  int64_t negative_elements_count = 0;
  for(int64_t level = height; level >= 0; level--) {
    int64_t num_nodes = level_blocks[level];
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
  const auto ldl_max_rank = A_shifted.get_basis_max_rank();
  return {negative_elements_count, ldl_max_rank};
}

std::tuple<double, double, int64_t>
SymmetricH2::get_kth_eigenvalue(const Domain& domain,
                                const double ev_tol, const int64_t idx_k,
                                const std::vector<int64_t>& k_list,
                                std::vector<double>& a, std::vector<double>& b) const {
  Hatrix::profiling::PAPI papi;
  papi.add_fp_ops(0);
  papi.start();
  int64_t ldl_max_rank = 0;
  double max_rank_shift = 0;
  bool singular = false;
  while((b[idx_k] - a[idx_k]) >= ev_tol) {
    const auto mid = (a[idx_k] + b[idx_k]) / 2;
    int64_t v_mid, factor_max_rank;
    std::tie(v_mid, factor_max_rank) = (*this).inertia(domain, mid, singular);
    if(factor_max_rank >= ldl_max_rank) {
      ldl_max_rank = factor_max_rank;
      max_rank_shift = mid;
    }
    if(singular) {
      std::cout << "Shifted matrix became singular (shift=" << mid << ")" << std::endl;
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
  const auto fp_ops = (int64_t)papi.fp_ops();
  return {(a[idx_k] + b[idx_k]) / 2, max_rank_shift, fp_ops};
}

} // namespace Hatrix

int main(int argc, char ** argv) {
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
  int64_t m_begin = argc > 15 ? atol(argv[15]) : 1;
  int64_t m_end = argc > 16 ? atol(argv[16]) : m_begin;
  double a = argc > 17 ? atof(argv[17]) : 0;
  double b = argc > 18 ? atof(argv[18]) : 0;
  const bool compute_eig_acc = argc > 19 ? (atol(argv[19]) == 1) : true;
  const int64_t print_csv_header = argc > 20 ? atol(argv[20]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 21 ? std::string(argv[21]) : "";
  const int64_t sort_bodies = argc > 22 ? atol(argv[22]) : 0;

  Hatrix::Context::init();

#ifdef OUTPUT_CSV
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,leaf_size,accuracy,acc_type,max_rank,LRA,admis,matrix_type,kernel,geometry"
              << ",sampling_algo,sample_self_size,sample_far_size,sample_farfield_max_size,sample_time"
              << ",height,lr_ratio,construct_min_rank,construct_max_rank,construct_avg_rank,construct_mem,construct_time,construct_error"
              << ",dense_eig_time,build_basis_time"
              << ",m,a0,b0,v_a0,v_b0,ev_tol,h2_eig_ops,h2_eig_time,ldl_min_rank,ldl_max_rank,ldl_avg_rank,h2_eig_mem,max_rank_shift,dense_eigv,h2_eigv,eig_abs_err,success"
              << std::endl;
  }
#endif

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
      geom_name = file_name;
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

  // Pre-processing step for ELSES geometry
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

  const auto start_sample = std::chrono::system_clock::now();
  domain.build_sample_bodies(sample_self_size, sample_far_size, sample_far_size, sampling_algo, geom_type == 3);
  const auto stop_sample = std::chrono::system_clock::now();
  const double sample_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_sample - start_sample).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type, false);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  const auto construct_min_rank = A.get_basis_min_rank();
  const auto construct_max_rank = A.get_basis_max_rank();
  const auto construct_avg_rank = A.get_basis_avg_rank();
  const auto construct_error = A.construction_error(domain);
  const auto construct_mem = A.memory_usage();
  const auto lr_ratio = A.low_rank_block_ratio();

#ifndef OUTPUT_CSV
  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " acc_type=" << (use_rel_acc ? "rel_err" : "abs_err")
            << " max_rank=" << max_rank
            << " LRA="
#ifdef USE_SVD_COMPRESSION
            << "SVD+ID+QR"
#else
            << "ID+QR"
#endif
            << " admis=" << admis << std::setw(3)
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " sampling_algo=" << sampling_algo_name
            << " sample_self_size=" << sample_self_size
            << " sample_far_size=" << sample_far_size
            << " sample_farfield_max_size=" << domain.get_max_farfield_size()
            << " sample_time=" << sample_time
            << " height=" << A.height
            << " lr_ratio=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << construct_min_rank
            << " construct_max_rank=" << construct_max_rank
            << " construct_avg_rank=" << construct_avg_rank
            << " construct_mem=" << construct_mem
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error << std::defaultfloat
            << std::endl;
#endif

  std::vector<double> dense_eigv;
  double dense_eig_time = 0;
  if (compute_eig_acc) {
    Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
    const auto dense_eig_start = std::chrono::system_clock::now();
    dense_eigv = Hatrix::get_eigenvalues(Adense);
    const auto dense_eig_stop = std::chrono::system_clock::now();
    dense_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                     (dense_eig_stop - dense_eig_start).count();
  }
  const auto build_basis_start = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 M(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type, true);
  const auto build_basis_stop = std::chrono::system_clock::now();
  const double build_basis_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                  (build_basis_stop - build_basis_start).count();
#ifdef DEBUG_OUTPUT
  std::cout << std::endl;
  M.print_ranks();
  std::cout << std::endl;
#endif
#ifndef OUTPUT_CSV
  std::cout << "dense_eig_time=" << dense_eig_time
            << " build_basis_time=" << build_basis_time
            << std::endl;
#endif

  bool s = false;
  if (a == 0 && b == 0) {
    b = N < 10000 || is_non_synthetic ?
        Hatrix::norm(Hatrix::generate_p2p_matrix(domain)) : N * (1. / Hatrix::PV);
    a = -b;
  }
  int64_t v_a, v_b, temp;
  std::tie(v_a, temp) = M.inertia(domain, a, s);
  std::tie(v_b, temp) = M.inertia(domain, b, s);
  if(v_a != 0 || v_b != N) {
#ifndef OUTPUT_CSV
    std::cerr << std::endl
              << "Warning: starting interval does not contain the whole spectrum "
              << "(v(a)=v(" << a << ")=" << v_a << ","
              << " v(b)=v(" << b << ")=" << v_b << ")"
              << std::endl;
#endif
  }
  // Determine which eigenvalue(s) to approximate
  std::vector<int64_t> target_m;
  if (m_begin <= 0) {
    const auto num = m_end;
    if (m_begin == 0) {
      std::mt19937 g(N);
      std::vector<int64_t> random_m(N, 0);
      for (int64_t i = 0; i < N; i++) {
        random_m[i] = i + 1;
      }
      std::shuffle(random_m.begin(), random_m.end(), g);
      for (int64_t i = 0; i < num; i++) {
        target_m.push_back(random_m[i]);
      }
    }
    if (m_begin == -1) {
      const auto linspace = Hatrix::equally_spaced_vector(num, 1, N, true);
      for (int64_t i = 0; i < num; i++) {
        target_m.push_back((int64_t)linspace[i]);
      }
    }
  }
  else {
    for (int64_t m = m_begin; m <= m_end; m++) {
      target_m.push_back(m);
    }
  }

  const int64_t num_ev = target_m.size();
  std::vector<double> target_a(num_ev, a), target_b(num_ev, b); // Target starting intervals
  for (int64_t k = 0; k < target_m.size(); k++) {
    const int64_t m = target_m[k];
    double h2_mth_eigv, max_rank_shift;
    int64_t h2_eig_ops;
    const auto h2_eig_start = std::chrono::system_clock::now();
    std::tie(h2_mth_eigv, max_rank_shift, h2_eig_ops) =
        M.get_kth_eigenvalue(domain, ev_tol, k, target_m, target_a, target_b);
    const auto h2_eig_stop = std::chrono::system_clock::now();
    // Since we are using the composite basis for shift=0 for all shifted matrix factorization
    const auto ldl_min_rank = construct_min_rank;
    const auto ldl_max_rank = construct_max_rank;
    const auto ldl_avg_rank = construct_avg_rank;
    const auto h2_eig_mem = 2 * construct_mem;
    const double h2_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                               (h2_eig_stop - h2_eig_start).count();
    const double dense_mth_eigv = compute_eig_acc ? dense_eigv[m - 1] : -1;
    const double eig_abs_err = compute_eig_acc ? std::abs(h2_mth_eigv - dense_mth_eigv) : -1;
    const bool success = compute_eig_acc ? (eig_abs_err < (0.5 * ev_tol)) : true;
#ifndef OUTPUT_CSV
    std::cout << "m=" << m
              << " a0=" << a
              << " b0=" << b
              << " v_a0=" << v_a
              << " v_b0=" << v_b
              << " ev_tol=" << ev_tol
              << " h2_eig_ops=" << h2_eig_ops
              << " h2_eig_time=" << h2_eig_time
              << " ldl_min_rank=" << ldl_min_rank
              << " ldl_max_rank=" << ldl_max_rank
              << " ldl_avg_rank=" << ldl_avg_rank
              << " h2_eig_mem=" << h2_eig_mem
              << " max_rank_shift=" << max_rank_shift
              << " dense_eigv=" << dense_mth_eigv
              << " h2_eigv=" << h2_mth_eigv
              << " eig_abs_err=" << std::scientific << eig_abs_err << std::defaultfloat
              << " success=" << (success ? "TRUE" : "FALSE")
              << std::endl;
#else
    std::cout << N
              << "," << leaf_size
              << "," << accuracy
              << "," << (use_rel_acc ? "rel_err" : "abs_err")
              << "," << max_rank
              << ","
#ifdef USE_SVD_COMPRESSION
              << "SVD+ID+QR"
#else
              << "ID+QR"
#endif
              << "," << admis
              << "," << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
              << "," << kernel_name
              << "," << geom_name
              << "," << sampling_algo_name
              << "," << sample_self_size
              << "," << sample_far_size
              << "," << domain.get_max_farfield_size()
              << "," << sample_time
              << "," << A.height
              << "," << lr_ratio
              << "," << construct_min_rank
              << "," << construct_max_rank
              << "," << construct_avg_rank
              << "," << construct_mem
              << "," << construct_time
              << "," << std::scientific << construct_error << std::defaultfloat
              << "," << dense_eig_time
              << "," << build_basis_time
              << "," << m
              << "," << a
              << "," << b
              << "," << v_a
              << "," << v_b
              << "," << ev_tol
              << "," << h2_eig_ops
              << "," << h2_eig_time
              << "," << ldl_min_rank
              << "," << ldl_max_rank
              << "," << ldl_avg_rank
              << "," << h2_eig_mem
              << "," << max_rank_shift
              << "," << dense_mth_eigv
              << "," << h2_mth_eigv
              << "," << std::scientific << eig_abs_err << std::defaultfloat
              << "," << (success ? "TRUE" : "FALSE")
              << std::endl;
#endif
  }

  Hatrix::Context::finalize();
  return 0;
}
