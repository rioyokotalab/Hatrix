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
#include <mpi.h>

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

constexpr double EPS = std::numeric_limits<double>::epsilon();
using vec = std::vector<int64_t>;

// Uncomment the following line to print output in CSV format
#define OUTPUT_CSV
// Uncomment the following line to enable debug output
// #define DEBUG_OUTPUT
// Uncomment the following line to enable timer
// #define USE_TIMER
// Uncomment the following line to output memory consumption
// #define OUTPUT_MEM
// Uncomment the following line to pivoted QR instead of SVD for low-rank compression
// #define USE_QR_COMPRESSION

/*
 * H2-Construction employ multiplication with pure SVD without sampling
 * Quite accurate but incur O(N^2) complexity to construct basis and coupling matrices
 *
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
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
  int64_t min_adm_level;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowColMap<std::vector<int64_t>> near_neighbors, far_neighbors;  // This is actually RowLevelMap
  RowLevelMap US_row;
  std::vector<int64_t> level_blocks;
  RowColLevelMap<Matrix> F;  // Fill-in blocks
  RowColMap<std::vector<int64_t>> fill_in_neighbors;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  std::tuple<Matrix, Matrix, int64_t> svd_like_compression(Matrix& A, const bool compute_S = true) const;

  std::tuple<Matrix, Matrix>
  generate_row_cluster_basis(const Domain& domain,
                             const int64_t node, const int64_t level) const;
  void generate_leaf_nodes(const Domain& domain);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(const Domain& domain,
                             const int64_t node, const int64_t level,
                             const Matrix& Ubig_child1, const Matrix& Ubig_child2) const;
  RowLevelMap
  generate_transfer_matrices(const Domain& domain, const int64_t level,
                             RowLevelMap& Uchild);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;

  void update_row_cluster_bases(const int64_t row, const int64_t level,
                                RowMap<Matrix>& r);
  void factorize_level(const Domain& domain, const int64_t level,
                       RowMap<Matrix>& r);

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const bool use_rel_acc,
              const int64_t max_rank, const double admis,
              const int64_t matrix_type);

  int64_t get_basis_min_rank(const int64_t level_begin, const int64_t level_end) const;
  int64_t get_basis_max_rank(const int64_t level_begin, const int64_t level_end) const;
  int64_t get_level_max_nblocks(const char nearfar,
                                const int64_t level_begin, const int64_t level_end) const;
  double construction_error(const Domain& domain) const;
  int64_t memory_usage() const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;

  void factorize(const Domain& domain);
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

int64_t SymmetricH2::get_block_size(const Domain& domain,
                                    const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  return domain.cells[idx].nbodies;
}

std::tuple<Matrix, Matrix, int64_t>
SymmetricH2::svd_like_compression(Matrix& A, const bool compute_S) const {
  Matrix Ui, Si, Vi;
  int64_t rank;
#ifdef USE_QR_COMPRESSION
  Matrix R;
  const double qr_tol = accuracy * 1e-1;
  std::tie(Ui, R, rank) = error_pivoted_qr(A, qr_tol, use_rel_acc, false);
  if (R.rows > R.cols) {
    R.shrink(R.cols, R.cols);  // Ignore zero entries below
  }
  if (compute_S) {
    Si = Matrix(R.rows, R.rows);
    Vi = Matrix(R.rows, R.cols);
    rq(R, Si, Vi);
  }
#else
  std::tie(Ui, Si, rank) = error_svd_U(A, accuracy, use_rel_acc, false);
#endif

  // Fixed-accuracy with bounded rank
  rank = max_rank > 0 ? std::min(max_rank, rank) : rank;

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(rank));
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_basis(const Domain& domain,
                                        const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  const auto& cell = domain.cells[idx];
  Matrix block_row = generate_p2p_matrix(domain, cell.get_bodies(), cell.sample_farfield);

  Matrix Ui, Si;
  int64_t rank;
  std::tie(Ui, Si, rank) = svd_like_compression(block_row);

  Matrix UxS = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);
  return std::make_tuple(std::move(Ui), std::move(UxS));
}

void SymmetricH2::generate_leaf_nodes(const Domain& domain) {
  const auto num_nodes = level_blocks[height];
  const auto leaf_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j: near_neighbors(i, height)) {
      D.insert(i, j, height,
               generate_p2p_matrix(domain, i, j, leaf_level));
    }
  }
  // Generate leaf level cluster basis
  #pragma omp parallel for
  for (int64_t i = 0; i < num_nodes; i++) {
    const auto idx = domain.get_cell_idx(i, leaf_level);
    const auto& cell = domain.cells[idx];
    if (cell.sample_farfield.size() > 0) {
      Matrix Ui, UxS;
      std::tie(Ui, UxS) =
          generate_row_cluster_basis(domain, i, height);
      #pragma omp critical
      {
        U.insert(i, height, std::move(Ui));
        US_row.insert(i, height, std::move(UxS));
      }
    }
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j: far_neighbors(i, height)) {
      Matrix Dij = generate_p2p_matrix(domain, i, j, leaf_level);
      S.insert(i, j, height,
               matmul(matmul(U(i, height), Dij, true, false),
                      U(j, height)));
    }
  }
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_U_transfer_matrix(const Domain& domain,
                                        const int64_t node, const int64_t level,
                                        const Matrix& Ubig_child1, const Matrix& Ubig_child2) const {
  const auto idx = domain.get_cell_idx(node, level);
  const auto& cell = domain.cells[idx];
  Matrix block_row = generate_p2p_matrix(domain, cell.get_bodies(), cell.sample_farfield);
  auto block_row_splits = block_row.split(vec{Ubig_child1.rows}, vec{});

  Matrix temp(Ubig_child1.cols + Ubig_child2.cols, block_row.cols);
  auto temp_splits = temp.split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, block_row_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Ubig_child2, block_row_splits[1], temp_splits[1], true, false, 1, 0);

  Matrix Ui, Si;
  int64_t rank;
  std::tie(Ui, Si, rank) = svd_like_compression(temp);

  Matrix UxS = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);
  return std::make_tuple(std::move(Ui), std::move(UxS));
}

RowLevelMap
SymmetricH2::generate_transfer_matrices(const Domain& domain,
                                        const int64_t level, RowLevelMap& Uchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;

  const int64_t num_nodes = level_blocks[level];
  #pragma omp parallel for
  for (int64_t node = 0; node < num_nodes; node++) {
    const auto idx = domain.get_cell_idx(node, level);
    const auto& cell = domain.cells[idx];
    const int64_t block_size = get_block_size(domain, node, level);
    const int64_t child1 = node * 2;
    const int64_t child2 = node * 2 + 1;
    const int64_t child_level = level + 1;

    if (cell.sample_farfield.size() > 0) {
      // Generate row cluster transfer matrix.
      const Matrix& Ubig_child1 = Uchild(child1, child_level);
      const Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix Utransfer, UxS;
      std::tie(Utransfer, UxS) =
          generate_U_transfer_matrix(domain, node, level, Ubig_child1, Ubig_child2);

      // Generate the full bases to pass onto the parent.
      auto Utransfer_splits = Utransfer.split(vec{Ubig_child1.cols}, vec{});
      Matrix Ubig(block_size, Utransfer.cols);
      auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);
      #pragma omp critical
      {
        U.insert(node, level, std::move(Utransfer));
        Ubig_parent.insert(node, level, std::move(Ubig));
        US_row.insert(node, level, std::move(UxS));
      }
    }
  }

  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j: far_neighbors(i, level)) {
      Matrix Dij = generate_p2p_matrix(domain, i, j, level);
      S.insert(i, j, level, matmul(matmul(Ubig_parent(i, level), Dij, true, false),
                                   Ubig_parent(j, level)));
    }
  }
  return Ubig_parent;
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
  initialize_geometry_admissibility(domain);
  generate_leaf_nodes(domain);
  RowLevelMap Uchild = U;

  for (int64_t level = height - 1; level > 0; level--) {
    Uchild = generate_transfer_matrices(domain, level, Uchild);
  }
}

int64_t SymmetricH2::get_basis_min_rank(const int64_t level_begin,
                                        const int64_t level_end) const {
  int64_t rank_min = N;
  for (int64_t level = level_begin; level <= level_end; level++) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (U.exists(node, level) && U(node, level).cols > 0) {
        rank_min = std::min(rank_min, U(node, level).cols);
      }
    }
  }
  return rank_min;
}

int64_t SymmetricH2::get_basis_max_rank(const int64_t level_begin,
                                        const int64_t level_end) const {
  int64_t rank_max = -N;
  for (int64_t level = level_begin; level <= level_end; level++) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (U.exists(node, level) && U(node, level).cols > 0) {
        rank_max = std::max(rank_max, U(node, level).cols);
      }
    }
  }
  return rank_max;
}

int64_t SymmetricH2::get_level_max_nblocks(const char nearfar,
                                           const int64_t level_begin, const int64_t level_end) const {
  int64_t csp = 0;
  const bool count_far = (nearfar == 'f' || nearfar == 'a');
  for (int64_t level = level_begin; level <= level_end; level++) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      const bool count_near = (nearfar == 'a') ? (level == height) : (nearfar == 'n');
      const int64_t num_dense   = count_near ? near_neighbors(node, level).size() : 0;
      const int64_t num_lowrank = count_far  ? far_neighbors(node, level).size()  : 0;
      csp = std::max(csp, num_dense + num_lowrank);
    }
  }
  return csp;
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
#ifdef OUTPUT_MEM
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t i = 0; i < num_nodes; i++) {
      if (U.exists(i, level)) {
        mem += U(i, level).memory_used();
      }
      if (US_row.exists(i, level)) {
        mem += US_row(i, level).memory_used();
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
      if (fill_in_neighbors.exists(i, level)) {
        for (auto j: fill_in_neighbors(i, level)) {
          if (F.exists(i, j, level)) {
            mem += F(i, j, level).memory_used();
          }
        }
      }
    }
  }
#endif
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
    for(int64_t node = 0; node < num_nodes; node++) {
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

void SymmetricH2::update_row_cluster_bases(const int64_t row, const int64_t level,
                                           RowMap<Matrix>& r) {
  const int64_t num_nodes = level_blocks[level];
  const int64_t block_size = D(row, row, level).rows;

  // Allocate block_row
  std::vector<int64_t> col_splits;
  int64_t ncols = US_row(row, level).cols;
  col_splits.push_back(ncols);
  for (int64_t j: fill_in_neighbors(row, level)) {
    ncols += F(row, j, level).cols;
    col_splits.push_back(ncols);
  }
  col_splits.pop_back();
  Matrix block_row(block_size, ncols);
  auto block_row_splits = block_row.split({}, col_splits);
  int64_t k = 0;

  // TODO consider implementing a more accurate variant from MiaoMiaoMa2019_UMV paper (Algorithm 1)
  // instead of using a pre-computed UxS from construction phase
  block_row_splits[k++] = US_row(row, level);

  // Concat fill-in blocks
  for (int64_t j: fill_in_neighbors(row, level)) {
    block_row_splits[k++] = F(row, j, level);
  }

  Matrix Ui, Si;
  int64_t rank;
  std::tie(Ui, Si, rank) = svd_like_compression(block_row, false);
  Ui.shrink(Ui.rows, rank);

  Matrix r_row = matmul(Ui, U(row, level), true, false);
  if (r.exists(row)) {
    r.erase(row);
  }
  r.insert(row, std::move(r_row));

  U.erase(row, level);
  U.insert(row, level, std::move(Ui));
}

void SymmetricH2::factorize_level(const Domain& domain, const int64_t level,
                                  RowMap<Matrix>& r) {
  const int64_t num_nodes = level_blocks[level];
  const int64_t parent_level = level - 1;
  for (int64_t node = 0; node < num_nodes; node++) {
    const int64_t parent_node = node / 2;
    const bool found_row_fill_in = (fill_in_neighbors(node, level).size() > 0);
    // Update cluster bases if necessary
    if (found_row_fill_in) {
      update_row_cluster_bases(node, level, r);
      // Project admissible blocks accordingly
      // Current level: update coupling matrix
      #pragma omp parallel for
      for (int64_t j: far_neighbors(node, level)) {
        S(node, j, level) = matmul(r(node), S(node, j, level), false, false);
        S(j, node, level) = matmul(S(j, node, level), r(node), false, true );
      }
      // Upper levels: update transfer matrix one level higher
      // also the pre-computed US_row
      const auto parent_idx = domain.get_cell_idx(parent_node, parent_level);
      const auto& parent_cell = domain.cells[parent_idx];
      if (parent_cell.sample_farfield.size() > 0) {
        const int64_t c1 = parent_node * 2;
        const int64_t c2 = parent_node * 2 + 1;
        Matrix& Utransfer = U(parent_node, parent_level);
        Matrix& US = US_row(parent_node, parent_level);
        Matrix Utransfer_new(U(c1, level).cols + U(c2, level).cols, Utransfer.cols);
        Matrix US_new(U(c1, level).cols + U(c2, level).cols, US.cols);

        auto Utransfer_new_splits = Utransfer_new.split(vec{U(c1, level).cols}, vec{});
        auto US_new_splits = US_new.split(vec{U(c1, level).cols}, vec{});
        if (node == c1) {
          auto Utransfer_splits = Utransfer.split(vec{r(c1).cols}, vec{});
          matmul(r(c1), Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1, 0);
          Utransfer_new_splits[1] = Utransfer_splits[1];

          auto US_splits = US.split(vec{r(c1).cols}, vec{});
          matmul(r(c1), US_splits[0], US_new_splits[0], false, false, 1, 0);
          US_new_splits[1] = US_splits[1];

          r.erase(c1);
        }
        else { // node == c2
          auto Utransfer_splits = Utransfer.split(vec{U(c1, level).cols}, vec{});
          Utransfer_new_splits[0] = Utransfer_splits[0];
          matmul(r(c2), Utransfer_splits[1], Utransfer_new_splits[1], false, false, 1, 0);

          auto US_splits = US.split(vec{U(c1, level).cols}, vec{});
          US_new_splits[0] = US_splits[0];
          matmul(r(c2), US_splits[1], US_new_splits[1], false, false, 1, 0);

          r.erase(c2);
        }
        U.erase(parent_node, parent_level);
        U.insert(parent_node, parent_level, std::move(Utransfer_new));
        US_row.erase(parent_node, parent_level);
        US_row.insert(parent_node, parent_level, std::move(US_new));
      }
    }

    // Multiplication with U_F
    Matrix U_F = prepend_complement_basis(U(node, level));
    // Multiply to dense blocks along the row in current level
    #pragma omp parallel for
    for (int64_t j: near_neighbors(node, level)) {
      if (j < node) {
        // Do not touch the eliminated part (cc and oc)
        int64_t left_col_split = D(node, j, level).cols - U(j, level).cols;
        auto D_splits = D(node, j, level).split(vec{}, vec{left_col_split});
        D_splits[1] = matmul(U_F, D_splits[1], true);
      }
      else {
        D(node, j, level) = matmul(U_F, D(node, j, level), true);
      }
    }
    // Multiply to dense blocks along the column in current level
    #pragma omp parallel for
    for (int64_t i: near_neighbors(node, level)) {
      if (i < node) {
        // Do not touch the eliminated part (cc and co)
        int64_t top_row_split = D(i, node, level).rows - U(i, level).cols;
        auto D_splits = D(i, node, level).split(vec{top_row_split}, vec{});
        D_splits[1] = matmul(D_splits[1], U_F);
      }
      else {
        D(i, node, level) = matmul(D(i, node, level), U_F);
      }
    }

    // The diagonal block is split along the row and column.
    Matrix& D_node = D(node, node, level);
    const auto node_c_size = D_node.rows - U(node, level).cols;
    if (node_c_size > 0) {
      auto D_node_splits = D_node.split(vec{node_c_size}, vec{node_c_size});
      Matrix& D_node_cc = D_node_splits[0];
      ldl(D_node_cc);

      // Lower elimination
      #pragma omp parallel for
      for (int64_t i: near_neighbors(node, level)) {
        Matrix& D_i = D(i, node, level);
        const auto lower_o_size =
            (i <= node || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
        const auto lower_c_size = D_i.rows - lower_o_size;
        auto D_i_splits = D_i.split(vec{lower_c_size}, vec{node_c_size});
        if (i > node && lower_c_size > 0) {
          Matrix& D_i_cc = D_i_splits[0];
          solve_triangular(D_node_cc, D_i_cc, Hatrix::Right, Hatrix::Lower, true, true);
          solve_diagonal(D_node_cc, D_i_cc, Hatrix::Right);
        }
        Matrix& D_i_oc = D_i_splits[2];
        solve_triangular(D_node_cc, D_i_oc, Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(D_node_cc, D_i_oc, Hatrix::Right);
      }

      // Right elimination
      #pragma omp parallel for
      for (int64_t j: near_neighbors(node, level)) {
        Matrix& D_j = D(node, j, level);
        const auto right_o_size =
            (j <= node || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
        const auto right_c_size = D_j.cols - right_o_size;
        auto D_j_splits  = D_j.split(vec{node_c_size}, vec{right_c_size});
        if (j > node && right_c_size > 0) {
          Matrix& D_j_cc = D_j_splits[0];
          solve_triangular(D_node_cc, D_j_cc, Hatrix::Left, Hatrix::Lower, true, false);
          solve_diagonal(D_node_cc, D_j_cc, Hatrix::Left);
        }
        Matrix& D_j_co = D_j_splits[1];
        solve_triangular(D_node_cc, D_j_co, Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(D_node_cc, D_j_co, Hatrix::Left);
      }

      // Schur's complement into inadmissible block
      #pragma omp parallel for collapse(2)
      for (int64_t i: near_neighbors(node, level)) {
        for (int64_t j: near_neighbors(node, level)) {
          if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
            const Matrix& D_i = D(i, node, level);
            const Matrix& D_j = D(node, j, level);
            const auto lower_o_size =
                (i <= node || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
            const auto right_o_size =
                (j <= node || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
            const auto lower_c_size = D_i.rows - lower_o_size;
            const auto right_c_size = D_j.cols - right_o_size;
            const auto D_i_splits  = D_i.split(vec{lower_c_size}, vec{node_c_size});
            const auto D_j_splits  = D_j.split(vec{node_c_size}, vec{right_c_size});
            auto D_ij_splits = D(i, j, level).split(vec{lower_c_size}, vec{right_c_size});

            const Matrix& D_j_cc = D_j_splits[0];
            const Matrix& D_j_co = D_j_splits[1];
            if (i > node && j > node && lower_c_size > 0 && right_c_size > 0) {
              // cc x cc -> cc
              Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
              Matrix& D_ij_cc = D_ij_splits[0];
              column_scale(D_i_cc, D_node_cc);  // LD
              matmul(D_i_cc, D_j_cc, D_ij_cc, false, false, -1, 1);  // LDL^T
            }
            if (i > node && lower_c_size > 0) {
              // cc x co -> co
              Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
              Matrix& D_ij_co = D_ij_splits[1];
              column_scale(D_i_cc, D_node_cc);  // LD
              matmul(D_i_cc, D_j_co, D_ij_co, false, false, -1, 1);  // LDL^T
            }
            if (j > node && right_c_size > 0) {
              // oc x cc -> oc
              Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
              Matrix& D_ij_oc = D_ij_splits[2];
              column_scale(D_i_oc, D_node_cc);  // LD
              matmul(D_i_oc, D_j_cc, D_ij_oc, false, false, -1, 1);  // LDL^T
            }
            {
              // oc x co -> oo
              Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
              Matrix& D_ij_oo = D_ij_splits[3];
              column_scale(D_i_oc, D_node_cc);  // LD
              matmul(D_i_oc, D_j_co, D_ij_oo, false, false, -1, 1);  // LDL^T
            }
          }
        }
      }

      // Schur's complement into admissible block (fill-in)
      #pragma omp parallel for collapse(2)
      for (int64_t i: near_neighbors(node, level)) {
        for (int64_t j: near_neighbors(node, level)) {
          const bool is_admissible_ij =
              !is_admissible.exists(i, j, level) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level));
          const bool fill_ij =
              (i > node && j > node) ||  // b*b       fill-in block
              (i > node && j < node) ||  // b*rank    fill-in block
              (i < node && j > node) ||  // rank*b    fill-in block
              (i < node && j < node);    // rank*rank fill-in block
          if (is_admissible_ij && fill_ij) {
            const Matrix& D_i = D(i, node, level);
            const Matrix& D_j = D(node, j, level);
            const auto lower_o_size =
                (i <= node || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
            const auto right_o_size =
                (j <= node || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
            const auto lower_c_size = D_i.rows - lower_o_size;
            const auto right_c_size = D_j.cols - right_o_size;
            const auto D_i_splits  = D_i.split(vec{lower_c_size}, vec{node_c_size});
            const auto D_j_splits  = D_j.split(vec{node_c_size}, vec{right_c_size});

            Matrix D_i_cc(D_i_splits[0], true);  // Deep-copy
            Matrix D_i_oc(D_i_splits[2], true);  // Deep-copy
            column_scale(D_i_cc, D_node_cc);
            column_scale(D_i_oc, D_node_cc);
            const Matrix& D_j_cc = D_j_splits[0];
            const Matrix& D_j_co = D_j_splits[1];

            Matrix F_ij(D_i.rows, D_j.cols);
            if (i > node && j > node && lower_c_size > 0 && right_c_size > 0) {
              // Create b*b fill-in block
              Matrix fill_in(D_i.rows, D_j.cols);
              auto fill_in_splits = fill_in.split(vec{lower_c_size}, vec{right_c_size});
              matmul(D_i_cc, D_j_cc, fill_in_splits[0], false, false, -1, 1);  // Fill cc part
              matmul(D_i_cc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill co part
              matmul(D_i_oc, D_j_cc, fill_in_splits[2], false, false, -1, 1);  // Fill oc part
              matmul(D_i_oc, D_j_co, fill_in_splits[3], false, false, -1, 1);  // Fill oo part
              F_ij += fill_in;
            }
            if (i > node && j < node && lower_c_size > 0) {
              // Create b*rank fill-in block
              Matrix fill_in(D_i.rows, right_o_size);
              auto fill_in_splits = fill_in.split(vec{lower_c_size}, vec{});
              matmul(D_i_cc, D_j_co, fill_in_splits[0], false, false, -1, 1);  // Fill co part
              matmul(D_i_oc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill oo part
              // b*rank fill-in always has a form of Aik*Vk_c * inv(Akk_cc) x (Uk_c)^T*Akj*Vj_o
              // Convert to b*b block by applying (Vj_o)^T from right
              // Which is safe from bases update since j has been eliminated before (j < k)
              F_ij += matmul(fill_in, U(j, level), false, true);
            }
            if (i < node && j > node && right_c_size > 0) {
              // Create rank*b fill-in block
              Matrix fill_in(lower_o_size, D_j.cols);
              auto fill_in_splits = fill_in.split(vec{}, vec{right_c_size});
              matmul(D_i_oc, D_j_cc, fill_in_splits[0], false, false, -1, 1);  // Fill oc part
              matmul(D_i_oc, D_j_co, fill_in_splits[1], false, false, -1, 1);  // Fill oo part
              // rank*b fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj
              // Convert to b*b block by applying Ui_o from left
              // Which is safe from bases update since i has been eliminated before (i < k)
              F_ij += matmul(U(i, level), fill_in, false, false);
            }
            if (i < node && j < node) {
              // Create rank*rank fill-in block
              Matrix fill_in(lower_o_size, right_o_size);
              matmul(D_i_oc, D_j_co, fill_in, false, false, -1, 1);  // Fill oo part
              // rank*rank fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj*Vj_o
              // Convert to b*b block by applying Ui_o from left and (Vj_o)^T from right
              // Which is safe from bases update since i and j have been eliminated before (i,j < k)
              F_ij += matmul(matmul(U(i, level), fill_in),
                             U(j, level), false, true);
            }
            // Save or accumulate with existing fill-in block that has been propagated from lower level
            #pragma omp critical
            {
              if (!F.exists(i, j, level)) {
                F.insert(i, j, level, std::move(F_ij));
                fill_in_neighbors(i, level).push_back(j);
              }
              else {
                F(i, j, level) += F_ij;
              }
            }
          }
        }
      }
    } // if (node_c_size > 0)
  } // for (int64_t node = 0; node < num_nodes; ++node)
}

void SymmetricH2::factorize(const Domain& domain) {
  // Initialize fill_in_neighbors array
  for (int64_t level = height; level >= min_adm_level; level--) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      fill_in_neighbors.insert(node, level, std::vector<int64_t>());
    }
  }
  for (int64_t level = height; level >= min_adm_level; level--) {
    RowMap<Matrix> r;
    const int64_t num_nodes = level_blocks[level];
    // Make sure all cluster bases exist and none of them is full-rank
    for (int64_t i = 0; i < num_nodes; ++i) {
      if (!U.exists(i, level)) {
        throw std::logic_error("Cluster bases not found at U(" + std::to_string(i) +
                               "," + std::to_string(level) + ")");
      }
    }
    factorize_level(domain, level, r);

    // Update coupling matrices of admissible blocks in the current level
    // To add fill-in contributions
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t j: far_neighbors(i, level)) {
        if (F.exists(i, j, level)) {
          S(i, j, level) += matmul(matmul(U(i, level), F(i, j, level), true),
                                   U(j, level));
        }
      }
    }

    const int64_t parent_level = level - 1;
    const int64_t parent_num_nodes = level_blocks[parent_level];
    // Propagate fill-in to upper level admissible blocks (if any)
    if (parent_level >= min_adm_level) {
      // Mark parent node that has fill-in coming from the current level
      RowMap<std::set<int64_t>> parent_fill_in_neighbors;
      for (int64_t i = 0; i < parent_num_nodes; i++) {
        parent_fill_in_neighbors.insert(i, std::set<int64_t>());
      }
      for (int64_t i = 0; i < num_nodes; i++) {
        for (int64_t j: fill_in_neighbors(i, level)) {
          const int64_t ip = i / 2;
          const int64_t jp = j / 2;
          if ((!is_admissible.exists(ip, jp, parent_level)) ||
              (is_admissible.exists(ip, jp, parent_level) && is_admissible(ip, jp, parent_level))) {
            parent_fill_in_neighbors(ip).insert(jp);
          }
        }
      }
      for (int64_t i = 0; i < parent_num_nodes; i++) {
        for (int64_t j: parent_fill_in_neighbors(i)) {
          fill_in_neighbors(i, parent_level).push_back(j);
        }
      }
      // Propagate fill-ins to parent level
      for (int64_t i = 0; i < parent_num_nodes; ++i) {
        for (int64_t j: fill_in_neighbors(i, parent_level)) {
          const auto i1 = i * 2;
          const auto i2 = i * 2 + 1;
          const auto j1 = j * 2;
          const auto j2 = j * 2 + 1;
          const auto nrows = U(i1, level).cols + U(i2, level).cols;
          const auto ncols = U(j1, level).cols + U(j2, level).cols;
          Matrix fill_in(nrows, ncols);
          auto fill_in_splits = fill_in.split(vec{U(i1, level).cols},
                                              vec{U(j1, level).cols});
          if (F.exists(i1, j1, level)) {
            matmul(matmul(U(i1, level), F(i1, j1, level), true, false),
                   U(j1, level), fill_in_splits[0], false, false, 1, 0);
          }
          if (F.exists(i1, j2, level)) {
            matmul(matmul(U(i1, level), F(i1, j2, level), true, false),
                   U(j2, level), fill_in_splits[1], false, false, 1, 0);
          }
          if (F.exists(i2, j1, level)) {
            matmul(matmul(U(i2, level), F(i2, j1, level), true, false),
                   U(j1, level), fill_in_splits[2], false, false, 1, 0);
          }
          if (F.exists(i2, j2, level)) {
            matmul(matmul(U(i2, level), F(i2, j2, level), true, false),
                   U(j2, level), fill_in_splits[3], false, false, 1, 0);
          }
          F.insert(i, j, parent_level, std::move(fill_in));
        }
      }
      // Put identity bases when all dense is encountered in parent level
      for (int64_t node = 0; node < num_nodes; node += 2) {
        int64_t parent_node = node / 2;
        if (!U.exists(parent_node, parent_level)) {
          // Use identity matrix as U bases whenever all dense row is encountered
          int64_t c1 = node;
          int64_t c2 = node + 1;
          int64_t rank_c1 = U(c1, level).cols;
          int64_t rank_c2 = U(c2, level).cols;
          int64_t rank_parent = std::max(rank_c1, rank_c2);
          Matrix Utransfer =
              generate_identity_matrix(rank_c1 + rank_c2, rank_parent);

          if (r.exists(c1)) r.erase(c1);
          if (r.exists(c2)) r.erase(c2);
          U.insert(parent_node, parent_level, std::move(Utransfer));
        }
      }
    }

    // Merge the unfactorized parts.
    for (int64_t i = 0; i < parent_num_nodes; ++i) {
      for (int64_t j: near_neighbors(i, parent_level)) {
        std::vector<int64_t> i_children, j_children;
        std::vector<int64_t> row_split, col_split;
        int64_t nrows=0, ncols=0;
        if (matrix_type == BLR2_MATRIX) {
          for (int64_t n = 0; n < level_blocks[level]; ++n) {
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
          for (int64_t n = 0; n < 2; ++n) {
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

        for (int64_t ic1 = 0; ic1 < i_children.size(); ++ic1) {
          for (int64_t jc2 = 0; jc2 < j_children.size(); ++jc2) {
            int64_t c1 = i_children[ic1], c2 = j_children[jc2];
            if (!U.exists(c1, level)) { continue; }

            if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {
              auto D_splits = D(c1, c2, level).split(
                  vec{D(c1, c2, level).rows - U(c1, level).cols},
                  vec{D(c1, c2, level).cols - U(c2, level).cols});
              D_unelim_splits[ic1 * j_children.size() + jc2] = D_splits[3];
            }
            else {
              D_unelim_splits[ic1 * j_children.size() + jc2] = S(c1, c2, level);
            }
          }
        }

        D.insert(i, j, parent_level, std::move(D_unelim));
      }
    }
  } // for (int64_t level = height; level >= min_adm_level; level--)

  // Factorize remaining blocks as block dense matrix
  const auto level = min_adm_level - 1;
  const auto num_nodes = level_blocks[level];
  for (int64_t k = 0; k < num_nodes; k++) {
    ldl(D(k, k, level));
    // Lower elimination
    #pragma omp parallel for
    for (int64_t i = k + 1; i < num_nodes; i++) {
      solve_triangular(D(k, k, level), D(i, k, level), Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(D(k, k, level), D(i, k, level), Hatrix::Right);
    }
    // Right elimination
    #pragma omp parallel for
    for (int64_t j = k + 1; j < num_nodes; j++) {
      solve_triangular(D(k, k, level), D(k, j, level), Hatrix::Left, Hatrix::Lower, true, false);
      solve_diagonal(D(k, k, level), D(k, j, level), Hatrix::Left);
    }
    // Schur's complement
    #pragma omp parallel for collapse(2)
    for (int64_t i = k + 1; i < num_nodes; i++) {
      for (int64_t j = k + 1; j < num_nodes; j++) {
        Matrix Dik(D(i, k, level), true);  // Deep-copy
        column_scale(Dik, D(k, k, level));  // LD
        matmul(Dik, D(k, j, level), D(i, j, level), false, false, -1, 1);
      }
    }
  }
}

void shift_diag(Matrix& A, const double shift) {
  for(int64_t i = 0; i < A.min_dim(); i++) {
    A(i, i) += shift;
  }
}

int inertia(const SymmetricH2& A, const Domain& domain,
            const double lambda, bool &singular) {
  SymmetricH2 A_shifted(A);
  // Shift leaf level diagonal blocks
  const int64_t leaf_num_nodes = A.level_blocks[A.height];
  for(int64_t node = 0; node < leaf_num_nodes; node++) {
    shift_diag(A_shifted.D(node, node, A.height), -lambda);
  }
  // LDL Factorize
  A_shifted.factorize(domain);
  // Count negative entries in D
  int negative_elements_count = 0;
  for(int64_t level = A.height; level >= A.min_adm_level; level--) {
    const int64_t num_nodes = A.level_blocks[level];
    for(int64_t node = 0; node < num_nodes; node++) {
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
  // Remaining blocks that are factorized as block-dense matrix
  {
    const auto level = A.min_adm_level - 1;
    const auto num_nodes = A.level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      const Matrix& D_lambda = A_shifted.D(node, node, level);
      for(int64_t i = 0; i < D_lambda.min_dim(); i++) {
        negative_elements_count += (D_lambda(i, i) < 0 ? 1 : 0);
        if(std::isnan(D_lambda(i, i)) || std::abs(D_lambda(i, i)) < EPS) singular = true;
      }
    }
  }
  return negative_elements_count;
}

double get_kth_eigenvalue(const SymmetricH2& A, const Domain& domain,
                          const double ev_tol, const int idx_k,
                          const int k_list_size, const std::vector<int>& k_list,
                          std::vector<double>& a,
                          std::vector<double>& b) {
  const auto k = k_list[idx_k];
  bool singular = false;
  while((b[idx_k] - a[idx_k]) >= ev_tol) {
#ifdef DEBUG_OUTPUT
    printf("Local starting intervals:\n");
    for (int64_t idx = 0; idx < k_list_size; idx++) {
      printf("k=%5d a=%10.5lf b=%10.5lf\n",
             (int)k_list[idx], a[idx], b[idx]);
    }
#endif
    const auto mid = (a[idx_k] + b[idx_k]) / 2;
    const int v_mid = inertia(A, domain, mid, singular);
    if(singular) {
      printf("Shifted matrix becomes singular (shift=%.5lf)\n", mid);
      break;
    }
    // Update intervals accordingly
    for (int idx = 0; idx < k_list_size; idx++) {
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

// MPI Related Definitions
#define MPI_TAG_TASK 0
#define MPI_TAG_SPLIT_TASK 1
#define MPI_TAG_RESULT 2
#define MPI_TAG_FINISH 3
typedef struct Task {
  // Defines a task to compute the k0-th to k1-th eigenvalues within the interval [a, b]
  int k0;
  int k1;
  double a;
  double b;
} Task;
#define TASK_LEN 4

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_nprocs < 2) {
    printf("Error: number of processes has to be larger than 1\n");
    exit(EXIT_FAILURE);
  }

  int64_t N = argc > 1 ? atol(argv[1]) : 256;
  int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1e-8;
  // Use relative or absolute error threshold for LRA
  const bool use_rel_acc = argc > 4 ? (atol(argv[4]) == 1) : false;
  // Fixed accuracy with bounded rank
  const int64_t max_rank = argc > 5 ? atol(argv[5]) : 0;
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
  // 4: Random Uniform Grid
  const int64_t geom_type = argc > 9 ? atol(argv[9]) : 0;
  int64_t ndim  = argc > 10 ? atol(argv[10]) : 1;
    // Eigenvalue computation parameters
  const double ev_tol = argc > 11 ? atof(argv[11]) : 1e-3;
  int64_t k_begin = argc > 12 ? atol(argv[12]) : 1;
  int64_t k_end = argc > 13 ? atol(argv[13]) : k_begin;
  const double a = argc > 14 ? atof(argv[14]) : 0;
  const double b = argc > 15 ? atof(argv[15]) : N;
  const int64_t m = argc > 16 ? atol(argv[16]) : 2;
  const bool compute_eig_acc = argc > 17 ? (atol(argv[17]) == 1) : true;
  const int64_t print_csv_header = argc > 18 ? atol(argv[18]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 19 ? std::string(argv[19]) : "";
  const int64_t sort_bodies = argc > 20 ? atol(argv[20]) : 0;

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

  // Construct H2-Matrix
  // At the moment all processes redundantly construct the same instance of H2-Matrix
  // TODO make only master construct and broadcast to all slaves
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
  domain.build_sample_bodies(N, N, N, 0, geom_type == 3);  // No sampling, use all bodies
  const auto start_construct = MPI_Wtime();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type);
  const auto stop_construct = MPI_Wtime();
  const auto construct_min_rank = A.get_basis_min_rank(1, A.height);
  const auto construct_max_rank = A.get_basis_max_rank(1, A.height);
  const auto construct_time = stop_construct - start_construct;
  const auto construct_error = A.construction_error(domain);
  const auto construct_mem = A.memory_usage();
  const auto construct_min_rank_leaf = A.get_basis_min_rank(A.height, A.height);
  const auto construct_max_rank_leaf = A.get_basis_max_rank(A.height, A.height);
  const auto csp = A.get_level_max_nblocks('a', 1, A.height);
  const auto csp_dense_leaf = A.get_level_max_nblocks('n', A.height, A.height);
  const auto csp_dense_all = A.get_level_max_nblocks('n', 1, A.height);
  const auto csp_lr_all = A.get_level_max_nblocks('f', 1, A.height);

  // All processes finished construction
  MPI_Barrier(MPI_COMM_WORLD);

#ifndef OUTPUT_CSV
  if (mpi_rank == 0) {
      printf("mpi_nprocs=%d N=%d leaf_size=%d accuracy=%.1e acc_type=%d max_rank=%d"
             " admis=%.1lf matrix_type=%d kernel=%s geometry=%s height=%d"
             " construct_min_rank=%d construct_max_rank=%d construct_mem=%d"
             " construct_time=%.3lf construct_error=%.5e"
             " csp=%d csp_dense_leaf=%d csp_dense_all=%d csp_lr_all=%d\n",
             mpi_nprocs, (int)N, (int)leaf_size, accuracy, (int)use_rel_acc, (int)max_rank, admis,
             (int)matrix_type, kernel_name.c_str(), geom_name.c_str(), (int)A.height,
             (int)construct_min_rank, (int)construct_max_rank, (int)construct_mem, construct_time,
             construct_error, (int)csp, (int)csp_dense_leaf, (int)csp_dense_all, (int)csp_lr_all);
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
    v_a = inertia(A, domain, a, s);
    v_b = inertia(A, domain, b, s);
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
    printf("mpi_nprocs,N,leaf_size,accuracy,acc_type,max_rank,admis,matrix_type,kernel,geometry"
           ",height,construct_min_rank,construct_max_rank,construct_mem,construct_time,construct_error"
           ",csp,csp_dense_leaf,csp_dense_all,csp_lr_all,dense_eig_time_all,h2_eig_time_all"
           ",m,k,a,b,v_a,v_b,ev_tol,dense_ev,h2_ev,eig_abs_err,success\n");
  }
#endif

  // Initialize variables
  MPI_Status status;
  bool is_finished;
  int result_count;
  double h2_ev_time = 0;
  std::vector<double> h2_ev;
  std::vector<double> result_buffer(2 + (2 * m)); // First two numbers are k0 and k1

  // Begin computing target eigenvalues
  MPI_Barrier(MPI_COMM_WORLD);
  h2_ev_time -= MPI_Wtime();

  if (mpi_rank == 0) {  // Master
    std::queue<Task> task_pool;
    std::vector<bool> is_idle(mpi_nprocs, true);
    std::vector<double> in_task_buffer(2 * TASK_LEN);
    std::vector<double> out_task_buffer(mpi_nprocs * TASK_LEN);
    std::vector<MPI_Request> out_task_requests(mpi_nprocs-1), finish_requests(mpi_nprocs-1);
    h2_ev.assign(N, -1);
    is_idle[mpi_rank] = false;  // Set master node as working

    const int64_t target_count = k_end - k_begin + 1;
    task_pool.push(Task{(int)k_begin, (int)k_end, a, b});
#ifdef DEBUG_OUTPUT
    printf("Master: Inserted task (%d, %d, %.5lf, %.5lf) into task_pool\n",
           (int)k_begin, (int)k_end, a, b);
#endif
    int64_t finished_count = 0;
    int64_t n_idle_procs = mpi_nprocs - 1;
    while (finished_count < target_count) {
      for (int i = 1; (i < mpi_nprocs) && (n_idle_procs > 0) && (!task_pool.empty()); i++) {
        if (is_idle[i]) {
          // Send task to process number i
          const auto task = task_pool.front();
          out_task_buffer[i * TASK_LEN + 0] = (double)task.k0;
          out_task_buffer[i * TASK_LEN + 1] = (double)task.k1;
          out_task_buffer[i * TASK_LEN + 2] = task.a;
          out_task_buffer[i * TASK_LEN + 3] = task.b;

          MPI_Isend(out_task_buffer.data() + i * TASK_LEN, TASK_LEN, MPI_DOUBLE,
                    i, MPI_TAG_TASK, MPI_COMM_WORLD, &out_task_requests[i-1]);
          MPI_Request_free(&out_task_requests[i-1]);
          task_pool.pop();
          is_idle[i] = false;  // Set process i as working
          n_idle_procs--;
#ifdef DEBUG_OUTPUT
          printf("Master: Sent task (%d, %d, %.5lf, %.5lf) to Slave-%d\n",
                 task.k0, task.k1, task.a, task.b, i);
#endif
        }
      }
      // Receive message from slave
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case MPI_TAG_SPLIT_TASK: {
          // Receive 8 numbers denoting two new tasks
          MPI_Recv(in_task_buffer.data(), 2 * TASK_LEN, MPI_DOUBLE,
                   status.MPI_SOURCE, MPI_TAG_SPLIT_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Master: Received split task from Slave-%d\n", status.MPI_SOURCE);
#endif
          // Insert two new tasks into task_pool
          Task task_left  {
            (int)in_task_buffer[0], (int)in_task_buffer[1],
            in_task_buffer[2], in_task_buffer[3]
          };
          Task task_right {
            (int)in_task_buffer[4], (int)in_task_buffer[5],
            in_task_buffer[6], in_task_buffer[7]
          };
          const auto left_size = task_left.k1 - task_left.k0 + 1;
          const auto right_size = task_right.k1 - task_right.k0 + 1;
          if (left_size > 0) {
            task_pool.push(task_left);
#ifdef DEBUG_OUTPUT
            printf("Master: Inserted task (%d, %d, %.5lf, %.5lf) into task_pool\n",
                   task_left.k0, task_left.k1, task_left.a, task_left.b);
#endif
          }
          if (right_size > 0) {
            task_pool.push(task_right);
#ifdef DEBUG_OUTPUT
            printf("Master: Inserted task (%d, %d, %.5lf. %.5lf) into task_pool\n",
                   task_right.k0, task_right.k1, task_right.a, task_right.b);
#endif
          }
          break;
        }
        case MPI_TAG_RESULT: {
          // Receive a number of eigenvalues from slave
          MPI_Get_count(&status, MPI_DOUBLE, &result_count);
          MPI_Recv(result_buffer.data(), result_count, MPI_DOUBLE,
                   status.MPI_SOURCE, MPI_TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // The first two numbers are the order of the eigenvalues
          const int k0 = (int)result_buffer[0];
          const int k1 = (int)result_buffer[1];
          int k = k0;
          for (int j = 2; j < result_count; j++) {
            h2_ev[k - 1] = result_buffer[j];
            k++;
          }
#ifdef DEBUG_OUTPUT
          char outBuffer[1024];
          int offset = 0;
          int charsWritten;
          for (int j = 2; j < result_count; j++) {
            charsWritten = snprintf(outBuffer+offset, 1024-offset, "%.5lf ", result_buffer[j]);
            offset += charsWritten;
          }
          printf("Master: Received %d to %d eigenvalues=[ %s] from Slave-%d\n",
                 k0, k1, outBuffer, status.MPI_SOURCE);
#endif
          finished_count += result_count - 2;
          is_idle[status.MPI_SOURCE] = true;
          n_idle_procs++;
          break;
        }
      }
    }
#ifdef DEBUG_OUTPUT
    printf("Master: All %d eigenvalues have been received\n", (int)finished_count);
#endif
    // Send finish signal to all process
    is_finished = true;
    for (int i = 1; i < mpi_nprocs; i++) {
      MPI_Isend(&is_finished, 1, MPI_C_BOOL, i, MPI_TAG_FINISH, MPI_COMM_WORLD, &finish_requests[i-1]);
      MPI_Request_free(&finish_requests[i-1]);
#ifdef DEBUG_OUTPUT
      printf("Master: Sent finish signal to Slave-%d\n", i);
#endif
    }
#ifdef DEBUG_OUTPUT
    printf("Master: Finished\n");
#endif
  }
  else {  // Slave
    std::vector<MPI_Request> send_requests(2);
    std::vector<double> in_task_buffer(TASK_LEN);
    std::vector<double> out_task_buffer(2 * TASK_LEN);
    std::vector<int> local_k (2 * m);  // Local target eigenvalue indices
    std::vector<double> local_a(2 * m), local_b(2 * m);  // Local starting intervals
    is_finished = false;
    while(!is_finished) {
      // Wait for message from master
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      switch (status.MPI_TAG) {
        case MPI_TAG_TASK: {
          // Receive task from master
          MPI_Recv(in_task_buffer.data(), TASK_LEN, MPI_DOUBLE,
                   0, MPI_TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Slave-%d: Received task (%d, %d, %.5lf, %.5lf) from Master\n",
                 mpi_rank, (int)in_task_buffer[0], (int)in_task_buffer[1],
                 in_task_buffer[2], in_task_buffer[3]);
#endif
          Task task {
            (int)in_task_buffer[0], (int)in_task_buffer[1],
            in_task_buffer[2], in_task_buffer[3]
          };
          std::fill(local_a.begin(), local_a.end(), task.a);
          std::fill(local_b.begin(), local_b.end(), task.b);
          int task_size = task.k1 - task.k0 + 1;
          const bool split_task = task_size > 2 * m;
          if (split_task) {
            // Compute only m inner eigenvalues
            const int new_k0 = task.k0 + (task_size / 2) - (m / 2);
            const int new_k1 = new_k0 + m - 1;
            task_size = new_k1 - new_k0 + 1;
            for (int k = new_k0; k <= new_k1; k++) {
              local_k[k - new_k0] = k;
            }
            const double new_k0_ev = get_kth_eigenvalue(A, domain, ev_tol, 0, task_size,
                                                        local_k, local_a, local_b);
            const double new_k1_ev = get_kth_eigenvalue(A, domain, ev_tol, task_size - 1, task_size,
                                                        local_k, local_a, local_b);
            // Split into two tasks
            Task task_left   { task.k0, new_k0 - 1, task.a, new_k0_ev + ev_tol };
            Task task_right  { new_k1 + 1, task.k1, new_k1_ev - ev_tol, task.b };
            out_task_buffer[0] = (double)task_left.k0;
            out_task_buffer[1] = (double)task_left.k1;
            out_task_buffer[2] = task_left.a;
            out_task_buffer[3] = task_left.b;
            out_task_buffer[4] = (double)task_right.k0;
            out_task_buffer[5] = (double)task_right.k1;
            out_task_buffer[6] = task_right.a;
            out_task_buffer[7] = task_right.b;
            // Update current task
            task.k0 = new_k0;
            task.k1 = new_k1;
            task.a = new_k0_ev;
            task.b = new_k1_ev;
            // Send split task to master
            MPI_Isend(out_task_buffer.data(), 2 * TASK_LEN, MPI_DOUBLE,
                      0, MPI_TAG_SPLIT_TASK, MPI_COMM_WORLD, &send_requests[1]);
#ifdef DEBUG_OUTPUT
            printf("Slave-%d: Return split task (%d, %d, %.5lf, %.5lf) and "
                   "(%d, %d, %.5lf, %.5lf) to Master\n", mpi_rank,
                   (int)out_task_buffer[0], (int)out_task_buffer[1], out_task_buffer[2], out_task_buffer[3],
                   (int)out_task_buffer[4], (int)out_task_buffer[5], out_task_buffer[6], out_task_buffer[7]);
#endif
          }
          else {
            // Initialize local_k
            for (int k = task.k0; k <= task.k1; k++) {
              local_k[k - task.k0] = k;
            }
          }
          // Perform current task
          result_count = 2 + (task.k1 - task.k0 + 1);  // First two numbers are k0 and k1
          result_buffer[0] = task.k0;
          result_buffer[1] = task.k1;
          int idx_k_begin = 0;
          int idx_k_end = task_size - 1;
          if (split_task) {
            // First and last eigenvalues have been computed during split process. Save them
            result_buffer[2 + idx_k_begin] = task.a;
            result_buffer[2 + idx_k_end] = task.b;
            idx_k_begin++;
            idx_k_end--;
          }
          for (int idx_k = idx_k_begin; idx_k <= idx_k_end; idx_k++) {
            result_buffer[2 + idx_k] = get_kth_eigenvalue(A, domain, ev_tol, idx_k, task_size,
                                                          local_k, local_a, local_b);
          }
          // Send result of current task
          MPI_Isend(result_buffer.data(), result_count, MPI_DOUBLE,
                    0, MPI_TAG_RESULT, MPI_COMM_WORLD, &send_requests[0]);
#ifdef DEBUG_OUTPUT
          char outBuffer[1024];
          int offset = 0;
          int charsWritten;
          for (int j = 2; j < result_count; j++) {
            charsWritten = snprintf(outBuffer+offset, 1024-offset, "%.5lf ", result_buffer[j]);
            offset += charsWritten;
          }
          printf("Slave-%d: Sent %d eigenvalues=[ %s] to Master\n",
                 mpi_rank, result_count - 2, outBuffer);
#endif
          // Wait until all sends are done
          int ready = 0;
          while (!ready) {
            MPI_Testall(split_task ? 2 : 1, send_requests.data(), &ready, MPI_STATUS_IGNORE);
          }
          break;
        }
        case MPI_TAG_FINISH: {
          // Receive finish signal
          MPI_Recv(&is_finished, 1, MPI_C_BOOL, 0, MPI_TAG_FINISH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef DEBUG_OUTPUT
          printf("Slave-%d: Received finish signal from Master\n", mpi_rank);
#endif
          break;
        }
      }
    }
#ifdef DEBUG_OUTPUT
    printf("Slave-%d: Finished\n", mpi_rank);
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  h2_ev_time += MPI_Wtime();

  if (mpi_rank == 0) {
    for (int k = k_begin; k <= k_end; k++) {
      const double dense_ev_k = compute_eig_acc ? dense_ev[k-1] : -1;
      const double h2_ev_k = h2_ev[k-1];
      const double eig_abs_err = compute_eig_acc ? std::abs(dense_ev_k - h2_ev_k) : -1;
      const std::string success = eig_abs_err < (0.5 * ev_tol) ? "TRUE" : "FALSE";
#ifndef OUTPUT_CSV
      printf("h2_eig_time_all=%.3lf m=%d k=%d a=%.2lf b=%.2lf v_a=%d v_b=%d ev_tol=%.1e"
             " dense_ev=%.8lf h2_ev=%.8lf eig_abs_err=%.2e success=%s\n",
             h2_ev_time, (int)m, k, a, b, v_a, v_b, ev_tol,
             dense_ev_k, h2_ev_k, eig_abs_err, success.c_str());
#else
      printf("%d,%d,%d,%.1e,%d,%d,%.1lf,%d,%s,%s,%d,%d,%d,%d,%.3lf,%.5e,%d,%d,%d,%d"
             ",%.3lf,%.3lf,%d,%d,%.2lf,%.2lf,%d,%d,%.1e,%.8lf,%.8lf,%.2e,%s\n",
             mpi_nprocs,(int)N, (int)leaf_size, accuracy, (int)use_rel_acc, (int)max_rank,
             admis, (int)matrix_type, kernel_name.c_str(), geom_name.c_str(), (int)A.height,
             (int)construct_min_rank, (int)construct_max_rank, (int)construct_mem, construct_time,
             construct_error, (int)csp, (int)csp_dense_leaf, (int)csp_dense_all, (int)csp_lr_all,
             dense_eig_time, h2_ev_time, (int)m, k, a, b, v_a, v_b, ev_tol,
             dense_ev_k, h2_ev_k, eig_abs_err, success.c_str());
#endif
    }
  }

  Hatrix::Context::finalize();
  MPI_Finalize();
  return 0;
}

