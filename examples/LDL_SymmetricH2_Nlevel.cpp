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
#include <set>

#include "nlohmann/json.hpp"

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

using vec = std::vector<int64_t>;

#define OUTPUT_CSV

// Comment the following line to use SVD instead of pivoted QR for low-rank compression
// #define USE_QR_COMPRESSION

// H2-Construction employ multiplication with a random matrix to reduce far-block matrix size
// Quite accurate and does not rely on ID but incur O(N^2) complexity to construct basis and coupling matrices
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
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
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
  std::tuple<Matrix, Matrix, Matrix, int64_t> svd_like_compression(Matrix& A) const;

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
  void fill_JSON(const Domain& domain,
                 const int64_t i, const int64_t j,
                 const int64_t level, nlohmann::json& json) const;

  void update_row_cluster_bases(const int64_t row, const int64_t level,
                                RowMap<Matrix>& r);
  void factorize_level(const Domain& domain, const int64_t level,
                       RowMap<Matrix>& r);

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

  int64_t get_basis_min_rank(const int64_t level_begin, const int64_t level_end) const;
  int64_t get_basis_max_rank(const int64_t level_begin, const int64_t level_end) const;
  int64_t get_level_max_nblocks(const char nearfar,
                                const int64_t level_begin, const int64_t level_end) const;
  double construction_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  void write_JSON(const Domain& domain, const std::string filename) const;

  long long int factorize(const Domain& domain);
  Matrix solve(const Matrix& b) const;
  double solve_error(const Matrix& x, const Matrix& ref) const;
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
  }
}

int64_t SymmetricH2::get_block_size(const Domain& domain,
                                    const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  return domain.cells[idx].nbodies;
}

std::tuple<Matrix, Matrix, Matrix, int64_t>
SymmetricH2::svd_like_compression(Matrix& A) const {
  Matrix Ui, Si, Vi;
  int64_t rank;
#ifdef USE_QR_COMPRESSION
  Matrix R;
  const double qr_tol = accuracy * 1e-1;
  std::tie(Ui, R, rank) = error_pivoted_qr(A, qr_tol, use_rel_acc, false);
  Si = Matrix(R.rows, R.rows);
  Vi = Matrix(R.rows, R.cols);
  rq(R, Si, Vi);
#else
  if (max_rank > 0) { // Randomized SVD
    const auto k = max_rank + 10; // oversampling
    const Matrix Y = generate_random_matrix(A.cols, k);
    Matrix AY = matmul(A, Y);
    std::tie(Ui, Si, Vi, rank) = error_svd(AY, accuracy, use_rel_acc, false);
  }
  else { // SVD
    std::tie(Ui, Si, Vi, rank) = error_svd(A, accuracy, use_rel_acc, false);
  }
#endif

  // Fixed-accuracy with bounded rank
  rank = max_rank > 0 ? std::min(max_rank, rank) : rank;

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi), std::move(rank));
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_basis(const Domain& domain,
                                        const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  const auto& cell = domain.cells[idx];
  Matrix block_row = generate_p2p_matrix(domain, cell.get_bodies(), cell.sample_farfield);

  Matrix Ui, Si, Vi_T;
  int64_t rank;
  std::tie(Ui, Si, Vi_T, rank) = svd_like_compression(block_row);

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
  for (int64_t i = 0; i < num_nodes; i++) {
    const auto idx = domain.get_cell_idx(i, leaf_level);
    const auto& cell = domain.cells[idx];
    if (cell.sample_farfield.size() > 0) {
      Matrix Ui, UxS;
      std::tie(Ui, UxS) =
          generate_row_cluster_basis(domain, i, height);
      U.insert(i, height, std::move(Ui));
      US_row.insert(i, height, std::move(UxS));
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

  Matrix Ui, Si, Vi;
  int64_t rank;
  std::tie(Ui, Si, Vi, rank) = svd_like_compression(temp);

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
      U.insert(node, level, std::move(Utransfer));
      US_row.insert(node, level, std::move(UxS));

      // Generate the full bases to pass onto the parent.
      auto Utransfer_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});
      Matrix Ubig(block_size, U(node, level).cols);
      auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});

      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);
      Ubig_parent.insert(node, level, std::move(Ubig));
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
  bool is_near = (nearfar == 'n' || nearfar == 'a');
  bool is_far = (nearfar == 'f' || nearfar == 'a');
  for (int64_t level = level_begin; level <= level_end; level++) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      const int64_t num_dense   = is_near ? near_neighbors(node, level).size() : 0;
      const int64_t num_lowrank = is_far  ? far_neighbors(node, level).size()  : 0;
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

void SymmetricH2::update_row_cluster_bases(const int64_t row, const int64_t level,
                                           RowMap<Matrix>& r) {
  const int64_t num_nodes = level_blocks[level];
  const int64_t block_size = D(row, row, level).rows;
  Matrix block_row(block_size, 0);

  // TODO consider implementing a more accurate variant from MiaoMiaoMa2019_UMV paper (Algorithm 1)
  // instead of using a pre-computed UxS from construction phase
  block_row = concat(block_row, US_row(row, level), 1);

  // Concat fill-in blocks
  for (int64_t j: fill_in_neighbors(row, level)) {
    block_row = concat(block_row, F(row, j, level), 1);
  }

  Matrix Ui, Si, Vi;
  int64_t rank;
  std::tie(Ui, Si, Vi, rank) = svd_like_compression(block_row);
  Matrix US = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);

  Matrix r_row = matmul(Ui, U(row, level), true, false);
  if (r.exists(row)) {
    r.erase(row);
  }
  r.insert(row, std::move(r_row));

  U.erase(row, level);
  U.insert(row, level, std::move(Ui));

  US_row.erase(row, level);
  US_row.insert(row, level, std::move(US));
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
    } // if (node_c_size > 0)
  } // for (int64_t node = 0; node < num_nodes; ++node)
}

long long int SymmetricH2::factorize(const Domain& domain) {
  Hatrix::profiling::PAPI papi;
  papi.add_fp_ops(0);
  papi.start();
  // Initialize fill_in_neighbors array
  for (int64_t level = height; level > 0; level--) {
    const int64_t num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      fill_in_neighbors.insert(node, level, std::vector<int64_t>());
    }
  }
  for (int64_t level = height; level > 0; level--) {
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
    if (parent_level > 0) {
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
  } // for (int64_t level = height; level > 0; level--)

  // Factorize remaining root level
  ldl(D(0, 0, 0));

  auto fp_ops = papi.fp_ops();
  return fp_ops;
}

// Permute the vector forward and return the offset at which the new vector begins.
int64_t SymmetricH2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const {
  Matrix copy(x);
  const int64_t num_nodes = level_blocks[level];
  const int64_t c_offset = rank_offset;
  for (int64_t node = 0; node < num_nodes; ++node) {
    rank_offset += D(node, node, level).rows - U(node, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t node = 0; node < num_nodes; ++node) {
    const int64_t rows = D(node, node, level).rows;
    const int64_t rank = U(node, level).cols;
    const int64_t c_size = rows - rank;
    // Copy the complement part of the vector into the temporary vector
    for (int64_t i = 0; i < c_size; ++i) {
      copy(c_offset + csize_offset + i, 0) = x(c_offset + bsize_offset + i, 0);
    }
    // Copy the rank part of the vector into the temporary vector
    for (int64_t i = 0; i < rank; ++i) {
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
  const int64_t num_nodes = level_blocks[level];
  int64_t c_offset = rank_offset;
  for (int64_t node = 0; node < num_nodes; ++node) {
    c_offset -= D(node, node, level).cols - U(node, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t node = 0; node < num_nodes; ++node) {
    const int64_t cols = D(node, node, level).cols;
    const int64_t rank = U(node, level).cols;
    const int64_t c_size = cols - rank;

    for (int64_t i = 0; i < c_size; ++i) {
      copy(c_offset + bsize_offset + i, 0) = x(c_offset + csize_offset + i, 0);
    }
    for (int64_t i = 0; i < rank; ++i) {
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
  const int64_t num_nodes = level_blocks[level];
  std::vector<int64_t> row_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < num_nodes; ++i) {
    row_offsets.push_back(nrows + D(i, i, level).rows);
    nrows += D(i, i, level).rows;
  }
  auto x_level_split = x_level.split(row_offsets, vec{});

  for (int64_t node = 0; node < num_nodes; ++node) {
    const int64_t diag_row_split = D(node, node, level).rows - U(node, level).cols;
    const int64_t diag_col_split = D(node, node, level).cols - U(node, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    // Multiply with (U_F)^T
    Matrix U_F = prepend_complement_basis(U(node, level));
    Matrix x_node = matmul(U_F, x_level_split[node], true);
    auto x_node_splits = x_node.split(vec{diag_col_split}, vec{});
    // Solve forward with diagonal L
    auto L_node_splits = D(node, node, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_triangular(L_node_splits[0], x_node_splits[0], Hatrix::Left, Hatrix::Lower, true);
    // Forward substitution with oc block on the diagonal
    matmul(L_node_splits[2], x_node_splits[0], x_node_splits[1], false, false, -1.0, 1.0);
    // Forward substitution with cc and oc blocks below the diagonal
    for (int64_t irow = node+1; irow < num_nodes; ++irow) {
      if (is_admissible.exists(irow, node, level) && !is_admissible(irow, node, level)) {
        auto lower_splits = D(irow, node, level).split(vec{}, vec{diag_col_split});
        matmul(lower_splits[0], x_node_splits[0], x_level_split[irow], false, false, -1.0, 1.0);
      }
    }
    // Forward substitution with oc blocks above the diagonal
    for (int64_t irow = 0; irow < node; ++irow) {
      if (is_admissible.exists(irow, node, level) && !is_admissible(irow, node, level)) {
        const int64_t top_row_split = D(irow, node, level).rows - U(irow, level).cols;
        const int64_t top_col_split = diag_col_split;
        auto top_splits = D(irow, node, level).split(vec{top_row_split}, vec{top_col_split});

        Matrix x_irow(x_level_split[irow], true);  // Deep-copy of view
        auto x_irow_splits = x_irow.split(vec{top_row_split}, vec{});
        matmul(top_splits[2], x_node_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);
        x_level_split[irow] = x_irow;
      }
    }
    // Write x_node
    x_level_split[node] = x_node;
  }
}

void SymmetricH2::solve_backward_level(Matrix& x_level, const int64_t level) const {
  const int64_t num_nodes = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < num_nodes; ++i) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  for (int64_t node = num_nodes-1; node >= 0; --node) {
    const int64_t diag_row_split = D(node, node, level).rows - U(node, level).cols;
    const int64_t diag_col_split = D(node, node, level).cols - U(node, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    Matrix x_node(x_level_split[node], true);
    auto x_node_splits = x_node.split(vec{diag_row_split}, vec{});
    // Backward substitution with co blocks in the left of diagonal
    for (int64_t jcol = node-1; jcol >= 0; --jcol) {
      if (is_admissible.exists(node, jcol, level) && !is_admissible(node, jcol, level)) {
        const int64_t left_row_split = diag_row_split;
        const int64_t left_col_split = D(node, jcol, level).cols - U(jcol, level).cols;
        auto left_splits = D(node, jcol, level).split(vec{left_row_split}, vec{left_col_split});

        Matrix x_jcol(x_level_split[jcol], true);  // Deep-copy of view
        auto x_jcol_splits = x_jcol.split(vec{left_col_split}, vec{});
        matmul(left_splits[1], x_jcol_splits[1], x_node_splits[0], false, false, -1.0, 1.0);
      }
    }
    // Backward substitution with cc and co blocks in the right of diagonal
    for (int64_t jcol = num_nodes-1; jcol > node; --jcol) {
      if (is_admissible.exists(node, jcol, level) && !is_admissible(node, jcol, level)) {
        auto right_splits = D(node, jcol, level).split(vec{diag_row_split}, vec{});
        matmul(right_splits[0], x_level_split[jcol], x_node_splits[0], false, false, -1.0, 1.0);
      }
    }
    // Solve backward with diagonal L
    auto L_node_splits = D(node, node, level).split(vec{diag_row_split}, vec{diag_col_split});
    matmul(L_node_splits[1], x_node_splits[1], x_node_splits[0], false, false, -1.0, 1.0);
    solve_triangular(L_node_splits[0], x_node_splits[0], Hatrix::Left, Hatrix::Lower, true, true);
    // Multiply with U_F
    Matrix U_F = prepend_complement_basis(U(node, level));
    x_node = matmul(U_F, x_node);
    // Write x_node
    x_level_split[node] = x_node;
  }
}

void SymmetricH2::solve_diagonal_level(Matrix& x_level, const int64_t level) const {
  const int64_t num_nodes = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < num_nodes; ++i) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  // Solve diagonal using cc blocks
  for (int64_t node = num_nodes-1; node >= 0; --node) {
    int64_t diag_row_split = D(node, node, level).rows - U(node, level).cols;
    int64_t diag_col_split = D(node, node, level).cols - U(node, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    Matrix x_node(x_level_split[node], true);  // Deep-copy of view
    auto x_node_splits = x_node.split(vec{diag_col_split}, {});
    // Solve with cc block on the diagonal
    auto D_node_splits = D(node, node, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_diagonal(D_node_splits[0], x_node_splits[0], Hatrix::Left);
    x_level_split[node] = x_node;
  }
}

Matrix SymmetricH2::solve(const Matrix& b) const {
  Matrix x(b);
  int64_t level = height;
  int64_t rhs_offset = 0;

  // Forward
  for (; level > 0; --level) {
    const int64_t num_nodes = level_blocks[level];
    int64_t nrows = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
      nrows += D(i, i, level).rows;
    }

    Matrix x_level(nrows, 1);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(rhs_offset + i, 0);
    }
    solve_forward_level(x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
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
  for (; level <= height; ++level) {
    const int64_t num_nodes = level_blocks[level];

    int64_t nrows = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
      nrows += D(i, i, level).cols;
    }
    Matrix x_level(nrows, 1);

    rhs_offset = permute_backward(x, level, rhs_offset);

    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(rhs_offset + i, 0);
    }
    solve_diagonal_level(x_level, level);
    solve_backward_level(x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(rhs_offset + i, 0) = x_level(i, 0);
    }
  }

  return x;
}

double SymmetricH2::solve_error(const Matrix& x, const Matrix& ref) const {
  double diff_norm = 0.;
  double ref_norm = 0.;
  for (int64_t i = 0; i < x.rows; i++) {
    double diff = x(i, 0) - ref(i, 0);
    diff_norm += diff * diff;
    ref_norm += ref(i, 0) * ref(i, 0);
  }
  return std::sqrt(diff_norm / (use_rel_acc ? ref_norm : 1.));
}

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

} // namespace Hatrix

int main(int argc, char ** argv) {
  int64_t N = argc > 1 ? atol(argv[1]) : 256;
  int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-8;
  // Use relative or absolute error threshold for LRA
  const bool use_rel_acc = argc > 4 ? (atol(argv[4]) == 1) : false;
  // Randomized SVD with bounded rank
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

  const int64_t print_csv_header = argc > 11 ? atol(argv[11]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 12 ? std::string(argv[12]) : "";
  const int64_t sort_bodies = argc > 13 ? atol(argv[13]) : 0;

  Hatrix::Context::init();

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
  // Pre-processing step for ELSES geometry
  if (geom_type == 3) {
    assert(file_name.length() > 0);
    domain.read_bodies_ELSES(file_name + ".xyz");
    assert(N == domain.N);

    if (sort_bodies) {
      domain.sort_bodies_ELSES();
      geom_name = file_name + "_sorted";
    }
    domain.build_tree_from_sorted_bodies(leaf_size, std::vector<int64_t>(N / leaf_size, leaf_size));
    domain.read_p2p_matrix_ELSES(file_name + ".dat");
  }
  else {
    domain.build_tree(leaf_size);
  }
  domain.build_interactions(admis);
  domain.build_sample_bodies(N, N, N, 0, geom_type == 3);  // No sampling, use all bodies

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  const auto construct_min_rank = A.get_basis_min_rank(1, A.height);
  const auto construct_max_rank = A.get_basis_max_rank(1, A.height);
  const auto construct_error = A.construction_error(domain);
  const auto construct_min_rank_leaf = A.get_basis_min_rank(A.height, A.height);
  const auto construct_max_rank_leaf = A.get_basis_max_rank(A.height, A.height);
  const auto csp_all = A.get_level_max_nblocks('a', 1, A.height);
  const auto csp_dense = A.get_level_max_nblocks('n', 1, A.height);
  const auto csp_lowrank = A.get_level_max_nblocks('f', 1, A.height);

#ifndef OUTPUT_CSV
  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " acc_type=" << (use_rel_acc ? "rel_err" : "abs_err")
            << " max_rank=" << max_rank
            << " LRA="
#ifdef USE_QR_COMPRESSION
            << "QR"
#else
            << (max_rank > 0 ? "RandSVD" : "SVD")
#endif
            << " admis=" << admis << std::setw(3)
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " height=" << A.height
            << " construct_min_rank=" << construct_min_rank
            << " construct_max_rank=" << construct_max_rank
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error << std::defaultfloat
            << std::endl
            << "construct_min_rank_leaf=" << construct_min_rank_leaf
            << " construct_max_rank_leaf=" << construct_max_rank_leaf
            << " csp_all=" << csp_all
            << " csp_dense=" << csp_dense
            << " csp_lowrank=" << csp_lowrank
            << std::endl;
#endif

  const auto start_factor = std::chrono::system_clock::now();
  const auto factor_fp_ops = A.factorize(domain);
  const auto stop_factor = std::chrono::system_clock::now();
  const double factor_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_factor - start_factor).count();
  const auto factor_min_rank = A.get_basis_min_rank(1, A.height);
  const auto factor_max_rank = A.get_basis_max_rank(1, A.height);
#ifndef OUTPUT_CSV
  std::cout << "factor_min_rank=" << factor_min_rank
            << " factor_max_rank=" << factor_max_rank
            << " factor_fp_ops=" << factor_fp_ops
            << " factor_time=" << factor_time
            << std::endl;
#endif
  /*
  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  Hatrix::Matrix x = Hatrix::body_neutral_charge(domain, 1, 0);
  Hatrix::Matrix b = Hatrix::matmul(Adense, x);
  const auto solve_start = std::chrono::system_clock::now();
  Hatrix::Matrix x_solve = A.solve(b);
  const auto solve_stop = std::chrono::system_clock::now();
  const double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (solve_stop - solve_start).count();
  const auto solve_error = A.solve_error(x_solve, x);
  */
  const int64_t solve_time = 0;
  const double solve_error = -1;
#ifndef OUTPUT_CSV
  std::cout << "solve_time=" << solve_time
            << " solve_error=" << std::scientific << solve_error
            << std::defaultfloat << std::endl;
#endif

#ifdef OUTPUT_CSV
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,leaf_size,accuracy,acc_type,max_rank,LRA,admis,matrix_type,kernel,geometry"
              << ",height,construct_min_rank,construct_max_rank,construct_time,construct_error"
              << ",construct_min_rank_leaf,construct_max_rank_leaf,csp_all,csp_dense,csp_lowrank"
              << ",factor_min_rank,factor_max_rank,factor_fp_ops,factor_time"
              << ",solve_time,solve_error"
              << std::endl;
  }
  std::cout << N
            << "," << leaf_size
            << "," << accuracy
            << "," << (use_rel_acc ? "rel_err" : "abs_err")
            << "," << max_rank
            << ","
#ifdef USE_QR_COMPRESSION
            << "QR"
#else
            << "SVD"
#endif
            << "," << admis
            << "," << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << "," << kernel_name
            << "," << geom_name
            << "," << A.height
            << "," << construct_min_rank
            << "," << construct_max_rank
            << "," << construct_time
            << "," << std::scientific << construct_error << std::defaultfloat
            << "," << construct_min_rank_leaf
            << "," << construct_max_rank_leaf
            << "," << csp_all
            << "," << csp_dense
            << "," << csp_lowrank
            << "," << factor_min_rank
            << "," << factor_max_rank
            << "," << factor_fp_ops
            << "," << factor_time
            << "," << solve_time
            << "," << std::scientific << solve_error << std::defaultfloat
            << std::endl;
#endif

  Hatrix::Context::finalize();
  return 0;
}
