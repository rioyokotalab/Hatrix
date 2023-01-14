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

#include "nlohmann/json.hpp"

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

constexpr double EPS = std::numeric_limits<double>::epsilon();
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

Hatrix::Matrix diag(const Hatrix::Matrix& A) {
  Hatrix::Matrix diag(A.min_dim(), 1);
  for(int64_t i = 0; i < A.min_dim(); i++) {
    diag(i, 0) = A(i, i);
  }
  return diag;
}

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
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowLevelMap US_row;
  std::vector<int64_t> level_blocks;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t find_all_dense_row() const;
  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;

  std::tuple<Matrix, Matrix, Matrix, int64_t> svd_like_compression(Matrix& A) const;

  Matrix generate_block_row(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level) const;
  std::tuple<Matrix, Matrix>
  generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level) const;
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Ubig_child1, const Matrix& Ubig_child2,
                             const int64_t node, const int64_t level) const;
  RowLevelMap
  generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                             RowLevelMap& Uchild);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;
  void fill_JSON(const Domain& domain,
                 const int64_t i, const int64_t j,
                 const int64_t level, nlohmann::json& json) const;

  void update_row_cluster_bases(const int64_t row, const int64_t level,
                                const RowColLevelMap<Matrix>& F,
                                RowMap<Matrix>& r);
  void factorize_level(const int64_t level, const int64_t num_nodes,
                       RowColLevelMap<Matrix>& F, RowMap<Matrix>& r);

 public:
  SymmetricH2(const Domain& domain, const Matrix& rand,
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
  void write_JSON(const Domain& domain, const std::string filename) const;

  void factorize();
  std::tuple<int64_t, int64_t, int64_t> inertia(const double lambda, bool &singular) const;
  std::tuple<double, int64_t, int64_t, double>
  get_mth_eigenvalue(const int64_t m, const double ev_tol,
                     double left, double right) const;
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
    // Subdivide into BLR
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

int64_t SymmetricH2::find_all_dense_row() const {
  const int64_t num_nodes = level_blocks[height];
  for (int64_t i = 0; i < num_nodes; i++) {
    bool all_dense_row = true;
    for (int64_t j = 0; j < num_nodes; j++) {
      if (!is_admissible.exists(i, j, height) ||
          (is_admissible.exists(i, j, height) && is_admissible(i, j, height))) {
        all_dense_row = false;
      }
    }
    if (all_dense_row) {
      return i;
    }
  }
  return -1;
}

int64_t SymmetricH2::get_block_size(const Domain& domain,
                                    const int64_t node, const int64_t level) const {
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
  std::tie(Ui, Si, Vi, rank) = error_svd(A, accuracy, use_rel_acc, false);
#endif

  // Fixed-rank or fixed-accuracy with bounded rank
  rank = max_rank > 0 ? std::min(max_rank, rank) : rank;

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi), std::move(rank));
}

Matrix SymmetricH2::generate_block_row(const Domain& domain, const Matrix& rand,
                                       const int64_t node, const int64_t level) const {
  const int64_t num_nodes = level_blocks[level];
  const int64_t block_size = get_block_size(domain, node, level);
  const bool sample = (rand.cols > 0);
  std::vector<Matrix> rand_splits;
  if (sample) {
    int64_t count = 0;
    std::vector<int64_t> row_splits;
    for (int64_t j = 0; j < num_nodes; j++) {
      count += get_block_size(domain, j, level);
      if (j < (num_nodes - 1)) {
        row_splits.push_back(count);
      }
    }
    rand_splits = rand.split(row_splits, vec{});
  }

  Matrix block_row(block_size, sample ? rand.cols : 0);
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  for (int64_t j = 0; j < num_nodes; j++) {
    if ((!is_admissible.exists(node, j, level)) || // part of upper level admissible block
        (is_admissible.exists(node, j, level) && is_admissible(node, j, level))) {
      if (sample) {
        matmul(generate_p2p_matrix(domain, node, j, node_level), rand_splits[j],
               block_row, false, false, 1.0, 1.0);
      }
      else {
        block_row =
            concat(block_row, generate_p2p_matrix(domain, node, j, node_level), 1);
      }
    }
  }
  return block_row;
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                                        const int64_t node, const int64_t level) const {
  Matrix block_row = generate_block_row(domain, rand, node, level);
  Matrix Ui, Si, Vi_T;
  int64_t rank;
  std::tie(Ui, Si, Vi_T, rank) = svd_like_compression(block_row);

  Matrix UxS = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);
  return std::make_tuple(std::move(Ui), std::move(UxS));
}

void SymmetricH2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
  const int64_t num_nodes = level_blocks[height];
  const auto leaf_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j = 0; j < num_nodes; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_matrix(domain, i, j, leaf_level));
      }
    }
  }
  // Generate leaf level U
  for (int64_t i = 0; i < num_nodes; i++) {
    if (row_has_admissible_blocks(i, height)) {
      Matrix Ui, UxS;
      std::tie(Ui, UxS) =
          generate_row_cluster_basis(domain, rand, i, height);
      U.insert(i, height, std::move(Ui));
      US_row.insert(i, height, std::move(UxS));
    }
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < num_nodes; i++) {
    for (int64_t j = 0; j < num_nodes; j++) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_matrix(domain, i, j, leaf_level);
        S.insert(i, j, height,
                 matmul(matmul(U(i, height), dense, true, false),
                        U(j, height)));
      }
    }
  }
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
                                        const Matrix& Ubig_child1, const Matrix& Ubig_child2,
                                        const int64_t node, const int64_t level) const {
  Matrix block_row = generate_block_row(domain, rand, node, level);
  auto block_row_splits = block_row.split(2, 1);

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
SymmetricH2::generate_transfer_matrices(const Domain& domain, const Matrix& rand,
                                        const int64_t level, RowLevelMap& Uchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;

  const int64_t num_nodes = level_blocks[level];
  for (int64_t node = 0; node < num_nodes; node++) {
    const int64_t block_size = get_block_size(domain, node, level);
    const int64_t child1 = node * 2;
    const int64_t child2 = node * 2 + 1;
    const int64_t child_level = level + 1;

    if (level > 0 && row_has_admissible_blocks(node, level)) {
      // Generate row cluster transfer matrix.
      const Matrix& Ubig_child1 = Uchild(child1, child_level);
      const Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix Utransfer, UxS;
      std::tie(Utransfer, UxS) =
          generate_U_transfer_matrix(domain, rand, Ubig_child1, Ubig_child2, node, level);
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
    for (int64_t j = 0; j < num_nodes; j++) {
      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
        Matrix D = generate_p2p_matrix(domain, i, j, level);

        S.insert(i, j, level, matmul(matmul(Ubig_parent(i, level), D, true, false),
                                     Ubig_parent(j, level)));
      }
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

SymmetricH2::SymmetricH2(const Domain& domain, const Matrix& rand,
                         const int64_t N, const int64_t leaf_size,
                         const double accuracy, const bool use_rel_acc,
                         const int64_t max_rank, const double admis,
                         const int64_t matrix_type)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis), matrix_type(matrix_type) {
  initialize_geometry_admissibility(domain);
  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;

  for (int64_t level = height - 1; level > 0; level--) {
    Uchild = generate_transfer_matrices(domain, rand, level, Uchild);
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
                                           const RowColLevelMap<Matrix>& F,
                                           RowMap<Matrix>& r) {
  const int64_t num_nodes = level_blocks[level];
  const int64_t block_size = D(row, row, level).rows;
  Matrix block_row(block_size, 0);

  // TODO implement a more accurate variant from MiaoMiaoMa2019_UMV paper (Algorithm 1)
  // instead of using a pre-computed UxS from construction phase
  block_row = concat(block_row, US_row(row, level), 1);

  // Concat fill-in blocks
  for (int64_t j = 0; j < num_nodes; j++) {
    if (F.exists(row, j, level)) {
      block_row = concat(block_row, F(row, j, level), 1);
    }
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

void SymmetricH2::factorize_level(const int64_t level, const int64_t num_nodes,
                                  RowColLevelMap<Matrix>& F, RowMap<Matrix>& r) {
  const int64_t parent_level = level - 1;
  for (int64_t node = 0; node < num_nodes; node++) {
    const int64_t parent_node = node / 2;
    // Check for fill-ins along row
    bool found_row_fill_in = false;
    for (int64_t j = 0; j < num_nodes; ++j) {
      if (F.exists(node, j, level)) {
        found_row_fill_in = true;
        break;
      }
    }
    // Update cluster bases if necessary
    if (found_row_fill_in) {
      update_row_cluster_bases(node, level, F, r);
      // Project admissible blocks accordingly
      // Current level: update coupling matrix
      for (int j = 0; j < num_nodes; ++j) {
        if (is_admissible.exists(node, j, level) && is_admissible(node, j, level)) {
          S(node, j, level) = matmul(r(node), S(node, j, level));
        }
      }
      for (int i = 0; i < num_nodes; ++i) {
        if (is_admissible.exists(i, node, level) && is_admissible(i, node, level)) {
          S(i, node, level) = matmul(S(i, node, level), r(node), false, true);
        }
      }
      // Upper levels: update transfer matrix one level higher
      // also the pre-computed US_row
      if (parent_level > 0 && row_has_admissible_blocks(parent_node, parent_level)) {
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
    for (int j = 0; j < num_nodes; ++j) {
      if (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) {
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
    }
    // Multiply to dense blocks along the column in current level
    for (int i = 0; i < num_nodes; ++i) {
      if (is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) {
        if (i < node) {
          // Do not touch the eliminated part (cc and co)
          int64_t up_row_split = D(i, node, level).rows - U(i, level).cols;
          auto D_splits = D(i, node, level).split(vec{up_row_split}, vec{});
          D_splits[1] = matmul(D_splits[1], U_F);
        }
        else {
          D(i, node, level) = matmul(D(i, node, level), U_F);
        }
      }
    }

    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(node, node, level).rows - U(node, level).cols;
    int64_t diag_col_split = D(node, node, level).cols - U(node, level).cols;
    auto diagonal_splits = D(node, node, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = node+1; i < num_nodes; ++i) {
      if (is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) {
        int64_t lower_row_split = D(i, node, level).rows -
                                  (level == height ? U(i, level).cols : U(i * 2, level + 1).cols);
        auto D_splits = D(i, node, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < num_nodes; ++i) {
      if (is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) {
        int64_t lower_row_split = D(i, node, level).rows -
                                  (i <= node || level == height ?
                                   U(i, level).cols :
                                   U(i * 2, level + 1).cols);
        auto D_splits = D(i, node, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = node + 1; j < num_nodes; ++j) {
      if (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) {
        int64_t right_col_split = D(node, j, level).cols -
                                  (level == height ? U(j, level).cols : U(j * 2, level + 1).cols);
        auto D_splits = D(node, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }
    // TRSM with co blocks on this row
    for (int64_t j = 0; j < num_nodes; ++j) {
      if (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) {
        int64_t right_col_split = D(node, j, level).cols -
                                  (j <= node || level == height ?
                                   U(j, level).cols :
                                   U(j * 2, level + 1).cols);
        auto D_splits = D(node, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[1], Hatrix::Left);
      }
    }

    // Schur's complement into dense block
    // cc x cc -> cc
    for (int64_t i = node+1; i < num_nodes; ++i) {
      for (int64_t j = node+1; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower0_scaled(lower_splits[0], true);
          column_scale(lower0_scaled, Dcc);
          // Update cc part
          matmul(lower0_scaled, right_splits[0], reduce_splits[0],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // cc x co -> co
    for (int64_t i = node+1; i < num_nodes; ++i) {
      for (int64_t j = 0; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= node || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower0_scaled(lower_splits[0], true);
          column_scale(lower0_scaled, Dcc);
          // Update co part
          matmul(lower0_scaled, right_splits[1], reduce_splits[1],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // oc x cc -> oc
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t j = node+1; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= node || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);
          // Update oc part
          matmul(lower2_scaled, right_splits[0], reduce_splits[2],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // oc x co -> oo
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t j = 0; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= node || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= node || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);
          // Update oo part
          matmul(lower2_scaled, right_splits[1], reduce_splits[3],
                 false, false, -1.0, 1.0);
        }
      }
    }

    // Schur's complement into low-rank block (fill-in)
    // Produces b*b fill-in
    for (int64_t i = node+1; i < num_nodes; ++i) {
      for (int64_t j = node+1; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*b fill-in block
            int64_t nrows = D(i, node, level).rows;
            int64_t ncols = D(node, j, level).cols;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{nrows - lower_row_rank},
                                                vec{ncols - right_col_rank});
            // Fill cc part
            matmul(lower0_scaled, right_splits[0], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill co part
            matmul(lower0_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);
            // Fill oc part
            matmul(lower2_scaled, right_splits[0], fill_in_splits[2],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[3],
                   false, false, -1.0, 1.0);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, node, level).rows);
              assert(F(i, j, level).cols == D(node, j, level).cols);
              F(i, j, level) += fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces b*rank fill-in
    for (int64_t i = node+1; i < num_nodes; ++i) {
      for (int64_t j = 0; j < node; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*rank fill-in block
            int64_t nrows = D(i, node, level).rows;
            int64_t ncols = right_col_rank;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{nrows - lower_row_rank},
                                                vec{});
            // Fill co part
            matmul(lower0_scaled, right_splits[1], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);

            // b*rank fill-in always has a form of Aik*Vk_c * inv(Akk_cc) x (Uk_c)^T*Akj*Vj_o
            // Convert to b*b block by applying (Vj_o)^T from right
            // Which is safe from bases update since j has been eliminated before (j < k)
            Matrix projected_fill_in = matmul(fill_in, U(j, level), false, true);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, node, level).rows);
              assert(F(i, j, level).cols == D(node, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*b fill-in
    for (int64_t i = 0; i < node; ++i) {
      for (int64_t j = node+1; j < num_nodes; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create rank*b fill-in block
            int64_t nrows = lower_row_rank;
            int64_t ncols = D(node, j, level).cols;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{},
                                                vec{ncols - right_col_rank});
            // Fill oc part
            matmul(lower2_scaled, right_splits[0], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);

            // rank*b fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj
            // Convert to b*b block by applying Ui_o from left
            // Which is safe from bases update since i has been eliminated before (i < k)
            Matrix projected_fill_in = matmul(U(i, level), fill_in);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, node, level).rows);
              assert(F(i, j, level).cols == D(node, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*rank fill-in
    for (int64_t i = 0; i < node; ++i) {
      for (int64_t j = 0; j < node; ++j) {
        if ((is_admissible.exists(i, node, level) && !is_admissible(i, node, level)) &&
            (is_admissible.exists(node, j, level) && !is_admissible(node, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, node, level).split(
              vec{D(i, node, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(node, j, level).split(
              vec{diag_row_split}, vec{D(node, j, level).cols - right_col_rank});
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create rank*rank fill-in block
            int64_t nrows = lower_row_rank;
            int64_t ncols = right_col_rank;
            Matrix fill_in(nrows, ncols);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in,
                   false, false, -1.0, 1.0);

            // rank*rank fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj*Vj_o
            // Convert to b*b block by applying Ui_o from left and (Vj_o)^T from right
            // Which is safe from bases update since i and j have been eliminated before (i,j < k)
            Matrix projected_fill_in = matmul(matmul(U(i, level), fill_in),
                                              U(j, level), false, true);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, node, level).rows);
              assert(F(i, j, level).cols == D(node, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
  } // for (int node = 0; node < num_nodes; ++node)
}

void SymmetricH2::factorize() {
  int64_t level = height;
  RowColLevelMap<Matrix> F;

  for (; level > 0; --level) {
    RowMap<Matrix> r;
    const int64_t num_nodes = level_blocks[level];
    // Make sure all cluster bases exist and none of them is full-rank
    for (int64_t i = 0; i < num_nodes; ++i) {
      if (!U.exists(i, level)) {
        throw std::logic_error("Cluster bases not found at U(" + std::to_string(i) +
                               "," + std::to_string(level) + ")");
      }
      // if (U(i, level).rows <= U(i, level).cols) {
      //   throw std::domain_error("Full rank cluster bases found at U(" + std::to_string(i) +
      //                           "," + std::to_string(level) + ")");
      // }
    }

    factorize_level(level, num_nodes, F, r);

    // Update coupling matrices of admissible blocks in the current level
    // To add fill-in contributions
    for (int64_t i = 0; i < num_nodes; ++i) {
      for (int64_t j = 0; j < num_nodes; ++j) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          if (F.exists(i, j, level)) {
            Matrix projected_fill_in = matmul(matmul(U(i, level), F(i, j, level), true),
                                              U(j, level));
            S(i, j, level) += projected_fill_in;
          }
        }
      }
    }

    const int64_t parent_level = level - 1;
    const int64_t parent_num_nodes = level_blocks[parent_level];
    // Propagate fill-in to upper level admissible blocks (if any)
    if (parent_level > 0) {
      for (int64_t i = 0; i < parent_num_nodes; ++i) {
        for (int64_t j = 0; j < parent_num_nodes; ++j) {
          if ((!is_admissible.exists(i, j, parent_level)) ||
              (is_admissible.exists(i, j, parent_level) && is_admissible(i, j, parent_level))) {
            int64_t i1 = i * 2;
            int64_t i2 = i * 2 + 1;
            int64_t j1 = j * 2;
            int64_t j2 = j * 2 + 1;
            if (F.exists(i1, j1, level) || F.exists(i1, j2, level) ||
                F.exists(i2, j1, level) || F.exists(i2, j2, level)) {
              int64_t nrows = U(i1, level).cols + U(i2, level).cols;
              int64_t ncols = U(j1, level).cols + U(j2, level).cols;
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
      for (int64_t j = 0; j < parent_num_nodes; ++j) {
        if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
          // TODO: need to switch to morton indexing so finding the parent is straightforward.
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
    }
  } // for (; level > 0; --level)
  F.erase_all();

  // Factorize remaining root level
  ldl(D(0, 0, level));
}

std::tuple<int64_t, int64_t, int64_t>
SymmetricH2::inertia(const double lambda, bool &singular) const {
  SymmetricH2 A_shifted(*this);
  // Shift leaf level diagonal blocks
  int64_t leaf_num_nodes = level_blocks[height];
  for(int64_t node = 0; node < leaf_num_nodes; node++) {
    shift_diag(A_shifted.D(node, node, height), -lambda);
  }
  // LDL Factorize
  A_shifted.factorize();
  // Gather values in D
  Matrix D_lambda(0, 0);
  for(int64_t level = height; level >= 0; level--) {
    int64_t num_nodes = level_blocks[level];
    for(int64_t node = 0; node < num_nodes; node++) {
      const Matrix& D_node = A_shifted.D(node, node, level);
      if(level == 0) {
        D_lambda = concat(D_lambda, diag(D_node), 0);
      }
      else {
        const int64_t rank = A_shifted.U(node, level).cols;
        const int64_t row_split = D_node.rows - rank;
        const int64_t col_split = D_node.cols - rank;
        auto D_node_splits = D_node.split(vec{row_split}, vec{col_split});
        Matrix& D_node_cc = D_node_splits[0];
        D_lambda = concat(D_lambda, diag(D_node_cc), 0);
      }
    }
  }
  int64_t negative_elements_count = 0;
  for(int64_t i = 0; i < D_lambda.rows; i++) {
    negative_elements_count += (D_lambda(i, 0) < 0 ? 1 : 0);
    if(std::isnan(D_lambda(i, 0)) || std::abs(D_lambda(i, 0)) < EPS) singular = true;
  }
  const auto ldl_min_rank = A_shifted.get_basis_min_rank();
  const auto ldl_max_rank = A_shifted.get_basis_max_rank();
  return {negative_elements_count, ldl_min_rank, ldl_max_rank};
}

std::tuple<double, int64_t, int64_t, double>
SymmetricH2::get_mth_eigenvalue(const int64_t m, const double ev_tol,
                                double left, double right) const {
  int64_t shift_min_rank = get_basis_min_rank();
  int64_t shift_max_rank = get_basis_max_rank();
  double max_rank_shift = -1;
  bool singular = false;
  while((right - left) >= ev_tol) {
    const auto mid = (left + right) / 2;
    int64_t value, factor_min_rank, factor_max_rank;
    std::tie(value, factor_min_rank, factor_max_rank) = (*this).inertia(mid, singular);
    if(factor_max_rank >= shift_max_rank) {
      shift_min_rank = factor_min_rank;
      shift_max_rank = factor_max_rank;
      max_rank_shift = mid;
    }
    if(singular) {
      std::cout << "Shifted matrix became singular (shift=" << mid << ")" << std::endl;
      break;
    }
    if(value >= m) right = mid;
    else left = mid;
  }
  return {(left + right) / 2, shift_min_rank, shift_max_rank, max_rank_shift};
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
  // Multiplication with random matrix instead of sample points
  const int64_t random_matrix_size = argc > 11 ? atol(argv[11]) : 100;
    // Eigenvalue computation parameters
  const double ev_tol = argc > 12 ? atof(argv[12]) : 1.e-3;
  int64_t m_begin = argc > 13 ? atol(argv[13]) : 1;
  int64_t m_end = argc > 14 ? atol(argv[14]) : m_begin;
  const bool compute_eig_acc = argc > 15 ? (atol(argv[15]) == 1) : false;
  const int64_t print_csv_header = argc > 16 ? atol(argv[16]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 17 ? std::string(argv[17]) : "";
  const int64_t read_sorted_bodies = argc > 18 ? atol(argv[18]) : 0;

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1.e-3 / (double)N, 1.);
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
  const bool is_non_synthetic = (geom_type == 3);
  if (is_non_synthetic) {
    if (read_sorted_bodies == 1) {
      geom_name = file_name + "_" + std::to_string(leaf_size);
      const auto buckets = domain.read_bodies_ELSES_sorted(geom_name + ".xyz");
      domain.build_tree_from_kmeans_ordering(leaf_size, buckets);
      leaf_size = *(std::max_element(buckets.begin(), buckets.end()));
    }
    else {
      domain.read_bodies_ELSES(file_name + ".xyz");
      domain.build_tree(leaf_size);
    }
    N = domain.N;
    domain.read_p2p_matrix_ELSES(file_name + ".dat");
  }
  else {
    domain.build_tree(leaf_size);
  }
  domain.build_interactions(admis);

  Hatrix::Matrix rand = Hatrix::generate_random_matrix(N, random_matrix_size);
  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, rand, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  const auto construct_min_rank = A.get_basis_min_rank();
  const auto construct_max_rank = A.get_basis_max_rank();
  const auto construct_error = A.construction_error(domain);
  const auto lr_ratio = A.low_rank_block_ratio();

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
            << "SVD"
#endif
            << " admis=" << admis << std::setw(3)
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " random_matrix_size=" << random_matrix_size
            << " height=" << A.height
            << " lr_ratio=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << construct_min_rank
            << " construct_max_rank=" << construct_max_rank
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
#ifndef OUTPUT_CSV
  std::cout << "dense_eig_time=" << dense_eig_time
            << std::endl;
#endif

  bool s = false;
  auto b = is_non_synthetic ?
           Hatrix::norm(Hatrix::generate_p2p_matrix(domain)) : 10 * (1. / Hatrix::PV);
  auto a = -b;
  int64_t v_a, v_b, temp1, temp2;
  std::tie(v_a, temp1, temp2) = A.inertia(a, s);
  std::tie(v_b, temp1, temp2) = A.inertia(b, s);
  if(v_a != 0 || v_b != N) {
    std::cerr << "Warning: starting interval does not contain the whole spectrum" << std::endl
              << "at N=" << N << ",leaf_size=" << leaf_size << ",accuracy=" << accuracy
              << ",admis=" << admis << ",b=" << b << std::endl;
    a *= 2;
    b *= 2;
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
#ifdef OUTPUT_CSV
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,leaf_size,accuracy,acc_type,max_rank,LRA,admis,matrix_type,kernel,geometry,random_matrix_size"
              << ",height,lr_ratio,construct_min_rank,construct_max_rank,construct_time,construct_error"
              << ",dense_eig_time"
              << ",m,a0,b0,ev_tol,h2_eig_time,ldl_min_rank,ldl_max_rank,max_rank_shift,dense_eigv,h2_eigv,eig_abs_err,success"
              << std::endl;
  }
#endif
  for (int64_t k = 0; k < target_m.size(); k++) {
    const int64_t m = target_m[k];
    double h2_mth_eigv, max_rank_shift;
    int64_t ldl_min_rank, ldl_max_rank;
    const auto h2_eig_start = std::chrono::system_clock::now();
    std::tie(h2_mth_eigv, ldl_min_rank, ldl_max_rank, max_rank_shift) =
        A.get_mth_eigenvalue(m, ev_tol, a, b);
    const auto h2_eig_stop = std::chrono::system_clock::now();
    const double h2_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                               (h2_eig_stop - h2_eig_start).count();
    const double dense_mth_eigv = compute_eig_acc ? dense_eigv[m - 1] : -1;
    const double eig_abs_err = compute_eig_acc ? std::abs(h2_mth_eigv - dense_mth_eigv) : -1;
    const bool success = compute_eig_acc ? (eig_abs_err < (0.5 * ev_tol)) : true;
#ifndef OUTPUT_CSV
    std::cout << "m=" << m
              << " a0=" << a
              << " b0=" << b
              << " ev_tol=" << ev_tol
              << " h2_eig_time=" << h2_eig_time
              << " ldl_min_rank=" << ldl_min_rank
              << " ldl_max_rank=" << ldl_max_rank
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
#ifdef USE_QR_COMPRESSION
              << "QR"
#else
              << "SVD"
#endif
              << "," << admis
              << "," << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
              << "," << kernel_name
              << "," << geom_name
              << "," << random_matrix_size
              << "," << A.height
              << "," << lr_ratio
              << "," << construct_min_rank
              << "," << construct_max_rank
              << "," << construct_time
              << "," << std::scientific << construct_error << std::defaultfloat
              << "," << dense_eig_time
              << "," << m
              << "," << a
              << "," << b
              << "," << ev_tol
              << "," << h2_eig_time
              << "," << ldl_min_rank
              << "," << ldl_max_rank
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

