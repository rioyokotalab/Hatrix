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

// #include "Hatrix/Hatrix.hpp"
// #include "Domain.hpp"
#include "functions.hpp"

Hatrix::Matrix generate_random_matrix(int64_t rows, int64_t cols) {
  std::mt19937 gen(100);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Hatrix::Matrix out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}


using vec = std::vector<int64_t>;

// H2-Construction employ multiplication with a random matrix to reduce far-block matrix size
// Quite accurate and does not rely on ID but incur O(N^2) complexity to construct basis and coupling matrices

// Comment the following line to use SVD instead of pivoted QR for low-rank compression
// #define USE_QR_COMPRESSION

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

class H2 {
 public:
  int64_t N, leaf_size;
  double accuracy;
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  ColLevelMap V;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowLevelMap US_row;
  ColLevelMap SV_col;
  std::vector<int64_t> level_blocks;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t find_all_dense_row() const;
  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;
  bool col_has_admissible_blocks(const int64_t col, const int64_t level) const;

  std::tuple<Matrix, Matrix, Matrix, int64_t> svd_like_compression(Matrix& A) const;

  Matrix generate_block_row(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level) const;
  Matrix generate_block_col(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level) const;
  std::tuple<Matrix, Matrix>
  generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level) const;
  std::tuple<Matrix, Matrix>
  generate_col_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level) const;
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Ubig_child1, const Matrix& Ubig_child2,
                             const int64_t node, const int64_t level) const;
  std::tuple<Matrix, Matrix>
  generate_V_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Vbig_child1, const Matrix& Vbig_child2,
                             const int64_t node, const int64_t level) const;
  std::tuple<RowLevelMap, ColLevelMap>
  generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                             RowLevelMap& Uchild, ColLevelMap& Vchild);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;
  Matrix get_Vbig(const int64_t node, const int64_t level) const;

 public:
  H2(const Domain& domain, const Matrix& rand,
     const int64_t N, const int64_t leaf_size,
     const double accuracy, const int64_t max_rank,
     const double admis, const int64_t matrix_type);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double construction_absolute_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;
};

void H2::initialize_geometry_admissibility(const Domain& domain) {
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

int64_t H2::find_all_dense_row() const {
  const int64_t nblocks = level_blocks[height];
  for (int64_t i = 0; i < nblocks; i++) {
    bool all_dense_row = true;
    for (int64_t j = 0; j < nblocks; j++) {
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

int64_t H2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  const auto idx = domain.get_cell_idx(node, node_level);
  return domain.cells[idx].nbodies;
}

bool H2::row_has_admissible_blocks(const int64_t row, const int64_t level) const {
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

bool H2::col_has_admissible_blocks(const int64_t col, const int64_t level) const {
  bool has_admis = false;
  for (int64_t i = 0; i < level_blocks[level]; i++) {
    if ((!is_admissible.exists(i, col, level)) || // part of upper level admissible block
        (is_admissible.exists(i, col, level) && is_admissible(i, col, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

std::tuple<Matrix, Matrix, Matrix, int64_t> H2::svd_like_compression(Matrix& A) const {
  Matrix Ui, Si, Vi;
  int64_t rank;
#ifdef USE_QR_COMPRESSION
  Matrix R;
  std::tie(Ui, R, rank) = error_pivoted_qr(A, accuracy, false, false);
  Si = Matrix(R.rows, R.rows);
  Vi = Matrix(R.rows, R.cols);
  rq(R, Si, Vi);
#else
  std::tie(Ui, Si, Vi, rank) = error_svd(A, accuracy, false, false);
#endif

  // Fixed-rank or fixed-accuracy with bounded rank
  rank = accuracy == 0. ? max_rank : std::min(max_rank, rank);

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi), std::move(rank));
}

Matrix H2::generate_block_row(const Domain& domain, const Matrix& rand,
                              const int64_t node, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  const int64_t block_size = get_block_size(domain, node, level);
  const bool sample = (rand.cols > 0);
  std::vector<Matrix> rand_splits;
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_row(block_size, sample ? rand.cols : 0);
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  for (int64_t j = 0; j < nblocks; j++) {
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

Matrix H2::generate_block_col(const Domain& domain, const Matrix& rand,
                              const int64_t node, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  const int64_t block_size = get_block_size(domain, node, level);
  const bool sample = (rand.cols > 0);
  std::vector<Matrix> rand_splits;
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_column(sample ? rand.cols : 0, block_size);
  const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
  for (int64_t i = 0; i < nblocks; i++) {
    if ((!is_admissible.exists(i, node, level)) || // part of upper level admissible block
        (is_admissible.exists(i, node, level) && is_admissible(i, node, level))) {
      if (sample) {
        matmul(rand_splits[i],
               generate_p2p_matrix(domain, i, node, node_level),
               block_column, true, false, 1.0, 1.0);
      }
      else {
        block_column =
            concat(block_column, generate_p2p_matrix(domain, i, node, node_level), 0);
      }
    }
  }
  return block_column;
}

std::tuple<Matrix, Matrix>
H2::generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                               const int64_t node, const int64_t level) const {
  Matrix block_row = generate_block_row(domain, rand, node, level);
  Matrix Ui, Si, Vi_T;
  int64_t rank;
  std::tie(Ui, Si, Vi_T, rank) = svd_like_compression(block_row);

  Matrix UxS = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);
  return std::make_tuple(std::move(Ui), std::move(UxS));
}

std::tuple<Matrix, Matrix>
H2::generate_col_cluster_basis(const Domain& domain, const Matrix& rand,
                               const int64_t node, const int64_t level) const {
  Matrix block_column_T = transpose(generate_block_col(domain, rand, node, level));
  Matrix Vj, Sj_T, Uj_T;
  int64_t rank;
  std::tie(Vj, Sj_T, Uj_T, rank) = svd_like_compression(block_column_T);

  Matrix SxV_T = matmul(Sj_T, Vj, true, true);
  Vj.shrink(Vj.rows, rank);
  return std::make_tuple(std::move(Vj), std::move(SxV_T));
}

void H2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
  const int64_t nblocks = level_blocks[height];
  const auto leaf_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_matrix(domain, i, j, leaf_level));
      }
    }
  }
  // Generate leaf level U
  for (int64_t i = 0; i < nblocks; i++) {
    if (row_has_admissible_blocks(i, height)) {
      Matrix Ui, UxS;
      std::tie(Ui, UxS) =
          generate_row_cluster_basis(domain, rand, i, height);
      U.insert(i, height, std::move(Ui));
      US_row.insert(i, height, std::move(UxS));
    }
  }
  // Generate leaf level V
  for (int64_t j = 0; j < nblocks; j++) {
    if (col_has_admissible_blocks(j, height)) {
      Matrix Vj, SxV;
      std::tie(Vj, SxV) =
          generate_col_cluster_basis(domain, rand, j, height);
      V.insert(j, height, std::move(Vj));
      SV_col.insert(j, height, std::move(SxV));
    }
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_matrix(domain, i, j, leaf_level);
        S.insert(i, j, height,
                 matmul(matmul(U(i, height), dense, true, false),
                        V(j, height)));
      }
    }
  }
}

std::tuple<Matrix, Matrix>
H2::generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
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

std::tuple<Matrix, Matrix>
H2::generate_V_transfer_matrix(const Domain& domain, const Matrix& rand,
                               const Matrix& Vbig_child1, const Matrix& Vbig_child2,
                               const int64_t node, const int64_t level) const {
  Matrix block_column_T = transpose(generate_block_col(domain, rand, node, level));
  auto block_column_T_splits = block_column_T.split(2, 1);

  Matrix temp(Vbig_child1.cols + Vbig_child2.cols, block_column_T.cols);
  auto temp_splits = temp.split(vec{Vbig_child1.cols}, vec{});

  matmul(Vbig_child1, block_column_T_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Vbig_child2, block_column_T_splits[1], temp_splits[1], true, false, 1, 0);

  Matrix Vj, Sj_T, Uj_T;
  int64_t rank;
  std::tie(Vj, Sj_T, Uj_T, rank) = svd_like_compression(temp);

  Matrix SxV_T = matmul(Sj_T, Vj, true, true);
  Vj.shrink(Vj.rows, rank);
  return std::make_tuple(std::move(Vj), std::move(SxV_T));
}

std::tuple<RowLevelMap, ColLevelMap>
H2::generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                               RowLevelMap& Uchild, ColLevelMap& Vchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;
  ColLevelMap Vbig_parent;

  const int64_t nblocks = level_blocks[level];
  for (int64_t node = 0; node < nblocks; node++) {
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
    if (level > 0 && col_has_admissible_blocks(node, level)) {
      // Generate column cluster transfer Matrix.
      const Matrix& Vbig_child1 = Vchild(child1, child_level);
      const Matrix& Vbig_child2 = Vchild(child2, child_level);
      Matrix Vtransfer, SxV;
      std::tie(Vtransfer, SxV) =
          generate_V_transfer_matrix(domain, rand, Vbig_child1, Vbig_child2, node, level);
      V.insert(node, level, std::move(Vtransfer));
      SV_col.insert(node, level, std::move(SxV));

      // Generate the full bases to pass onto the parent.
      auto Vtransfer_splits = V(node, level).split(vec{Vbig_child1.cols}, vec{});
      Matrix Vbig(block_size, V(node, level).cols);
      auto Vbig_splits = Vbig.split(vec{Vbig_child1.rows}, vec{});

      matmul(Vbig_child1, Vtransfer_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, Vtransfer_splits[1], Vbig_splits[1]);
      Vbig_parent.insert(node, level, std::move(Vbig));
    }
  }

  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
        Matrix D = generate_p2p_matrix(domain, i, j, level);

        S.insert(i, j, level, matmul(matmul(Ubig_parent(i, level), D, true, false),
                                     Vbig_parent(j, level)));
      }
    }
  }
  return {Ubig_parent, Vbig_parent};
}

Matrix H2::get_Ubig(const int64_t node, const int64_t level) const {
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

Matrix H2::get_Vbig(const int64_t node, const int64_t level) const {
  if (level == height) {
    return V(node, level);
  }

  const int64_t child1 = node * 2;
  const int64_t child2 = node * 2 + 1;
  const Matrix Vbig_child1 = get_Vbig(child1, level + 1);
  const Matrix Vbig_child2 = get_Vbig(child2, level + 1);

  const int64_t block_size = Vbig_child1.rows + Vbig_child2.rows;
  Matrix Vbig(block_size, V(node, level).cols);
  auto Vbig_splits = Vbig.split(vec{Vbig_child1.rows}, vec{});
  auto V_splits = V(node, level).split(vec{Vbig_child1.cols}, vec{});

  matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
  matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);
  return Vbig;
}

H2::H2(const Domain& domain, const Matrix& rand,
       const int64_t N, const int64_t leaf_size,
       const double accuracy, const int64_t max_rank,
       const double admis, const int64_t matrix_type)
    : N(N), leaf_size(leaf_size), accuracy(accuracy), max_rank(max_rank),
      admis(admis), matrix_type(matrix_type) {
  initialize_geometry_admissibility(domain);
  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;
  ColLevelMap Vchild = V;

  for (int64_t level = height - 1; level > 0; level--) {
    std::tie(Uchild, Vchild) = generate_transfer_matrices(domain, rand, level, Uchild, Vchild);
  }
}

int64_t H2::get_basis_min_rank() const {
  int64_t rank_min = N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    for (int64_t node = 0; node < nblocks; node++) {
      if (U.exists(node, level)) {
        rank_min = std::min(rank_min, U(node, level).cols);
      }
      if (V.exists(node, level)) {
        rank_min = std::min(rank_min, V(node, level).cols);
      }
    }
  }
  return rank_min;
}

int64_t H2::get_basis_max_rank() const {
  int64_t rank_max = -N;
  for (int64_t level = height; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    for (int64_t node = 0; node < nblocks; node++) {
      if (U.exists(node, level)) {
        rank_max = std::max(rank_max, U(node, level).cols);
      }
      if (V.exists(node, level)) {
        rank_max = std::max(rank_max, V(node, level).cols);
      }
    }
  }
  return rank_max;
}

double H2::construction_absolute_error(const Domain& domain) const {
  double error = 0;
  // Inadmissible blocks (only at leaf level)
  for (int64_t i = 0; i < level_blocks[height]; i++) {
    for (int64_t j = 0; j < level_blocks[height]; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : height;
        const Matrix actual = Hatrix::generate_p2p_matrix(domain, i, j, node_level);
        const Matrix expected = D(i, j, height);
        error += pow(norm(actual - expected), 2);
      }
    }
  }
  // Admissible blocks
  for (int64_t level = height; level > 0; level--) {
    for (int64_t i = 0; i < level_blocks[level]; i++) {
      for (int64_t j = 0; j < level_blocks[level]; j++) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const Matrix Ubig = get_Ubig(i, level);
          const Matrix Vbig = get_Vbig(j, level);
          const Matrix expected_matrix = matmul(matmul(Ubig, S(i, j, level)), Vbig, false, true);
          const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
          const Matrix actual_matrix =
              Hatrix::generate_p2p_matrix(domain, i, j, node_level);
          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }
  return std::sqrt(error);
}

void H2::print_structure(const int64_t level) const {
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

void H2::print_ranks() const {
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
                << ", col_rank=" << (V.exists(node, level) ?
                                     V(node, level).cols : -1)
                << std::endl;
    }
  }
}

double H2::low_rank_block_ratio() const {
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

} // namespace Hatrix

int main(int argc, char ** argv) {
  if (argc == 1) {
    std::cout << "HELP SCREEN FOR H2_strong_CON.cpp" << std::endl;
    std::cout << "Specify arguments as follows: " << std::endl;
    std::cout << "N leaf_size accuracy max_rank random_matrix_size admis kernel_type geom_type ndim matrix_type" << std::endl;
    return 0;
  }

  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-5;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const int64_t random_matrix_size = argc > 5 ? atol(argv[5]) : 100;
  const double admis = argc > 6 ? atof(argv[6]) : 1.0;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim  = argc > 9 ? atol(argv[9]) : 2;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 10 ? atol(argv[10]) : 1;

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

  const auto start_particles = std::chrono::system_clock::now();
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
  domain.build_tree(leaf_size);
  domain.build_interactions(admis);
  const auto stop_particles = std::chrono::system_clock::now();
  const double particle_construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                         (stop_particles - start_particles).count();

  Hatrix::Matrix rand = generate_random_matrix(N, random_matrix_size);
  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(domain, rand, N, leaf_size, accuracy, max_rank, admis, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error = A.construction_absolute_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " max_rank=" << max_rank
            << " random_matrix_size=" << random_matrix_size
            << " compress_alg="
#ifdef USE_QR_COMPRESSION
            << "QR"
#else
            << "SVD"
#endif
            << " admis=" << admis << std::setw(3)
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " height=" << A.height
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error
            << std::endl;

  return 0;
}
