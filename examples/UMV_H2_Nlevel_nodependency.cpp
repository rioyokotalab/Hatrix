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

// Comment the following line to use SVD instead of pivoted QR for low-rank compression
// #define USE_QR_COMPRESSION

using vec = std::vector<int64_t>;
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};

namespace Hatrix {

class H2 {
 public:
  int64_t N, nleaf, n_blocks;
  double accuracy;
  int64_t max_rank;
  double admis;
  std::string admis_kind;
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
  int64_t find_all_dense_row() const;
  void coarsen_blocks(const int64_t level);

  int64_t geometry_admis_non_leaf(const int64_t nblocks, const int64_t level);
  int64_t calc_geometry_based_admissibility(const Domain& domain);
  void calc_diagonal_based_admissibility(const int64_t level);

  int64_t get_block_size_row(const Domain& domain, const int64_t node, const int64_t level) const;
  int64_t get_block_size_col(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;
  bool col_has_admissible_blocks(const int64_t col, const int64_t level) const;

  std::tuple<Matrix, Matrix, Matrix, int64_t> svd_like_compression(Matrix& A) const;

  Matrix generate_block_row(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level,
                            const int64_t block_size) const;
  Matrix generate_block_col(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level,
                            const int64_t block_size) const;
  std::tuple<Matrix, Matrix>
  generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  std::tuple<Matrix, Matrix>
  generate_col_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Ubig_child1, const Matrix& Ubig_child2,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  std::tuple<Matrix, Matrix>
  generate_V_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Vbig_child1, const Matrix& Vbig_child2,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  std::tuple<RowLevelMap, ColLevelMap>
  generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                             RowLevelMap& Uchild, ColLevelMap& Vchild);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;
  Matrix get_Vbig(const int64_t node, const int64_t level) const;

  void pre_compute_fill_in(const int64_t level, RowColLevelMap<Matrix>& F) const;
  void update_row_cluster_basis(const int64_t level, const RowColLevelMap<Matrix>& F);
  void update_col_cluster_basis(const int64_t level, const RowColLevelMap<Matrix>& F);
  void propagate_upper_level_fill_in(const int64_t level, RowColLevelMap<Matrix>& F) const;
  void factorize_level(const int64_t level);

  int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  void solve_forward_level(Matrix& x_level, const int64_t level) const;
  void solve_backward_level(Matrix& x_level, const int64_t level) const;

 public:
  H2(const Domain& domain, const Matrix& rand,
     const int64_t N, const int64_t nleaf,
     const double accuracy, const int64_t max_rank,
     const double admis, const std::string& admis_kind,
     const int64_t matrix_type);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double construction_absolute_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;

  void factorize();
  Matrix solve(const Matrix& b) const;
};

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

void H2::coarsen_blocks(const int64_t level) {
  const int64_t child_level = level + 1;
  const int64_t nblocks = (int64_t)std::pow(2, level);
  for (int64_t i = 0; i < nblocks; i++) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; j++) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; c1++) {
        for (int64_t c2 = 0; c2 < 2; c2++) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }
      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; c1++) {
          for (int64_t c2 = 0; c2 < 2; c2++) {
            is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }
      is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

int64_t H2::geometry_admis_non_leaf(const int64_t nblocks, const int64_t level) {
  const int64_t child_level = level - 1;
  level_blocks.push_back(nblocks);

  if (nblocks == 1) { return level; }

  for (int64_t i = 0; i < nblocks; i++) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; j++) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; c1++) {
        for (int64_t c2 = 0; c2 < 2; c2++) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }
      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; c1++) {
          for (int64_t c2 = 0; c2 < 2; c2++) {
            is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }
      is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }

  return geometry_admis_non_leaf(nblocks/2, level+1);
}

int64_t H2::calc_geometry_based_admissibility(const Domain& domain) {
  const int64_t nblocks = domain.boxes.size();
  level_blocks.push_back(nblocks);
  const int64_t level = 0;
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      is_admissible.insert(i, j, level, domain.check_admis(admis, i, j));
    }
  }

  if (matrix_type == BLR2_MATRIX) {
    level_blocks.push_back(1);
    return 1;
  }
  else {
    return geometry_admis_non_leaf(nblocks / 2, level+1);
  }
}

void H2::calc_diagonal_based_admissibility(const int64_t level) {
  const int64_t nblocks = (int64_t)std::pow(2, level);
  level_blocks.push_back(nblocks);
  if (level == 0) { return; }
  if (level == height) {
    for (int64_t i = 0; i < nblocks; i++) {
      for (int64_t j = 0; j < nblocks; j++) {
        bool is_admissible_block = std::abs(i - j) > admis;
        is_admissible.insert(i, j, level, std::move(is_admissible_block));
      }
    }
  }
  else {
    coarsen_blocks(level);
  }

  calc_diagonal_based_admissibility(level-1);
}

int64_t H2::get_block_size_row(const Domain& domain, const int64_t node, const int64_t level) const {
  if (level == height) {
    return domain.boxes[node].num_particles;
  }
  const int64_t child_level = level + 1;
  const int64_t child1 = node * 2;
  const int64_t child2 = node * 2 + 1;

  return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
}

int64_t H2::get_block_size_col(const Domain& domain, const int64_t node, const int64_t level) const {
  if (level == height) {
    return domain.boxes[node].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;

  return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
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
                              const int64_t node, const int64_t level,
                              const int64_t block_size) const {
  const int64_t nblocks = level_blocks[level];
  const bool sample = (rand.cols > 0);
  std::vector<Matrix> rand_splits;
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_row(block_size, sample ? rand.cols : 0);
  for (int64_t j = 0; j < nblocks; j++) {
    if ((!is_admissible.exists(node, j, level)) || // part of upper level admissible block
        (is_admissible.exists(node, j, level) && is_admissible(node, j, level))) {
      if (sample) {
        matmul(generate_p2p_interactions(domain, node, j, level, height), rand_splits[j],
               block_row, false, false, 1.0, 1.0);
      }
      else {
        block_row =
            concat(block_row, generate_p2p_interactions(domain, node, j, level, height), 1);
      }
    }
  }
  return block_row;
}

Matrix H2::generate_block_col(const Domain& domain, const Matrix& rand,
                              const int64_t node, const int64_t level,
                              const int64_t block_size) const {
  const int64_t nblocks = level_blocks[level];
  const bool sample = (rand.cols > 0);
  std::vector<Matrix> rand_splits;
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_column(sample ? rand.cols : 0, block_size);
  for (int64_t i = 0; i < nblocks; i++) {
    if ((!is_admissible.exists(i, node, level)) || // part of upper level admissible block
        (is_admissible.exists(i, node, level) && is_admissible(i, node, level))) {
      if (sample) {
        matmul(rand_splits[i],
               generate_p2p_interactions(domain, i, node, level, height),
               block_column, true, false, 1.0, 1.0);
      }
      else {
        block_column =
            concat(block_column, generate_p2p_interactions(domain, i, node, level, height), 0);
      }
    }
  }
  return block_column;
}

std::tuple<Matrix, Matrix>
H2::generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                               const int64_t node, const int64_t level,
                               const int64_t block_size) const {
  Matrix block_row = generate_block_row(domain, rand, node, level, block_size);
  Matrix Ui, Si, Vi_T;
  int64_t rank;
  std::tie(Ui, Si, Vi_T, rank) = svd_like_compression(block_row);

  Matrix UxS = matmul(Ui, Si);
  Ui.shrink(Ui.rows, rank);
  return std::make_tuple(std::move(Ui), std::move(UxS));
}

std::tuple<Matrix, Matrix>
H2::generate_col_cluster_basis(const Domain& domain, const Matrix& rand,
                               const int64_t node, const int64_t level,
                               const int64_t block_size) const {
  Matrix block_column_T = transpose(generate_block_col(domain, rand, node, level, block_size));
  Matrix Vj, Sj_T, Uj_T;
  int64_t rank;
  std::tie(Vj, Sj_T, Uj_T, rank) = svd_like_compression(block_column_T);

  Matrix SxV_T = matmul(Sj_T, Vj, true, true);
  Vj.shrink(Vj.rows, rank);
  return std::make_tuple(std::move(Vj), std::move(SxV_T));
}

void H2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
  const int64_t nblocks = level_blocks[height];
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_interactions(domain, i, j, height, height));
      }
    }
  }
  // Generate leaf level U
  for (int64_t i = 0; i < nblocks; i++) {
    if (row_has_admissible_blocks(i, height)) {
      Matrix Ui, UxS;
      std::tie(Ui, UxS) =
          generate_row_cluster_basis(domain, rand, i, height, domain.boxes[i].num_particles);
      U.insert(i, height, std::move(Ui));
      US_row.insert(i, height, std::move(UxS));
    }
  }
  // Generate leaf level V
  for (int64_t j = 0; j < nblocks; j++) {
    if (col_has_admissible_blocks(j, height)) {
      Matrix Vj, SxV;
      std::tie(Vj, SxV) =
          generate_col_cluster_basis(domain, rand, j, height, domain.boxes[j].num_particles);
      V.insert(j, height, std::move(Vj));
      SV_col.insert(j, height, std::move(SxV));
    }
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_interactions(domain, i, j, height, height);
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
                               const int64_t node, const int64_t level,
                               const int64_t block_size) const {
  Matrix block_row = generate_block_row(domain, rand, node, level, block_size);
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
                               const int64_t node, const int64_t level,
                               const int64_t block_size) const {
  Matrix block_column_T = transpose(generate_block_col(domain, rand, node, level, block_size));
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
    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;
    int64_t child_level = level + 1;

    if (level > 0 && row_has_admissible_blocks(node, level)) {
      const int64_t block_size = get_block_size_row(domain, node, level);
      // Generate row cluster transfer matrix.
      const Matrix& Ubig_child1 = Uchild(child1, child_level);
      const Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix Utransfer, UxS;
      std::tie(Utransfer, UxS) =
          generate_U_transfer_matrix(domain, rand, Ubig_child1, Ubig_child2,
                                     node, level, block_size);
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
      const int64_t block_size = get_block_size_col(domain, node, level);
      // Generate column cluster transfer Matrix.
      const Matrix& Vbig_child1 = Vchild(child1, child_level);
      const Matrix& Vbig_child2 = Vchild(child2, child_level);
      Matrix Vtransfer, SxV;
      std::tie(Vtransfer, SxV) =
          generate_V_transfer_matrix(domain, rand, Vbig_child1, Vbig_child2,
                                     node, level, block_size);
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
        Matrix D = generate_p2p_interactions(domain, i, j, level, height);

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
       const int64_t N, const int64_t nleaf,
       const double accuracy, const int64_t max_rank,
       const double admis, const std::string& admis_kind,
       const int64_t matrix_type)
    : N(N), nleaf(nleaf), accuracy(accuracy), max_rank(max_rank),
      admis(admis), admis_kind(admis_kind), matrix_type(matrix_type) {
  if (admis_kind == "geometry_admis") {
    // TODO: use dual tree traversal for this.
    height = calc_geometry_based_admissibility(domain);
    // reverse the levels stored in the admis blocks.
    RowColLevelMap<bool> temp_is_admissible;

    for (int64_t level = 0; level < height; level++) {
      const int64_t nblocks = level_blocks[level];
      for (int64_t i = 0; i < nblocks; i++) {
        for (int64_t j = 0; j < nblocks; j++) {
          if (is_admissible.exists(i, j, level)) {
            bool value = is_admissible(i, j, level);
            temp_is_admissible.insert(i, j, height - level,
                                      std::move(value));
          }
        }
      }
    }

    is_admissible = temp_is_admissible;
    std::reverse(std::begin(level_blocks), std::end(level_blocks));
  }
  else if (admis_kind == "diagonal_admis") {
    if (matrix_type == BLR2_MATRIX) {
      height = 1;
      const int64_t nblocks = domain.boxes.size();
      for (int64_t i = 0; i < nblocks; i++) {
        for (int64_t j = 0; j < nblocks; j++) {
          is_admissible.insert(i, j, height, std::abs(i - j) > admis);
        }
      }
      level_blocks.push_back(1);
      level_blocks.push_back(nblocks);
    }
    else if (matrix_type == H2_MATRIX) {
      height = int64_t(log2(N / nleaf));
      calc_diagonal_based_admissibility(height);
      std::reverse(std::begin(level_blocks), std::end(level_blocks));
    }
  }
  else {
    std::cout << "wrong admis condition: " << admis_kind << std::endl;
    exit(EXIT_FAILURE);
  }

  is_admissible.insert(0, 0, 0, false);

  // const int64_t all_dense_row = find_all_dense_row();
  // if (all_dense_row != -1) {
  //   std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
  //   exit(EXIT_FAILURE);
  // }

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
        const Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j, height, height);
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
          const Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, i, j, level, height);
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

void H2::pre_compute_fill_in(const int64_t level,
                             RowColLevelMap<Matrix>& F) const {
  const int64_t nblocks = level_blocks[level];
  for (int64_t k = 0; k < nblocks; k++) {
    Matrix Dkk = D(k, k, level);
    lu(Dkk);
    for (int64_t i = 0; i < nblocks; i++) {
      if (i != k && is_admissible.exists(i, k, level) && !is_admissible(i, k, level)) {
        Matrix Dik = D(i, k, level);
        solve_triangular(Dkk, Dik, Hatrix::Right, Hatrix::Upper, false, false);
        for (int64_t j = 0; j < nblocks; j++) {
          if (j != k && is_admissible.exists(k, j, level) && !is_admissible(k, j, level)) {
            Matrix Dkj = D(k, j, level);
            solve_triangular(Dkk, Dkj, Hatrix::Left, Hatrix::Lower, true, false);

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

void H2::update_row_cluster_basis(const int64_t level,
                                  const RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  RowMap<Matrix> r;
  for (int64_t i = 0; i < nblocks; i++) {  // Parallel OK
    const int64_t block_size = D(i, i, level).rows;
    // Concat fill-ins along the row
    Matrix fill_in(block_size, 0);
    for (int64_t j = 0; j < nblocks; j++) {
      if (F.exists(i, j, level)) {
        fill_in = concat(fill_in, F(i, j, level), 1);
      }
    }
    if (fill_in.cols > 0) {  // Fill-in found
      // Compute new shared basis for both fill-in and low-rank blocks
      Matrix block_row = U.exists(i, level) ?
                         concat(US_row(i, level), fill_in, 1) :
                         fill_in;
      Matrix Ui, Si, Vi;
      int64_t rank;
      std::tie(Ui, Si, Vi, rank) = svd_like_compression(block_row);
      Ui.shrink(Ui.rows, rank);

      if (U.exists(i, level)) {  // not all dense row
        r.insert(i, matmul(Ui, U(i, level), true, false));
        // Update coupling matrices along the row
        for (int64_t j = 0; j < nblocks; j++) {
          if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
            S(i, j, level) = matmul(r(i), S(i, j, level));
          }
        }
        U.erase(i, level);
      }
      U.insert(i, level, std::move(Ui));
    }
  }
  // Update transfer matrices one level higher
  const int64_t parent_level = level - 1;
  if (parent_level > 0) {
    const int64_t parent_nblocks = level_blocks[parent_level];
    for (int64_t parent_node = 0; parent_node < parent_nblocks; parent_node++) {  // Parallel OK
      const int64_t c1 = parent_node * 2;
      const int64_t c2 = parent_node * 2 + 1;
      if (U.exists(parent_node, parent_level) && (r.exists(c1) || r.exists(c2))) {
        Matrix& Utransfer = U(parent_node, parent_level);
        Matrix Utransfer_new(U(c1, level).cols + U(c2, level).cols, Utransfer.cols);
        auto Utransfer_splits = Utransfer.split(vec{r.exists(c1) ? r(c1).cols : U(c1, level).cols}, vec{});
        auto Utransfer_new_splits = Utransfer_new.split(vec{U(c1, level).cols}, vec{});
        Utransfer_new_splits[0] = r.exists(c1) ? matmul(r(c1), Utransfer_splits[0]) : Utransfer_splits[0];
        Utransfer_new_splits[1] = r.exists(c2) ? matmul(r(c2), Utransfer_splits[1]) : Utransfer_splits[1];

        // Project saved US_row
        Matrix& US = US_row(parent_node, parent_level);
        Matrix US_new(U(c1, level).cols + U(c2, level).cols, US.cols);
        auto US_splits = US.split(vec{r.exists(c1) ? r(c1).cols : U(c1, level).cols}, vec{});
        auto US_new_splits = US_new.split(vec{U(c1, level).cols}, vec{});
        US_new_splits[0] = r.exists(c1) ? matmul(r(c1), US_splits[0]) : US_splits[0];
        US_new_splits[1] = r.exists(c2) ? matmul(r(c2), US_splits[1]) : US_splits[1];

        if (r.exists(c1)) r.erase(c1);
        if (r.exists(c2)) r.erase(c2);
        U.erase(parent_node, parent_level);
        US_row.erase(parent_node, parent_level);
        U.insert(parent_node, parent_level, std::move(Utransfer_new));
        US_row.insert(parent_node, parent_level, std::move(US_new));
      }
    }
  }
}

void H2::update_col_cluster_basis(const int64_t level,
                                  const RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  RowMap<Matrix> t;
  for (int64_t j = 0; j < nblocks; j++) {  // Parallel OK
    const int64_t block_size = D(j, j, level).cols;
    // Concat fill-ins along the column
    Matrix fill_in(0, block_size);
    for (int64_t i = 0; i < nblocks; i++) {
      if (F.exists(i, j, level)) {
        fill_in = concat(fill_in, F(i, j, level), 0);
      }
    }
    if (fill_in.rows > 0) {  // Fill-in found
      // Compute new shared basis for both fill-in and low-rank blocks
      Matrix block_column = V.exists(j, level) ?
                            concat(SV_col(j, level), fill_in, 0) :
                            fill_in;
      Matrix block_column_T = transpose(block_column);
      Matrix Vj, Sj_T, Uj_T;
      int64_t rank;
      std::tie(Vj, Sj_T, Uj_T, rank) = svd_like_compression(block_column_T);
      Vj.shrink(Vj.rows, rank);

      if (V.exists(j, level)) {  // Not all dense column
        t.insert(j, matmul(Vj, V(j, level), true, false));
        // Update coupling matrices along the column
        for (int64_t i = 0; i < nblocks; i++) {
          if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
            S(i, j, level) = matmul(S(i, j, level), t(j), false, true);
          }
        }
        V.erase(j, level);
      }
      V.insert(j, level, std::move(Vj));
    }
  }
  // Update transfer matrices one level higher
  const int64_t parent_level = level - 1;
  if (parent_level > 0) {
    const int64_t parent_nblocks = level_blocks[parent_level];
    for (int64_t parent_node = 0; parent_node < parent_nblocks; parent_node++) {  // Parallel OK
      const int64_t c1 = parent_node * 2;
      const int64_t c2 = parent_node * 2 + 1;
      if (V.exists(parent_node, parent_level) && (t.exists(c1) || t.exists(c2))) {
        Matrix& Vtransfer = V(parent_node, parent_level);
        Matrix Vtransfer_new(V(c1, level).cols + V(c2, level).cols, Vtransfer.cols);
        auto Vtransfer_splits = Vtransfer.split(vec{t.exists(c1) ? t(c1).cols : V(c1, level).cols}, vec{});
        auto Vtransfer_new_splits = Vtransfer_new.split(vec{V(c1, level).cols}, vec{});
        Vtransfer_new_splits[0] = t.exists(c1) ? matmul(t(c1), Vtransfer_splits[0]) : Vtransfer_splits[0];
        Vtransfer_new_splits[1] = t.exists(c2) ? matmul(t(c2), Vtransfer_splits[1]) : Vtransfer_splits[1];

        // Project saved SV_col
        Matrix& SV = SV_col(parent_node, parent_level);
        Matrix SV_new(SV.rows, V(c1, level).cols + V(c2, level).cols);
        auto SV_splits = SV.split(vec{}, vec{t.exists(c1) ? t(c1).cols : V(c1, level).cols});
        auto SV_new_splits = SV_new.split(vec{}, vec{V(c1, level).cols});
        SV_new_splits[0] = t.exists(c1) ? matmul(SV_splits[0], t(c1), false, true) : SV_splits[0];
        SV_new_splits[1] = t.exists(c2) ? matmul(SV_splits[1], t(c2), false, true) : SV_splits[1];

        if (t.exists(c1)) t.erase(c1);
        if (t.exists(c2)) t.erase(c2);
        V.erase(parent_node, parent_level);
        SV_col.erase(parent_node, parent_level);
        V.insert(parent_node, parent_level, std::move(Vtransfer_new));
        SV_col.insert(parent_node, parent_level, std::move(SV_new));
      }
    }
  }
}

void H2::propagate_upper_level_fill_in(const int64_t level,
                                       RowColLevelMap<Matrix>& F) const {
  const int64_t parent_level = level - 1;
  if (parent_level == 0) return;

  const int64_t parent_nblocks = level_blocks[parent_level];
  for (int64_t i = 0; i < parent_nblocks; i++) {  // Parallel OK (collapse i and j)
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
          const int64_t ncols = V(j_c1, level).cols + V(j_c2, level).cols;
          Matrix fill_in(nrows, ncols);
          auto fill_in_splits = fill_in.split(vec{U(i_c1, level).cols},
                                              vec{V(j_c1, level).cols});
          if (F.exists(i_c1, j_c1, level)) {
            matmul(matmul(U(i_c1, level), F(i_c1, j_c1, level), true, false),
                   V(j_c1, level), fill_in_splits[0], false, false, 1, 0);
          }
          if (F.exists(i_c1, j_c2, level)) {
            matmul(matmul(U(i_c1, level), F(i_c1, j_c2, level), true, false),
                   V(j_c2, level), fill_in_splits[1], false, false, 1, 0);
          }
          if (F.exists(i_c2, j_c1, level)) {
            matmul(matmul(U(i_c2, level), F(i_c2, j_c1, level), true, false),
                   V(j_c1, level), fill_in_splits[2], false, false, 1, 0);
          }
          if (F.exists(i_c2, j_c2, level)) {
            matmul(matmul(U(i_c2, level), F(i_c2, j_c2, level), true, false),
                   V(j_c2, level), fill_in_splits[3], false, false, 1, 0);
          }
          F.insert(i, j, parent_level, std::move(fill_in));
        }
      }
    }
  }
}

void H2::factorize_level(const int64_t level) {
  const int64_t parent_level = level - 1;
  const int64_t nblocks = level_blocks[level];
  for (int64_t block = 0; block < nblocks; ++block) {  // Parallel OK
    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - V(block, level).cols;
    /*
      Checks whether row basis rank == column basis rank
      If the following assertion failed, it means that there exists a cluster whose
      row basis rank is not equal to its column basis rank, leading to a rectangular diagonal block.
      The code should be adjusted to handle this case if necessary
     */
    assert(diag_row_split == diag_col_split);

    auto diagonal_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    lu(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = block+1; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false, false);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false, false);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = block+1; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        auto D_splits = D(block, j, level).split(vec{diag_row_split},
                                                 vec{D(block, j, level).cols - V(j, level).cols});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
      }
    }
    // TRSM with co blocks on the row
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        auto D_splits = D(block, j, level).split(vec{diag_row_split},
                                                 vec{D(block, j, level).cols - V(j, level).cols});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
      }
    }

    // Schur's complement into own oo part
    // oc x co -> oo
    matmul(diagonal_splits[2], diagonal_splits[1], diagonal_splits[3],
           false, false, -1.0, 1.0);
  }
}

void H2::factorize() {
  int64_t level = height;
  RowColLevelMap<Matrix> F;

  for (; level > 0; level--) {
    const int64_t nblocks = level_blocks[level];
    // Make sure there is no full-rank cluster basis
    for (int64_t i = 0; i < nblocks; ++i) {
      if (U.exists(i, level) && (U(i, level).rows <= U(i, level).cols)) {
        throw std::domain_error("Full rank cluster basis found at U(" + std::to_string(i) +
                                "," + std::to_string(level) + ")");
      }
    }
    for (int64_t j = 0; j < nblocks; ++j) {
      if (V.exists(j, level) && (V(j, level).rows <= V(j, level).cols)) {
        throw std::domain_error("Full rank cluster basis found at V(" + std::to_string(j) +
                                "," + std::to_string(level) + ")");
      }
    }

    pre_compute_fill_in(level, F);
    update_row_cluster_basis(level, F);
    update_col_cluster_basis(level, F);
    propagate_upper_level_fill_in(level, F);

    // Multiply with (U_F)^T from left
    for (int64_t i = 0; i < nblocks; i++) {  // Parallel OK
      Matrix U_F = prepend_complement_basis(U(i, level));
      for (int64_t j = 0; j < nblocks; j++) {
        if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          D(i, j, level) = matmul(U_F, D(i, j, level), true);
        }
      }
    }
    // Multiply with V_F from right
    for (int64_t j = 0; j < nblocks; j++) {  // Parallel OK
      Matrix V_F = prepend_complement_basis(V(j, level));
      for (int64_t i = 0; i < nblocks; i++) {
        if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          D(i, j, level) = matmul(D(i, j, level), V_F);
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
          int64_t nrows=0, ncols=0;
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
                    vec{D(c1, c2, level).cols - V(c2, level).cols});
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
  lu(D(0, 0, level));
}

// Permute the vector forward and return the offset at which the new vector begins.
int64_t H2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const {
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
int64_t H2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) const {
  Matrix copy(x);
  const int64_t nblocks = level_blocks[level];
  int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < nblocks; block++) {
    c_offset -= D(block, block, level).cols - V(block, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < nblocks; block++) {
    const int64_t cols = D(block, block, level).cols;
    const int64_t rank = V(block, level).cols;
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

void H2::solve_forward_level(Matrix& x_level, const int64_t level) const {
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
    const int64_t diag_col_split = D(block, block, level).cols - V(block, level).cols;
    assert(diag_row_split == diag_col_split); // Row basis rank = column basis rank

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

void H2::solve_backward_level(Matrix& x_level, const int64_t level) const {
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
    const int64_t diag_col_split = D(block, block, level).cols - V(block, level).cols;
    assert(diag_row_split == diag_col_split); // Row basis rank = column basis rank

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(vec{diag_row_split}, vec{});
    // Backward substitution with co blocks in the left of diagonal
    for (int64_t jcol = block-1; jcol >= 0; jcol--) {
      if (is_admissible.exists(block, jcol, level) && !is_admissible(block, jcol, level)) {
        const int64_t left_row_split = diag_row_split;
        const int64_t left_col_split = D(block, jcol, level).cols - V(jcol, level).cols;
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
    // Solve backward with diagonal U
    auto U_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    matmul(U_block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
    solve_triangular(U_block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
    // Write x_block
    x_level_split[block] = x_block;
  }
  // Multiply with V_F at the end
  for (int64_t block = nblocks-1; block >= 0; block--) {
    Matrix V_F = prepend_complement_basis(V(block, level));
    x_level_split[block] = matmul(V_F, x_level_split[block]);
  }
}

Matrix H2::solve(const Matrix& b) const {
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

  // Solve with root level LU
  auto x_splits = x.split(vec{rhs_offset}, vec{});
  const int64_t last_nodes = level_blocks[level];
  assert(level == 0);
  assert(last_nodes == 1);
  solve_triangular(D(0, 0, level), x_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
  solve_triangular(D(0, 0, level), x_splits[1], Hatrix::Left, Hatrix::Upper, false, false);
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
  const int64_t nleaf = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-5;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const int64_t sample_size = argc > 5 ? atol(argv[5]) : 100;
  const double admis = argc > 6 ? atof(argv[6]) : 1.0;

  // Specify admissibility type
  // diagonal_admis: Admissibility based on absolute distance from diagonal block
  // geometry_admis: Admissibility based on particles' geometric distance
  const std::string admis_kind = argc > 7 ? std::string(argv[7]) : "geometry_admis";

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 8 ? atol(argv[8]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 9 ? atol(argv[9]) : 0;
  const int64_t ndim  = argc > 10 ? atol(argv[10]) : 2;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 11 ? atol(argv[11]) : 1;

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1e-3 / (double)N, 1.);
  std::string kernel_name = "";
  switch (kernel_type) {
    case 0: {
      Hatrix::kernel_function = Hatrix::laplace_kernel;
      kernel_name = "laplace";
      break;
    }
    case 1: {
      Hatrix::kernel_function = Hatrix::yukawa_kernel;
      kernel_name = "yukawa";
      break;
    }
    default: {
      Hatrix::kernel_function = Hatrix::laplace_kernel;
      kernel_name = "laplace";
    }
  }

  const auto start_particles = std::chrono::system_clock::now();
  Hatrix::Domain domain(N, ndim);
  std::string geom_name = std::to_string(ndim) + "d-";
  switch (geom_type) {
    case 0: {
      domain.generate_unit_circular_mesh();
      geom_name += "circular_mesh";
      break;
    }
    case 1: {
      domain.generate_unit_cubical_mesh();
      geom_name += "cubical_mesh";
      break;
    }
    case 2: {
      domain.generate_starsh_uniform_grid();
      geom_name += "starsh_uniform_grid";
      break;
    }
    default: {
      domain.generate_unit_circular_mesh();
      geom_name += "circular_mesh";
    }
  }
  domain.divide_domain_and_create_particle_boxes(nleaf);
  const auto stop_particles = std::chrono::system_clock::now();
  const double particle_construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                         (stop_particles - start_particles).count();

  Hatrix::Matrix rand = Hatrix::generate_random_matrix(N, sample_size);
  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(domain, rand, N, nleaf, accuracy, max_rank, admis, admis_kind, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();  
  double construct_error = A.construction_absolute_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " nleaf=" << nleaf
            << " accuracy=" << accuracy
            << " max_rank=" << max_rank
            << " sample_size=" << sample_size
            << " compress_alg="
#ifdef USE_QR_COMPRESSION
            << "QR"
#else
            << "SVD"
#endif
            << " admis=" << admis << std::setw(3)
            << " admis_kind=" << admis_kind
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " height=" << A.height
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " construct_time=" << construct_time
            << std::scientific
            << " construct_error=" << construct_error
            << std::defaultfloat;

  const auto start_factor = std::chrono::system_clock::now();
  A.factorize();
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
  double solve_error = Hatrix::norm(x_solve - x);

  std::cout << " factor_min_rank=" << A.get_basis_min_rank()
            << " factor_max_rank=" << A.get_basis_max_rank()
            << " factor_time=" << factor_time
            << " solve_time=" << solve_time
            << " solve_error=" << solve_error
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
