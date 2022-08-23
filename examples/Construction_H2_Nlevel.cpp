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
            << " height=" << A.height
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
