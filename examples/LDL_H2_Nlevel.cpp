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

class SymmetricH2 {
 public:
  int64_t N, nleaf, n_blocks;
  double accuracy;
  int64_t max_rank;
  double admis;
  std::string admis_kind;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowLevelMap US_row;
  std::vector<int64_t> level_blocks;

 private:
  int64_t find_all_dense_row() const;
  void coarsen_blocks(const int64_t level);

  int64_t geometry_admis_non_leaf(const int64_t nblocks, const int64_t level);
  int64_t calc_geometry_based_admissibility(const Domain& domain);
  void calc_diagonal_based_admissibility(const int64_t level);

  int64_t get_block_size_row(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;

  std::tuple<Matrix, Matrix, Matrix, int64_t> svd_like_compression(Matrix& A) const;

  Matrix generate_block_row(const Domain& domain, const Matrix& rand,
                            const int64_t node, const int64_t level,
                            const int64_t block_size) const;
  std::tuple<Matrix, Matrix>
  generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(const Domain& domain, const Matrix& rand,
                             const Matrix& Ubig_child1, const Matrix& Ubig_child2,
                             const int64_t node, const int64_t level,
                             const int64_t block_size) const;
  RowLevelMap
  generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                             RowLevelMap& Uchild);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;

  void update_row_cluster_bases(const int64_t row, const int64_t level,
                                const RowColLevelMap<Matrix>& F,
                                RowMap<Matrix>& r);
  void factorize_level(const int64_t level, const int64_t nblocks,
                       RowColLevelMap<Matrix>& F, RowMap<Matrix>& r);

  int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) const;
  void solve_forward_level(Matrix& x_level, const int64_t level) const;
  void solve_diagonal_level(Matrix& x_level, const int64_t level) const;
  void solve_backward_level(Matrix& x_level, const int64_t level) const;

 public:
  SymmetricH2(const Domain& domain, const Matrix& rand,
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

int64_t SymmetricH2::find_all_dense_row() const {
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

void SymmetricH2::coarsen_blocks(const int64_t level) {
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

int64_t SymmetricH2::geometry_admis_non_leaf(const int64_t nblocks, const int64_t level) {
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

int64_t SymmetricH2::calc_geometry_based_admissibility(const Domain& domain) {
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

void SymmetricH2::calc_diagonal_based_admissibility(const int64_t level) {
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

int64_t SymmetricH2::get_block_size_row(const Domain& domain, const int64_t node, const int64_t level) const {
  if (level == height) {
    return domain.boxes[node].num_particles;
  }
  const int64_t child_level = level + 1;
  const int64_t child1 = node * 2;
  const int64_t child2 = node * 2 + 1;

  return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
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

std::tuple<Matrix, Matrix, Matrix, int64_t> SymmetricH2::svd_like_compression(Matrix& A) const {
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

Matrix SymmetricH2::generate_block_row(const Domain& domain, const Matrix& rand,
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

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_basis(const Domain& domain, const Matrix& rand,
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

void SymmetricH2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
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
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_interactions(domain, i, j, height, height);
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

RowLevelMap
SymmetricH2::generate_transfer_matrices(const Domain& domain, const Matrix& rand, const int64_t level,
                                        RowLevelMap& Uchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;

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
  }

  for (int64_t i = 0; i < nblocks; i++) {
    for (int64_t j = 0; j < nblocks; j++) {
      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
        Matrix D = generate_p2p_interactions(domain, i, j, level, height);

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

  const int64_t all_dense_row = find_all_dense_row();
  if (all_dense_row != -1) {
    std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
    exit(EXIT_FAILURE);
  }

  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;

  for (int64_t level = height - 1; level > 0; level--) {
    Uchild = generate_transfer_matrices(domain, rand, level, Uchild);
  }
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

double SymmetricH2::construction_absolute_error(const Domain& domain) const {
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
          const Matrix Vbig = get_Ubig(j, level);
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

void SymmetricH2::update_row_cluster_bases(const int64_t row, const int64_t level,
                                           const RowColLevelMap<Matrix>& F,
                                           RowMap<Matrix>& r) {
  const int64_t nblocks = level_blocks[level];
  const int64_t block_size = D(row, row, level).rows;
  Matrix block_row(block_size, 0);

  // TODO implement a more accurate variant from MiaoMiaoMa2019_UMV paper (Algorithm 1)
  // instead of using a pre-computed UxS from construction phase
  block_row = concat(block_row, US_row(row, level), 1);

  // Concat fill-in blocks
  for (int64_t j = 0; j < nblocks; j++) {
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

void SymmetricH2::factorize_level(const int64_t level, const int64_t nblocks,
                                  RowColLevelMap<Matrix>& F, RowMap<Matrix>& r) {
  const int64_t parent_level = level - 1;
  for (int64_t block = 0; block < nblocks; block++) {
    const int64_t parent_node = block / 2;
    // Check for fill-ins along row
    bool found_row_fill_in = false;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (F.exists(block, j, level)) {
        found_row_fill_in = true;
        break;
      }
    }
    // Update cluster bases if necessary
    if (found_row_fill_in) {
      update_row_cluster_bases(block, level, F, r);
      // Project admissible blocks accordingly
      // Current level: update coupling matrix
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
          S(block, j, level) = matmul(r(block), S(block, j, level));
        }
      }
      for (int i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
          S(i, block, level) = matmul(S(i, block, level), r(block), false, true);
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
        if (block == c1) {
          auto Utransfer_splits = Utransfer.split(vec{r(c1).cols}, vec{});
          matmul(r(c1), Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1, 0);
          Utransfer_new_splits[1] = Utransfer_splits[1];

          auto US_splits = US.split(vec{r(c1).cols}, vec{});
          matmul(r(c1), US_splits[0], US_new_splits[0], false, false, 1, 0);
          US_new_splits[1] = US_splits[1];

          r.erase(c1);
        }
        else { // block == c2
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
    Matrix U_F = prepend_complement_basis(U(block, level));
    // Multiply to dense blocks along the row in current level
    for (int j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        if (j < block) {
          // Do not touch the eliminated part (cc and oc)
          int64_t left_col_split = D(block, j, level).cols - U(j, level).cols;
          auto D_splits = D(block, j, level).split(vec{}, vec{left_col_split});
          D_splits[1] = matmul(U_F, D_splits[1], true);
        }
        else {
          D(block, j, level) = matmul(U_F, D(block, j, level), true);
        }
      }
    }
    // Multiply to dense blocks along the column in current level
    for (int i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        if (i < block) {
          // Do not touch the eliminated part (cc and co)
          int64_t up_row_split = D(i, block, level).rows - U(i, level).cols;
          auto D_splits = D(i, block, level).split(vec{up_row_split}, vec{});
          D_splits[1] = matmul(D_splits[1], U_F);
        }
        else {
          D(i, block, level) = matmul(D(i, block, level), U_F);
        }
      }
    }

    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    auto diagonal_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = block+1; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t lower_row_split = D(i, block, level).rows -
                                  (level == height ? U(i, level).cols : U(i * 2, level + 1).cols);
        auto D_splits = D(i, block, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t lower_row_split = D(i, block, level).rows -
                                  (i <= block || level == height ?
                                   U(i, level).cols :
                                   U(i * 2, level + 1).cols);
        auto D_splits = D(i, block, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = block + 1; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t right_col_split = D(block, j, level).cols -
                                  (level == height ? U(j, level).cols : U(j * 2, level + 1).cols);
        auto D_splits = D(block, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }
    // TRSM with co blocks on this row
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t right_col_split = D(block, j, level).cols -
                                  (j <= block || level == height ?
                                   U(j, level).cols :
                                   U(j * 2, level + 1).cols);
        auto D_splits = D(block, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[1], Hatrix::Left);
      }
    }

    // Schur's complement into dense block
    // cc x cc -> cc
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
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
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= block || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
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
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= block || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
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
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= block || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= block || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
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
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*b fill-in block
            int64_t nrows = D(i, block, level).rows;
            int64_t ncols = D(block, j, level).cols;
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
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces b*rank fill-in
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < block; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*rank fill-in block
            int64_t nrows = D(i, block, level).rows;
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
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*b fill-in
    for (int64_t i = 0; i < block; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create rank*b fill-in block
            int64_t nrows = lower_row_rank;
            int64_t ncols = D(block, j, level).cols;
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
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*rank fill-in
    for (int64_t i = 0; i < block; ++i) {
      for (int64_t j = 0; j < block; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
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
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
  } // for (int block = 0; block < nblocks; ++block)
}

void SymmetricH2::factorize() {
  int64_t level = height;
  RowColLevelMap<Matrix> F;

  for (; level > 0; --level) {
    RowMap<Matrix> r;
    const int64_t nblocks = level_blocks[level];
    // Make sure all cluster bases exist and none of them is full-rank
    for (int64_t i = 0; i < nblocks; ++i) {
      if (!U.exists(i, level)) {
        throw std::logic_error("Cluster bases not found at U(" + std::to_string(i) +
                               "," + std::to_string(level) + ")");
      }
      if (U(i, level).rows <= U(i, level).cols) {
        throw std::domain_error("Full rank cluster bases found at U(" + std::to_string(i) +
                                "," + std::to_string(level) + ")");
      }
    }

    factorize_level(level, nblocks, F, r);

    // Update coupling matrices of admissible blocks in the current level
    // To add fill-in contributions
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
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
    const int64_t parent_nblocks = level_blocks[parent_level];
    // Propagate fill-in to upper level admissible blocks (if any)
    if (parent_level > 0) {
      for (int64_t i = 0; i < parent_nblocks; ++i) {
        for (int64_t j = 0; j < parent_nblocks; ++j) {
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
      for (int64_t block = 0; block < nblocks; block += 2) {
        int64_t parent_node = block / 2;
        if (!U.exists(parent_node, parent_level)) {
          // Use identity matrix as U bases whenever all dense row is encountered
          int64_t c1 = block;
          int64_t c2 = block + 1;
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
    for (int64_t i = 0; i < parent_nblocks; ++i) {
      for (int64_t j = 0; j < parent_nblocks; ++j) {
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

// Permute the vector forward and return the offset at which the new vector begins.
int64_t SymmetricH2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) const {
  Matrix copy(x);
  const int64_t nblocks = level_blocks[level];
  const int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < nblocks; ++block) {
    rank_offset += D(block, block, level).rows - U(block, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < nblocks; ++block) {
    const int64_t rows = D(block, block, level).rows;
    const int64_t rank = U(block, level).cols;
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
  const int64_t nblocks = level_blocks[level];
  int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < nblocks; ++block) {
    c_offset -= D(block, block, level).cols - U(block, level).cols;
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < nblocks; ++block) {
    const int64_t cols = D(block, block, level).cols;
    const int64_t rank = U(block, level).cols;
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
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> row_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    row_offsets.push_back(nrows + D(i, i, level).rows);
    nrows += D(i, i, level).rows;
  }
  auto x_level_split = x_level.split(row_offsets, vec{});

  for (int64_t block = 0; block < nblocks; ++block) {
    const int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    const int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    // Multiply with (U_F)^T
    Matrix U_F = prepend_complement_basis(U(block, level));
    Matrix x_block = matmul(U_F, x_level_split[block], true);
    auto x_block_splits = x_block.split(vec{diag_col_split}, vec{});
    // Solve forward with diagonal L
    auto L_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_triangular(L_block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
    // Forward substitution with oc block on the diagonal
    matmul(L_block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
    // Forward substitution with cc and oc blocks below the diagonal
    for (int64_t irow = block+1; irow < nblocks; ++irow) {
      if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
        auto lower_splits = D(irow, block, level).split(vec{}, vec{diag_col_split});
        matmul(lower_splits[0], x_block_splits[0], x_level_split[irow], false, false, -1.0, 1.0);
      }
    }
    // Forward substitution with oc blocks above the diagonal
    for (int64_t irow = 0; irow < block; ++irow) {
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

void SymmetricH2::solve_backward_level(Matrix& x_level, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  for (int64_t block = nblocks-1; block >= 0; --block) {
    const int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    const int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(vec{diag_row_split}, vec{});
    // Backward substitution with co blocks in the left of diagonal
    for (int64_t jcol = block-1; jcol >= 0; --jcol) {
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
    for (int64_t jcol = nblocks-1; jcol > block; --jcol) {
      if (is_admissible.exists(block, jcol, level) && !is_admissible(block, jcol, level)) {
        auto right_splits = D(block, jcol, level).split(vec{diag_row_split}, vec{});
        matmul(right_splits[0], x_level_split[jcol], x_block_splits[0], false, false, -1.0, 1.0);
      }
    }
    // Solve backward with diagonal L
    auto L_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    matmul(L_block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
    solve_triangular(L_block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true, true);
    // Multiply with U_F
    Matrix U_F = prepend_complement_basis(U(block, level));
    x_block = matmul(U_F, x_block);
    // Write x_block
    x_level_split[block] = x_block;
  }
}

void SymmetricH2::solve_diagonal_level(Matrix& x_level, const int64_t level) const {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    col_offsets.push_back(nrows + D(i, i, level).cols);
    nrows += D(i, i, level).cols;
  }
  auto x_level_split = x_level.split(col_offsets, {});

  // Solve diagonal using cc blocks
  for (int64_t block = nblocks-1; block >= 0; --block) {
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    assert(diag_row_split == diag_col_split); // Row bases rank = column bases rank

    Matrix x_block(x_level_split[block], true);  // Deep-copy of view
    auto x_block_splits = x_block.split(vec{diag_col_split}, {});
    // Solve with cc block on the diagonal
    auto D_block_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    solve_diagonal(D_block_splits[0], x_block_splits[0], Hatrix::Left);
    x_level_split[block] = x_block;
  }
}

Matrix SymmetricH2::solve(const Matrix& b) const {
  Matrix x(b);
  int64_t level = height;
  int64_t rhs_offset = 0;

  // Forward
  for (; level > 0; --level) {
    const int64_t nblocks = level_blocks[level];
    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
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
    const int64_t nblocks = level_blocks[level];

    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
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
  Hatrix::SymmetricH2 A(domain, rand, N, nleaf, accuracy, max_rank, admis, admis_kind, matrix_type);
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
            << std::defaultfloat
            << " " << std::flush;

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

  std::cout << "factor_min_rank=" << A.get_basis_min_rank()
            << " factor_max_rank=" << A.get_basis_max_rank()
            << " factor_time=" << factor_time
            << " solve_time=" << solve_time
            << " solve_error=" << solve_error
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
