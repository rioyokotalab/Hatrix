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
  int64_t rank;
  double admis;
  std::string admis_kind;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowLevelMap Srow;
  std::vector<int64_t> level_blocks;
  int64_t min_rank, max_rank;

 private:
  int64_t find_all_dense_row();
  void coarsen_blocks(int64_t level);

  int64_t geometry_admis_non_leaf(int64_t nblocks, int64_t level);
  int64_t calc_geometry_based_admissibility(const Domain& domain);
  void calc_diagonal_based_admissibility(int64_t level);

  int64_t get_block_size_row(const Domain& domain, int64_t parent, int64_t level);
  int64_t get_block_size_col(const Domain& domain, int64_t parent, int64_t level);
  bool row_has_admissible_blocks(int64_t row, int64_t level);
  bool col_has_admissible_blocks(int64_t col, int64_t level);
  std::tuple<Matrix, Matrix, Matrix> svd_like_compression(Matrix& A);
  Matrix generate_block_row(int64_t block, int64_t block_size,
                            const Domain& domain, int64_t level,
                            const Matrix& rand);
  std::tuple<Matrix, Matrix>
  generate_row_cluster_bases(int64_t block, int64_t block_size,
                             const Domain& domain, int64_t level,
                             const Matrix& rand);
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                             int64_t block_size, const Domain& domain, int64_t level,
                             const Matrix& rand);
  RowLevelMap generate_transfer_matrices(const Domain& domain,
                                         int64_t level, const Matrix& rand,
                                         RowLevelMap& Uchild);
  Matrix get_Ubig(int64_t node, int64_t level);
  void actually_print_structure(int64_t level);

  void pre_compute_fill_in(const int64_t level, RowColLevelMap<Matrix>& F);
  void form_new_cluster_bases(const int64_t level, const RowColLevelMap<Matrix>& F);
  void add_fill_in_contribution(const int64_t level, const RowColLevelMap<Matrix>& F);
  void propagate_upper_level_fill_in(const int64_t level, RowColLevelMap<Matrix>& F);
  void factorize_level(const int64_t level);

  int64_t permute_forward(Matrix& x, int64_t level, int64_t rank_offset);
  int64_t permute_backward(Matrix& x, int64_t level, int64_t rank_offset);
  void solve_forward_level(Matrix& x_level, int64_t level);
  void solve_backward_level(Matrix& x_level, int64_t level);
  void solve_diagonal_level(Matrix& x_level, int64_t level);

 public:
  SymmetricH2(const Domain& domain, const int64_t N, const int64_t nleaf,
              const double accuracy, const int64_t rank, const double admis,
              const std::string& admis_kind, const int64_t matrix_type,
              const Matrix& rand);

  double construction_absolute_error(const Domain& domain);
  void print_structure();
  void print_ranks();
  double low_rank_block_ratio();
  void factorize();
  Matrix solve(const Matrix& b, int64_t _level);
};

int64_t SymmetricH2::find_all_dense_row() {
  int64_t nblocks = level_blocks[height];

  for (int64_t i = 0; i < nblocks; ++i) {
    bool all_dense_row = true;
    for (int64_t j = 0; j < nblocks; ++j) {
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

void SymmetricH2::coarsen_blocks(int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

int64_t SymmetricH2::geometry_admis_non_leaf(int64_t nblocks, int64_t level) {
  int64_t child_level = level - 1;
  level_blocks.push_back(nblocks);

  if (nblocks == 1) { return level; }

  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
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
  int64_t nblocks = domain.boxes.size();
  level_blocks.push_back(nblocks);
  int64_t level = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
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

void SymmetricH2::calc_diagonal_based_admissibility(int64_t level) {
  int64_t nblocks = (int64_t)std::pow(2., level);
  level_blocks.push_back(nblocks);
  if (level == 0) { return; }
  if (level == height) {
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
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

int64_t SymmetricH2::get_block_size_row(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
}

int64_t SymmetricH2::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
}

bool SymmetricH2::row_has_admissible_blocks(int64_t row, int64_t level) {
  bool has_admis = false;
  for (int64_t j = 0; j < level_blocks[level]; ++j) {
    if ((!is_admissible.exists(row, j, level)) || // part of upper level admissible block
        (is_admissible.exists(row, j, level) && is_admissible(row, j, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

bool SymmetricH2::col_has_admissible_blocks(int64_t col, int64_t level) {
  bool has_admis = false;
  for (int64_t i = 0; i < level_blocks[level]; ++i) {
    if ((!is_admissible.exists(i, col, level)) || // part of upper level admissible block
        (is_admissible.exists(i, col, level) && is_admissible(i, col, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

std::tuple<Matrix, Matrix, Matrix> SymmetricH2::svd_like_compression(Matrix& A) {
  Matrix Ui, Si, Vi;
  int64_t _rank;
  if (accuracy == 0.) {  // Fixed rank
    double error;
    std::tie(Ui, Si, Vi, error) = truncated_svd(A, std::min(rank, A.min_dim()));
  }
  else {  // Fixed accuracy
#ifdef USE_QR_COMPRESSION
    Matrix R;
    std::tie(Ui, R, _rank) = error_pivoted_qr(A, accuracy, false);
    Si = Matrix(R.rows, R.rows);
    Vi = Matrix(R.rows, R.cols);
    rq(R, Si, Vi);
#else
    std::tie(Ui, Si, Vi, _rank) = error_svd(A, accuracy, false);
#endif
  }
  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi));
}

Matrix SymmetricH2::generate_block_row(int64_t block, int64_t block_size,
                                       const Domain& domain, int64_t level,
                                       const Matrix& rand) {
  int64_t nblocks = level_blocks[level];
  std::vector<Matrix> rand_splits;
  bool sample = (rank > 0);
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_row(block_size, sample ? rand.cols : 0);
  for (int64_t j = 0; j < nblocks; ++j) {
    if ((!is_admissible.exists(block, j, level)) || // part of upper level admissible block
        (is_admissible.exists(block, j, level) && is_admissible(block, j, level))) {
      if (sample) {
        matmul(generate_p2p_interactions(domain, block, j, level, height), rand_splits[j],
               block_row, false, false, 1.0, 1.0);
      }
      else {
        block_row =
            concat(block_row, generate_p2p_interactions(domain, block, j, level, height), 1);
      }
    }
  }
  return block_row;
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_bases(int64_t block, int64_t block_size,
                                        const Domain& domain, int64_t level,
                                        const Matrix& rand) {
  Matrix block_row = generate_block_row(block, block_size, domain, level, rand);
  Matrix Ui, Si, Vi;
  std::tie(Ui, Si, Vi) = svd_like_compression(block_row);
  min_rank = std::min(min_rank, Ui.cols);
  max_rank = std::max(max_rank, Ui.cols);

  Matrix SV = matmul(Si, Vi);
  return std::make_tuple(std::move(Ui), std::move(SV));
}

void SymmetricH2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
  int64_t nblocks = level_blocks[height];
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_interactions(domain, i, j, height, height));
      }
    }
  }
  // Generate leaf level U
  for (int64_t i = 0; i < nblocks; ++i) {
    Matrix Utemp, Stemp;
    std::tie(Utemp, Stemp) =
        generate_row_cluster_bases(i, domain.boxes[i].num_particles, domain, height, rand);
    U.insert(i, height, std::move(Utemp));
    Srow.insert(i, height, std::move(Stemp));
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
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
SymmetricH2::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                        int64_t block_size, const Domain& domain, int64_t level,
                                        const Matrix& rand) {
  Matrix block_row = generate_block_row(node, block_size, domain, level, rand);
  auto block_row_splits = block_row.split(2, 1);

  Matrix temp(Ubig_child1.cols + Ubig_child2.cols, block_row.cols);
  auto temp_splits = temp.split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, block_row_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Ubig_child2, block_row_splits[1], temp_splits[1], true, false, 1, 0);

  Matrix Ui, Si, Vi;
  std::tie(Ui, Si, Vi) = svd_like_compression(temp);
  min_rank = std::min(min_rank, Ui.cols);
  max_rank = std::max(max_rank, Ui.cols);

  Matrix SV = matmul(Si, Vi);
  return std::make_tuple(std::move(Ui), std::move(SV));
}

RowLevelMap SymmetricH2::generate_transfer_matrices(const Domain& domain,
                                                    int64_t level, const Matrix& rand,
                                                    RowLevelMap& Uchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;

  int64_t nblocks = level_blocks[level];
  for (int64_t node = 0; node < nblocks; ++node) {
    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;
    int64_t child_level = level + 1;
    int64_t block_size = get_block_size_row(domain, node, level);

    if (level > 0 && row_has_admissible_blocks(node, level)) {
      // Generate row cluster transfer matrix.
      Matrix& Ubig_child1 = Uchild(child1, child_level);
      Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix Utransfer, Stemp;
      std::tie(Utransfer, Stemp) =
          generate_U_transfer_matrix(Ubig_child1, Ubig_child2,
                                     node, block_size, domain, level, rand);
      U.insert(node, level, std::move(Utransfer));
      Srow.insert(node, level, std::move(Stemp));

      // Generate the full bases to pass onto the parent.
      auto Utransfer_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});
      Matrix Ubig(block_size, U(node, level).cols);
      auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});

      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);
      Ubig_parent.insert(node, level, std::move(Ubig));
    }
  }

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < nblocks; ++col) {
      if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
        Matrix D = generate_p2p_interactions(domain, row, col, level, height);

        S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                         Ubig_parent(col, level)));
      }
    }
  }
  return Ubig_parent;
}

Matrix SymmetricH2::get_Ubig(int64_t node, int64_t level) {
  if (level == height) {
    return U(node, level);
  }

  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;
  Matrix Ubig_child1 = get_Ubig(child1, level+1);
  Matrix Ubig_child2 = get_Ubig(child2, level+1);

  int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, U(node, level).cols);
  auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
  auto U_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

SymmetricH2::SymmetricH2(const Domain& domain, const int64_t N, const int64_t nleaf,
                         const double accuracy, const int64_t rank, const double admis,
                         const std::string& admis_kind, const int64_t matrix_type,
                         const Matrix& rand)
    : N(N), nleaf(nleaf), accuracy(accuracy), rank(rank),
      admis(admis), admis_kind(admis_kind), matrix_type(matrix_type),
      min_rank(N), max_rank(-N) {
  if (admis_kind == "geometry_admis") {
    // TODO: use dual tree traversal for this.
    height = calc_geometry_based_admissibility(domain);
    // reverse the levels stored in the admis blocks.
    RowColLevelMap<bool> temp_is_admissible;

    for (int64_t level = 0; level < height; ++level) {
      int64_t nblocks = level_blocks[level];

      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
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
      int64_t nblocks = domain.boxes.size();
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
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

  int64_t all_dense_row = find_all_dense_row();
  if (all_dense_row != -1) {
    std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
    exit(EXIT_FAILURE);
  }

  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;

  for (int64_t level = height-1; level > 0; --level) {
    Uchild = generate_transfer_matrices(domain, level, rand, Uchild);
  }
}

double SymmetricH2::construction_absolute_error(const Domain& domain) {
  double error = 0;
  int64_t nblocks = level_blocks[height];

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j, height, height);
        Matrix expected = D(i, j, height);
        error += pow(norm(actual - expected), 2);
      }
    }
  }

  for (int64_t level = height; level > 0; --level) {
    int64_t nblocks = level_blocks[level];

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Ubig(col, level);

          Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, row, col, level, height);

          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }

  return std::sqrt(error);
}

void SymmetricH2::actually_print_structure(int64_t level) {
  if (level == 0) { return; }
  int64_t nblocks = level_blocks[level];
  std::cout << "LEVEL: " << level << " NBLOCKS: " << nblocks << std::endl;
  for (int64_t i = 0; i < nblocks; ++i) {
    if (level == height && D.exists(i, i, height)) {
      std::cout << D(i, i, height).rows << " ";
    }
    std::cout << "| ";
    for (int64_t j = 0; j < nblocks; ++j) {
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

  actually_print_structure(level-1);
}

void SymmetricH2::print_structure() {
  actually_print_structure(height);
}

void SymmetricH2::print_ranks() {
  for(int64_t level = height; level > 0; level--) {
    int64_t nblocks = level_blocks[level];
    for(int64_t block = 0; block < nblocks; block++) {
      std::cout << "block=" << block << "," << "level=" << level << ":\t"
                << "diag= ";
      if(D.exists(block, block, level)) {
        std::cout << D(block, block, level).rows << "x" << D(block, block, level).cols;
      }
      else {
        std::cout << "empty";
      }
      std::cout << ", row_rank=" << (U.exists(block, level) ?
                                     U(block, level).cols : -1)
                << ", col_rank=" << (U.exists(block, level) ?
                                     U(block, level).cols : -1)
                << std::endl;
    }
  }
}

double SymmetricH2::low_rank_block_ratio() {
  double total = 0, low_rank = 0;

  int64_t nblocks = level_blocks[height];
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
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
                                      RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  for (int64_t k = 0; k < nblocks; k++) {
    Matrix Dkk = D(k, k, level);
    ldl(Dkk);
    for (int64_t i = 0; i < nblocks; i++) {
      if (i != k && is_admissible.exists(i, k, level) && !is_admissible(i, k, level)) {
        Matrix Dik = D(i, k, level);
        solve_triangular(Dkk, Dik, Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dkk, Dik, Hatrix::Right);
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

void SymmetricH2::form_new_cluster_bases(const int64_t level,
                                         const RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  RowMap<Matrix> r;
  for (int64_t i = 0; i < nblocks; i++) {  // Parallel OK
    const int64_t block_size = D(i, i, level).rows;
    Matrix fill_in(block_size, 0);
    for (int64_t j = 0; j < nblocks; j++) {
      if (F.exists(i, j, level)) {
        fill_in = concat(fill_in, F(i, j, level), 1);
      }
    }
    if (fill_in.cols > 0) {  // Fill-in found
      // Compute new shared bases for both fill-in and low-rank blocks
      Matrix block_row = concat(matmul(U(i, level), Srow(i, level)), fill_in, 1);
      Matrix Ui, Si, Vi;
      std::tie(Ui, Si, Vi) = svd_like_compression(block_row);

      r.insert(i, matmul(Ui, U(i, level), true, false));
      U.erase(i, level);
      U.insert(i, level, std::move(Ui));

      // Update coupling matrices along the row
      for (int64_t j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          S(i, j, level) = matmul(r(i), S(i, j, level));
        }
      }
    }
  }
  // SymmetricH2: Since U(i) = V(i), update coupling matrices along the column too
  for (int64_t j = 0; j < nblocks; j++) {  // Parallel OK
    if (r.exists(j)) {
      for (int64_t i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          S(i, j, level) = matmul(S(i, j, level), r(j), false, true);
        }
      }
    }
  }
  // Update transfer matrices one level higher
  const int64_t parent_level = level - 1;
  if (parent_level > 0) {
    const int64_t parent_nblocks = level_blocks[parent_level];
    for (int64_t parent_node = 0; parent_node < parent_nblocks; parent_node++) {  // Parallel OK
      const int64_t c1 = parent_node * 2 + 0;
      const int64_t c2 = parent_node * 2 + 1;
      if (!U.exists(parent_node, parent_level)) {
        // Insert dummy identity bases to all dense row
        const int64_t rank_c1 = U(c1, level).cols;
        const int64_t rank_c2 = U(c2, level).cols;
        const int64_t rank_parent = std::max(rank_c1, rank_c2);
        Matrix Utransfer =
            generate_identity_matrix(rank_c1 + rank_c2, rank_parent);

        if (r.exists(c1)) r.erase(c1);
        if (r.exists(c2)) r.erase(c2);
        U.insert(parent_node, parent_level, std::move(Utransfer));
      }
      else if (r.exists(c1) || r.exists(c2)) {
        Matrix& Utransfer = U(parent_node, parent_level);
        auto Utransfer_splits = Utransfer.split(vec{r.exists(c1) ? r(c1).cols : U(c1, level).cols}, vec{});

        Matrix Utransfer_new(U(c1, level).cols + U(c2, level).cols, Utransfer.cols);
        auto Utransfer_new_splits = Utransfer_new.split(vec{U(c1, level).cols}, vec{});
        Utransfer_new_splits[0] = r.exists(c1) ? matmul(r(c1), Utransfer_splits[0]) : Utransfer_splits[0];
        Utransfer_new_splits[1] = r.exists(c2) ? matmul(r(c2), Utransfer_splits[1]) : Utransfer_splits[1];

        if (r.exists(c1)) r.erase(c1);
        if (r.exists(c2)) r.erase(c2);
        U.erase(parent_node, parent_level);
        U.insert(parent_node, parent_level, std::move(Utransfer_new));
      }
    }
  }
}

void SymmetricH2::add_fill_in_contribution(const int64_t level,
                                           const RowColLevelMap<Matrix>& F) {
  const int64_t nblocks = level_blocks[level];
  for (int64_t i = 0; i < nblocks; i++) {  // Parallel OK (collapse i and j)
    for (int64_t j = 0; j < nblocks; j++) {
      if (F.exists(i, j, level) && is_admissible.exists(i, j, level)) {  // Fill-in to this level
        if (is_admissible(i, j, level)) {
          S(i, j, level) += matmul(matmul(U(i, level), F(i, j, level), true), U(j, level));
        }
        else {
          assert(D(i, j, level).rows == F(i, j, level).rows);
          assert(D(i, j, level).cols == F(i, j, level).cols);
          D(i, j, level) += F(i, j, level);
        }
      }
    }
  }
}

void SymmetricH2::propagate_upper_level_fill_in(const int64_t level,
                                                RowColLevelMap<Matrix>& F) {
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
          const int64_t ncols = U(j_c1, level).cols + U(j_c2, level).cols;
          Matrix fill_in(nrows, ncols);
          auto fill_in_splits = fill_in.split(vec{U(i_c1, level).cols},
                                              vec{U(j_c1, level).cols});
          if (F.exists(i_c1, j_c1, level)) {
            matmul(matmul(U(i_c1, level), F(i_c1, j_c1, level), true, false),
                   U(j_c1, level), fill_in_splits[0], false, false, 1, 0);
          }
          if (F.exists(i_c1, j_c2, level)) {
            matmul(matmul(U(i_c1, level), F(i_c1, j_c2, level), true, false),
                   U(j_c2, level), fill_in_splits[1], false, false, 1, 0);
          }
          if (F.exists(i_c2, j_c1, level)) {
            matmul(matmul(U(i_c2, level), F(i_c2, j_c1, level), true, false),
                   U(j_c1, level), fill_in_splits[2], false, false, 1, 0);
          }
          if (F.exists(i_c2, j_c2, level)) {
            matmul(matmul(U(i_c2, level), F(i_c2, j_c2, level), true, false),
                   U(j_c2, level), fill_in_splits[3], false, false, 1, 0);
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
  for (int64_t block = 0; block < nblocks; ++block) {  // Parallel OK
    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    auto diagonal_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = block+1; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        auto D_splits = D(i, block, level).split(vec{D(i, block, level).rows - U(i, level).cols},
                                                 vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = block+1; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        auto D_splits = D(block, j, level).split(vec{diag_row_split},
                                                 vec{D(block, j, level).cols - U(j, level).cols});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }
    // TRSM with co blocks on the row
    for (int64_t j = 0; j < nblocks; ++j) {
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
  }
}

void SymmetricH2::factorize() {
  int64_t level = height;
  RowColLevelMap<Matrix> F;

  for (; level > 0; --level) {
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

    pre_compute_fill_in(level, F);
    form_new_cluster_bases(level, F);
    add_fill_in_contribution(level, F);
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
    // Multiply with U_F from right
    for (int64_t j = 0; j < nblocks; j++) {  // Parallel OK
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
  } // for (; level > 0; --level)

  // Factorize remaining root level
  ldl(D(0, 0, level));
}

// Permute the vector forward and return the offset at which the new vector begins.
int64_t SymmetricH2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) {
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
int64_t SymmetricH2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) {
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

void SymmetricH2::solve_forward_level(Matrix& x_level, int64_t level) {
  const int64_t nblocks = level_blocks[level];
  std::vector<int64_t> row_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    row_offsets.push_back(nrows + D(i, i, level).rows);
    nrows += D(i, i, level).rows;
  }
  auto x_level_split = x_level.split(row_offsets, vec{});

  // Multiply with (U_F)^T beforehand
  // for (int64_t block = 0; block < nblocks; ++block) {
  //   Matrix U_F = prepend_complement_basis(U(block, level));
  //   x_level_split[block] = matmul(U_F, x_level_split[block], true);
  // }

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

void SymmetricH2::solve_backward_level(Matrix& x_level, int64_t level) {
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
  // Multiply with U_F at the end
  // for (int64_t block = nblocks-1; block >= 0; --block) {
  //   Matrix U_F = prepend_complement_basis(U(block, level));
  //   x_level_split[block] = matmul(U_F, x_level_split[block]);
  // }
}

void SymmetricH2::solve_diagonal_level(Matrix& x_level, int64_t level) {
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

Matrix SymmetricH2::solve(const Matrix& b, int64_t _level) {
  Matrix x(b);
  int64_t level = _level;
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
  for (; level <= _level; ++level) {
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
  const int64_t rank = argc > 4 ? atol(argv[4]) : 50;
  const double admis = argc > 5 ? atof(argv[5]) : 1.0;

  // Specify admissibility type
  // diagonal_admis: Admissibility based on absolute distance from diagonal block
  // geometry_admis: Admissibility based on particles' geometric distance
  const std::string admis_kind = argc > 6 ? std::string(argv[6]) : "geometry_admis";

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

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1e-3, 1.);
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

  const int64_t oversampling = 5;
  const int64_t sample_size = rank > 0 ? rank + oversampling : 0;
  Hatrix::Matrix rand = Hatrix::generate_random_matrix(N, sample_size);
  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, nleaf, accuracy, rank, admis, admis_kind, matrix_type, rand);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();  
  double construct_error = A.construction_absolute_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " nleaf=" << nleaf
            << " accuracy=" << accuracy
            << " rank=" << rank
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
            << " construct_min_rank=" << A.min_rank
            << " construct_max_rank=" << A.max_rank
            << " construct_time=" << construct_time
            << std::scientific
            << " construct_error=" << construct_error
            << std::defaultfloat
            << std::endl;

  const auto start_factor = std::chrono::system_clock::now();
  A.factorize();
  const auto stop_factor = std::chrono::system_clock::now();
  const double factor_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_factor - start_factor).count();

  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix b = Hatrix::matmul(Adense, x);
  const auto solve_start = std::chrono::system_clock::now();
  Hatrix::Matrix x_solve = A.solve(b, A.height);
  const auto solve_stop = std::chrono::system_clock::now();
  const double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (solve_stop - solve_start).count();
  double solve_error = Hatrix::norm(x_solve - x);

  std::cout << "factor_min_rank=" << A.min_rank
            << " factor_max_rank=" << A.max_rank
            << " factor_time=" << factor_time
            << " solve_time=" << solve_time
            << " solve_error=" << solve_error
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
