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

using vec = std::vector<int64_t>;

namespace Hatrix {

class SymmetricH2 {
 public:
  int64_t N, leaf_size;
  double accuracy;
  double ID_tolerance;
  int64_t max_rank;
  double admis;
  int64_t height;
  RowLevelMap U, R_row;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  std::vector<int64_t> level_blocks;
  RowColMap<std::vector<int64_t>> skeleton_rows;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;

  Matrix generate_admissible_block_row(const Domain& domain,
                                       const int64_t node, const int64_t level,
                                       const std::vector<int64_t>& node_rows) const;
  void generate_row_cluster_basis(const Domain& domain);
  void generate_coupling_matrices(const Domain& domain);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const int64_t max_rank, const double admis);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double construction_absolute_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;
};

void SymmetricH2::initialize_geometry_admissibility(const Domain& domain) {
  height = domain.tree_height;
  level_blocks.assign(height + 1, 0);
  for (const auto& cell: domain.cells) {
    const auto level = cell.level;
    const auto i = cell.index;
    level_blocks[level]++;
    // Near interaction list: inadmissible dense blocks
    for (const auto near_loc: cell.near_list) {
      const auto j_near = domain.cells[near_loc].index;
      is_admissible.insert(i, j_near, level, false);
    }
    // Far interaction list: admissible low-rank blocks
    for (const auto far_loc: cell.far_list) {
      const auto j_far = domain.cells[far_loc].index;
      is_admissible.insert(i, j_far, level, true);
    }
  }
}

int64_t SymmetricH2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
  const auto loc = domain.get_cell_loc(node, level);
  return domain.cells[loc].nbodies;
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

Matrix SymmetricH2::generate_admissible_block_row(const Domain& domain,
                                                  const int64_t node, const int64_t level,
                                                  const std::vector<int64_t>& node_rows) const {
  const auto loc = domain.get_cell_loc(node, level);
  const auto& source = domain.cells[loc];
  return generate_p2p_matrix(domain, node_rows, source.sample_farfield,
                             source.body, 0);
}

void SymmetricH2::generate_row_cluster_basis(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (row_has_admissible_blocks(node, level)) {
        std::vector<int64_t> node_rows;
        if (level == height) {
          // Leaf level: use all bodies as skeleton
          const auto block_size = get_block_size(domain, node, level);
          node_rows.reserve(block_size);
          for (int64_t i = 0; i < block_size; i++) {
            node_rows.push_back(i);
          }
        }
        else {
          // Non-leaf level: concat children's skeleton
          const auto child_level = level + 1;
          const auto child1 = node * 2;
          const auto child2 = node * 2 + 1;
          const auto& child1_skeleton = skeleton_rows(child1, child_level);
          const auto& child2_skeleton = skeleton_rows(child2, child_level);
          node_rows.reserve(child1_skeleton.size() +
                            child2_skeleton.size());
          for (int64_t i = 0; i < child1_skeleton.size(); i++) {
            node_rows.push_back(child1_skeleton[i]);
          }
          const auto offset = get_block_size(domain, child1, child_level);
          for (int64_t i = 0; i < child2_skeleton.size(); i++) {
            node_rows.push_back(offset + child2_skeleton[i]);
          }
        }
        Matrix adm_block_row =
            generate_admissible_block_row(domain, node, level,node_rows);
        // SVD to get column basis
        Matrix Ui, Si, Vi;
        int64_t rank;
        std::tie(Ui, Si, Vi, rank) = error_svd(adm_block_row, ID_tolerance, false, true);
        // ID to get skeleton rows
        Matrix UxS = matmul(Ui, Si);
        Matrix U_node;
        std::vector<int64_t> skel_rows;
        std::tie(U_node, skel_rows) = truncated_id_row(UxS, rank);
        // Construct global skeleton row indices
        std::vector<int64_t> skel_node;
        skel_node.reserve(rank);
        for (int64_t i = 0; i < rank; i++) {
          skel_node.push_back(node_rows[skel_rows[i]]);
        }
        // Multiply with child R if necessary
        if (level < height) {
          const auto child_level = level + 1;
          const auto child1 = node * 2;
          const auto child2 = node * 2 + 1;
          auto U_node_splits = U_node.split(vec{U(child1, child_level).cols}, vec{});
          triangular_matmul(R_row(child1, child_level), U_node_splits[0],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(R_row(child2, child_level), U_node_splits[1],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
        }
        // Orthogonalize basis
        Matrix Q(U_node.rows, U_node.cols);
        Matrix R(U_node.cols, U_node.cols);
        qr(U_node, Q, R);
        U.insert(node, level, std::move(Q));
        R_row.insert(node, level, std::move(R));
        skeleton_rows.insert(node, level, std::move(skel_node));
      }
    }
  }
}

void SymmetricH2::generate_coupling_matrices(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t i = 0; i < num_nodes; i++) {
      for (int64_t j = 0; j < num_nodes; j++) {
        // Inadmissible leaf blocks
        if (level == height &&
            is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
          D.insert(i, j, level,
                   generate_p2p_matrix(domain, i, j, level));
        }
        // Admissible blocks
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const auto& skeleton_i = skeleton_rows(i, level);
          const auto& skeleton_j = skeleton_rows(j, level);
          const auto i_offset = domain.cells[domain.get_cell_loc(i, level)].body;
          const auto j_offset = domain.cells[domain.get_cell_loc(j, level)].body;
          Matrix skeleton = generate_p2p_matrix(domain, skeleton_i, skeleton_j,
                                                i_offset, j_offset);
          triangular_matmul(R_row(i, level), skeleton, Hatrix::Left, Hatrix::Upper,
                            false, false, 1.);
          triangular_matmul(R_row(j, level), skeleton, Hatrix::Right, Hatrix::Upper,
                            true, false, 1.);
          S.insert(i, j, level, std::move(skeleton));
        }
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
                         const double accuracy, const int64_t max_rank, const double admis)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      max_rank(max_rank), admis(admis) {
  // Set ID tolerance to be smaller than desired accuracy, based on HiDR paper source code
  // https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c#L859
  ID_tolerance = accuracy * 1e-2;
  initialize_geometry_admissibility(domain);
  generate_row_cluster_basis(domain);
  generate_coupling_matrices(domain);
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
        const Matrix actual = Hatrix::generate_p2p_matrix(domain, i, j, height);
        const Matrix expected = D(i, j, height);
        const auto diff = norm(expected - actual);
        error += diff * diff;
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
              Hatrix::generate_p2p_matrix(domain, i, j, level);
          const auto diff = norm(expected_matrix - actual_matrix);
          error += diff * diff;
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

} // namespace Hatrix

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-5;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const int64_t sample_size = argc > 5 ? atol(argv[5]) : 100;
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
  const auto start_sample = std::chrono::system_clock::now();
  domain.select_sample_bodies(2 * leaf_size, sample_size, 2);
  const auto stop_sample = std::chrono::system_clock::now();
  const double sample_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_sample - start_sample).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, max_rank, admis);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();  
  double construct_error = A.construction_absolute_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " max_rank=" << max_rank
            << " sample_size=" << sample_size
            << " compress_alg=" << "ID"
            << " admis=" << admis << std::setw(3)
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " height=" << A.height
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " sample_time=" << sample_time
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
