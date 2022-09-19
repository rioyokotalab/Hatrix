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
  bool use_rel_acc;
  double ID_tolerance;
  int64_t max_rank;
  double admis;
  int64_t height;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  std::vector<int64_t> level_blocks;
  RowColMap<std::vector<int64_t>> skeleton_rows;

 private:
  void initialize_geometry_admissibility(const Domain& domain);

  int64_t get_block_size(const Domain& domain, const int64_t node, const int64_t level) const;
  bool row_has_admissible_blocks(const int64_t row, const int64_t level) const;

  void generate_row_cluster_basis(const Domain& domain);
  void generate_coupling_matrices(const Domain& domain);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;

 public:
  SymmetricH2(const Domain& domain,
              const int64_t N, const int64_t leaf_size,
              const double accuracy, const bool use_rel_acc,
              const int64_t max_rank, const double admis);

  int64_t get_basis_min_rank() const;
  int64_t get_basis_max_rank() const;
  double construction_error(const Domain& domain) const;
  void print_structure(const int64_t level) const;
  void print_ranks() const;
  double low_rank_block_ratio() const;
};

void SymmetricH2::initialize_geometry_admissibility(const Domain& domain) {
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

int64_t SymmetricH2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
  const auto idx = domain.get_cell_idx(node, level);
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

void SymmetricH2::generate_row_cluster_basis(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (row_has_admissible_blocks(node, level)) {
        const auto idx = domain.get_cell_idx(node, level);
        const auto& cell = domain.cells[idx];
        std::vector<int64_t> skeleton;
        if (level == height) {
          // Leaf level: use all bodies as skeleton
          skeleton = cell.get_bodies();
        }
        else {
          // Non-leaf level: gather children's skeleton
          const auto& child1 = domain.cells[cell.child];
          const auto& child2 = domain.cells[cell.child + 1];
          const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
          const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
          skeleton.insert(skeleton.end(), child1_skeleton.begin(), child1_skeleton.end());
          skeleton.insert(skeleton.end(), child2_skeleton.begin(), child2_skeleton.end());
        }
        Matrix adm_block_row = generate_p2p_matrix(domain, skeleton, cell.sample_farfield);
        // ID compress
        Matrix U_node;
        std::vector<int64_t> ipiv_rows;
        std::tie(U_node, ipiv_rows) = error_id_row(adm_block_row, ID_tolerance, use_rel_acc);
        int64_t rank = U_node.cols;
        // Convert ipiv to node skeleton rows to be used by parent
        std::vector<int64_t> skel_rows;
        skel_rows.reserve(rank);
        for (int64_t i = 0; i < rank; i++) {
          skel_rows.push_back(skeleton[ipiv_rows[i]]);
        }
        U.insert(node, level, std::move(U_node));
        skeleton_rows.insert(node, level, std::move(skel_rows));
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
          D.insert(i, j, level, generate_p2p_matrix(domain, i, j, level));
        }
        // Admissible blocks
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const auto& skeleton_i = skeleton_rows(i, level);
          const auto& skeleton_j = skeleton_rows(j, level);
          S.insert(i, j, level, generate_p2p_matrix(domain, skeleton_i, skeleton_j));
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
                         const double accuracy, const bool use_rel_acc,
                         const int64_t max_rank, const double admis)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis) {
  // Set ID tolerance to be smaller than desired accuracy, based on HiDR paper source code
  // https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c#L859
  ID_tolerance = accuracy * 1e-1;
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

double SymmetricH2::construction_error(const Domain& domain) const {
  double dense_norm = 0;
  double diff_norm = 0;
  // Inadmissible blocks (only at leaf level)
  for (int64_t i = 0; i < level_blocks[height]; i++) {
    for (int64_t j = 0; j < level_blocks[height]; j++) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        const Matrix D_ij = Hatrix::generate_p2p_matrix(domain, i, j, height);
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
          const Matrix D_ij = Hatrix::generate_p2p_matrix(domain, i, j, level);
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
  const double admis = argc > 5 ? atof(argv[5]) : 1.0;
  int64_t sample_self_size = argc > 6 ? atol(argv[6]) : 2 * leaf_size;
  int64_t sample_far_size = argc > 7 ? atol(argv[7]) : 5 * sample_self_size;

  // Specify bodies sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net
  const int64_t sampling_algo = argc > 8 ? atol(argv[8]) : 0;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 9 ? atol(argv[9]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 10 ? atol(argv[10]) : 0;
  const int64_t ndim  = argc > 11 ? atol(argv[11]) : 2;
  const bool use_rel_acc = argc > 12 ? (atol(argv[12]) == 1) : false;

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

  domain.build_tree(leaf_size);
  domain.build_interactions(admis);
  const auto start_sample = std::chrono::system_clock::now();
  // if (sampling_algo == 3) {
  //   const auto r = domain.adaptive_anchor_grid_size(Hatrix::kernel_function, leaf_size, admis,
  //                                                   accuracy * 1e-1, accuracy * 1e-2);
  //   sample_self_size = r;
  //   sample_far_size = r > 3 ? std::max(r + 3, (int64_t)10) : r + 3;
  // }
  const int64_t initial_far_sp = 1;
  domain.build_sample_bodies(sample_self_size, sample_far_size, sampling_algo, initial_far_sp);
  const auto stop_sample = std::chrono::system_clock::now();
  const double sample_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_sample - start_sample).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error = A.construction_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " leaf_size=" << leaf_size
            << " accuracy=" << accuracy
            << " acc_type=" << (use_rel_acc ? "rel_err" : "abs_err")
            << " max_rank=" << max_rank
            << " admis=" << admis << std::setw(3)
            << " sampling_algo=" << sampling_algo_name
            << " sample_self_size=" << sample_self_size
            << " sample_far_size=" << sample_far_size
            << " sample_farfield_max_size=" << domain.get_max_farfield_size()
            << " compress_alg=" << "ID"
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
