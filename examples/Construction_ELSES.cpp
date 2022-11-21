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

#include "nlohmann/json.hpp"

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

using vec = std::vector<int64_t>;

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
  double ID_tolerance;
  int64_t max_rank;
  double admis;
  int64_t matrix_type;
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

  void generate_row_cluster_basis(const Domain& domain);
  void generate_coupling_matrices(const Domain& domain);

  Matrix get_Ubig(const int64_t node, const int64_t level) const;
  void fill_JSON(const Domain& domain,
                 const int64_t i, const int64_t j,
                 const int64_t level, nlohmann::json& json) const;

 public:
  SymmetricH2(const Domain& domain,
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

int64_t SymmetricH2::get_block_size(const Domain& domain, const int64_t node, const int64_t level) const {
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

void SymmetricH2::generate_row_cluster_basis(const Domain& domain) {
  for (int64_t level = height; level > 0; level--) {
    const auto num_nodes = level_blocks[level];
    for (int64_t node = 0; node < num_nodes; node++) {
      if (row_has_admissible_blocks(node, level)) {
        const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
        const auto idx = domain.get_cell_idx(node, node_level);
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
        // Key to order N complexity is here:
        // The size of adm_block_row is always constant: ID_skeleton_size x sample_far_size
        Matrix adm_block_row = generate_p2p_matrix(domain, skeleton, cell.sample_farfield);
        // ID compress
        Matrix U_node;
        std::vector<int64_t> ipiv_rows;
        std::tie(U_node, ipiv_rows) = error_id_row(adm_block_row, ID_tolerance, use_rel_acc);
        int64_t rank = U_node.cols;
        if (max_rank > 0 && rank > max_rank) {
          // Truncate to max_rank
          U_node.shrink(U_node.rows, max_rank);
          rank = max_rank;
        }
        // Convert ipiv to node skeleton rows to be used by parent
        std::vector<int64_t> skel_rows;
        skel_rows.reserve(rank);
        for (int64_t i = 0; i < rank; i++) {
          skel_rows.push_back(skeleton[ipiv_rows[i]]);
        }
        // Multiply U with child R
        if (level < height) {
          const auto& child1 = domain.cells[cell.child];
          const auto& child2 = domain.cells[cell.child + 1];
          const auto& child1_skeleton = skeleton_rows(child1.block_index, child1.level);
          const auto& child2_skeleton = skeleton_rows(child2.block_index, child2.level);
          auto U_node_splits = U_node.split(vec{(int64_t)child1_skeleton.size()}, vec{});
          triangular_matmul(R_row(child1.block_index, child1.level), U_node_splits[0],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(R_row(child2.block_index, child2.level), U_node_splits[1],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
        }
        // Orthogonalize basis with QR
        Matrix Q(U_node.rows, U_node.cols);
        Matrix R(U_node.cols, U_node.cols);
        qr(U_node, Q, R);
        U.insert(node, level, std::move(Q));
        R_row.insert(node, level, std::move(R));
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
          const auto node_level = matrix_type == BLR2_MATRIX ? domain.tree_height : level;
          D.insert(i, j, level, generate_p2p_matrix(domain, i, j, node_level));
        }
        // Admissible blocks
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          const auto& skeleton_i = skeleton_rows(i, level);
          const auto& skeleton_j = skeleton_rows(j, level);
          Matrix skeleton_matrix = generate_p2p_matrix(domain, skeleton_i, skeleton_j);
          // Multiply with R from left and right
          triangular_matmul(R_row(i, level), skeleton_matrix,
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(R_row(j, level), skeleton_matrix,
                            Hatrix::Right, Hatrix::Upper, true, false, 1);
          S.insert(i, j, level, std::move(skeleton_matrix));
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
                         const int64_t max_rank, const double admis,
                         const int64_t matrix_type)
    : N(N), leaf_size(leaf_size), accuracy(accuracy),
      use_rel_acc(use_rel_acc), max_rank(max_rank), admis(admis), matrix_type(matrix_type) {
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
    }
    else {
      if (level == height) {
        json["type"] = "Dense";
      }
      else {
        json["type"] = "Hierarchical";
        json["children"] = {};
        if (matrix_type == BLR2_MATRIX) {
          for (int64_t i_child = 0; i_child < level_blocks[height]; i_child++) {
            std::vector<nlohmann::json> row(level_blocks[height]);
            int64_t j_pos = 0;
            for (int64_t j_child = 0; j_child <= level_blocks[height]; j_child++) {
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

} // namespace Hatrix

int main(int argc, char ** argv) {
  const std::string file_name = argc > 1 ? std::string(argv[1]) : "";
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-8;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const double admis = argc > 5 ? atof(argv[5]) : 3;

  // Specify sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net method
  const int64_t sampling_algo = argc > 6 ? atol(argv[6]) : 3;
  // Specify maximum number of sample points that represents a cluster
  // For anchor-net method: size of grid
  int64_t sample_self_size = argc > 7 ? atol(argv[7]) :
                             (sampling_algo == 3 ? 30 : 10 * leaf_size);
  // Specify maximum number of sample points that represents a cluster's far-field
  // For anchor-net method: size of grid
  int64_t sample_far_size = argc > 8 ? atol(argv[8]) :
                            (sampling_algo == 3 ? sample_self_size + 5 : 10 * leaf_size + 10);

  // Use relative or absolute error threshold
  const bool use_rel_acc = argc > 9 ? (atol(argv[9]) == 1) : false;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 10 ? atol(argv[10]) : 1;

  // Whether reading sorted ELSES geometry or no
  const int64_t read_sorted_bodies = argc > 11 ? atol(argv[11]) : 0;

  // Output H2-Matrix structure as JSON file
  // Empty string: do not write JSON file
  const std::string out_filename = argc > 12 ? std::string(argv[12]) : "";

  Hatrix::Context::init();

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

  std::string kernel_name = "ELSES-kernel";
  Hatrix::set_kernel_function(Hatrix::ELSES_dense_input);
  std::string geom_name = file_name;

  Hatrix::Domain domain(0, 3);
  if (read_sorted_bodies == 1) {
    const auto buckets = domain.read_bodies_ELSES_sorted(file_name + "_kmeans" +
                                                         std::to_string(leaf_size) + ".xyz");
    domain.build_tree_from_kmeans_ordering(leaf_size, buckets);
  }
  else {
    domain.read_bodies_ELSES(file_name + ".xyz");
    domain.build_tree(leaf_size);
  }
  domain.build_interactions(admis);
  domain.read_p2p_matrix_ELSES(file_name + ".dat");
  const int64_t N = domain.N;

  const auto start_sample = std::chrono::system_clock::now();
  domain.build_sample_bodies(sample_self_size, sample_far_size, sampling_algo, true);
  const auto stop_sample = std::chrono::system_clock::now();
  const double sample_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_sample - start_sample).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, leaf_size, accuracy, use_rel_acc, max_rank, admis, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error = A.construction_error(domain);
  double lr_ratio = A.low_rank_block_ratio();
  A.print_structure(A.height);

  if (out_filename.length() > 0) {
    A.write_JSON(domain, out_filename);
  }

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
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " height=" << A.height
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.get_basis_min_rank()
            << " construct_max_rank=" << A.get_basis_max_rank()
            << " sample_time=" << sample_time
            << " construct_time=" << construct_time
            << " construct_error=" << std::scientific << construct_error
            << " out_filename=" << out_filename
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
