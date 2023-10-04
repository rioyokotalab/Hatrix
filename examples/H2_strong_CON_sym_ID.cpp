#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"
#include "Domain.hpp"
#include "admissibility.hpp"
#include "functions.hpp"

using namespace Hatrix;
using vec = std::vector<int64_t>;

/*
  Order N H2-Construction with ID and hierarchical sample points.
*/

namespace {

std::vector<int64_t> get_skeleton_particles(const SymmetricSharedBasisMatrix& A,
                                            const Domain& domain,
                                            const int64_t i, const int64_t level) {
  std::vector<int64_t> skeleton;
  if (level == A.max_level) {
    // Leaf level: use all bodies as skeleton
    const auto& Ci = domain.cells[domain.get_cell_index(i, level)];
    skeleton = Ci.get_bodies();
  }
  else {
    // Non-leaf level: gather children's multipoles
    const auto child_level = level + 1;
    const auto child1 = 2 * i + 0;
    const auto child2 = 2 * i + 1;
    const auto& child1_multipoles = A.multipoles(child1, child_level);
    const auto& child2_multipoles = A.multipoles(child2, child_level);
    skeleton.insert(skeleton.end(), child1_multipoles.begin(), child1_multipoles.end());
    skeleton.insert(skeleton.end(), child2_multipoles.begin(), child2_multipoles.end());
  }
  return skeleton;
}

void generate_cluster_bases(SymmetricSharedBasisMatrix& A,
                            const Domain& domain,
                            const Admissibility::CellInteractionLists& interactions,
                            const double err_tol, const int64_t max_rank,
                            const bool is_rel_tol) {
  // Bottom up pass
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    #pragma omp parallel for
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      const auto ii = domain.get_cell_index(i, level);
      if (interactions.far_particles[ii].size() > 0) {  // If row has admissible blocks
        const auto skeleton = get_skeleton_particles(A, domain, i, level);
        // Key to order N complexity is here:
        // The size of far_blocks is always constant: rank x sample_size
        Matrix far_blocks = generate_p2p_matrix(domain, skeleton, interactions.far_particles[ii]);
        // LRA with ID
        Matrix Ui;
        std::vector<int64_t> ipiv_rows;
        std::tie(Ui, ipiv_rows) = error_id_row(far_blocks, err_tol, is_rel_tol);
        int64_t rank = Ui.cols;
        // Fixed-accuracy with bounded rank
        rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
        Ui.shrink(Ui.rows, rank);
        // Construct right factor (skeleton rows) from Interpolative Decomposition (ID)
        Matrix SV(rank, far_blocks.cols);
        for (int64_t r = 0; r < rank; r++) {
          for (int64_t c = 0; c < far_blocks.cols; c++) {
            SV(r, c) = far_blocks(ipiv_rows[r], c);
          }
        }
        Matrix Si(rank, rank);
        Matrix Vi(rank, SV.cols);
        rq(SV, Si, Vi);
        // Convert ipiv to multipoles to be used by parent
        std::vector<int64_t> multipoles_i;
        multipoles_i.reserve(rank);
        for (int64_t j = 0; j < rank; j++) {
          multipoles_i.push_back(skeleton[ipiv_rows[j]]);
        }
        // Multiply U with child R
        if (level < A.max_level) {
          const auto child_level = level + 1;
          const auto child1 = 2 * i + 0;
          const auto child2 = 2 * i + 1;
          const auto& child1_multipoles = A.multipoles(child1, child_level);
          auto Ui_splits = Ui.split(vec{(int64_t)child1_multipoles.size()}, {});
          triangular_matmul(A.R_row(child1, child_level), Ui_splits[0],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
          triangular_matmul(A.R_row(child2, child_level), Ui_splits[1],
                            Hatrix::Left, Hatrix::Upper, false, false, 1);
        }
        // Orthogonalize basis with QR
        Matrix Q(Ui.rows, Ui.rows);
        Matrix R(Ui.rows, rank);
        Matrix Ui_copy(Ui);
        qr(Ui_copy, Q, R);
        // Separate Q into original and complement part
        auto Q_splits = Q.split(vec{}, vec{rank});
        Matrix Qo(Q_splits[0], true);
        Matrix Qc = rank < Ui.rows ? Matrix(Q_splits[1], true) : Matrix(Ui.rows, 0);
        R.shrink(rank, rank);
        // Insert
        #pragma omp critical
        {
          A.U.insert(i, level, std::move(Qo));
          A.Uc.insert(i, level, std::move(Qc));
          A.US_row.insert(i, level, matmul(Ui, Si));  // for ULV update basis operation
          A.R_row.insert(i, level, std::move(R));
          A.multipoles.insert(i, level, std::move(multipoles_i));
        }
      }
    }
  }
}

void generate_far_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                    const Domain& domain, const int64_t level) {
  #pragma omp parallel for
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    for (int64_t j: A.admissible_cols(i, level)) {
      Matrix Sij = generate_p2p_matrix(domain, A.multipoles(i, level), A.multipoles(j, level));
      // Multiply with R from left and right
      triangular_matmul(A.R_row(i, level), Sij,
                        Hatrix::Left, Hatrix::Upper, false, false, 1);
      triangular_matmul(A.R_row(j, level), Sij,
                        Hatrix::Right, Hatrix::Upper, true, false, 1);
      #pragma omp critical
      {
        A.S.insert(i, j, level, std::move(Sij));
      }
    }
  }
}

void generate_far_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                    const Domain& domain) {
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    generate_far_coupling_matrices(A, domain, level);
  }
}

void generate_near_coupling_matrices(SymmetricSharedBasisMatrix& A,
                                     const Domain& domain) {
  const int64_t level = A.max_level;  // Only generate inadmissible leaf blocks
  #pragma omp parallel for
  for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
    for (int64_t j: A.inadmissible_cols(i, level)) {
      Matrix Dij = generate_p2p_matrix(domain,
                                       domain.get_cell_index(i, level),
                                       domain.get_cell_index(j, level));
      #pragma omp critical
      {
        A.D.insert(i, j, level, std::move(Dij));
      }
    }
  }
}

void construct_H2(SymmetricSharedBasisMatrix& A,
                  Admissibility::CellInteractionLists& interactions,
                  const Domain& domain, const double admis,
                  const double err_tol, const int64_t max_rank,
                  const int64_t sampling_algo,
                  const int64_t sample_local_size, const int64_t sample_far_size,
                  const bool is_rel_tol = false) {
  // Initialize cell interactions for admissibility
  Admissibility::build_cell_interactions(interactions, domain, admis);
  Admissibility::assemble_farfields_sample(interactions, domain,
                                           sampling_algo, sample_local_size, sample_far_size);
  // Initialize matrix block structure and admissibility
  Admissibility::init_block_structure(A, domain);
  Admissibility::init_geometry_admissibility(A, interactions, domain, admis);
  // Generate cluster bases and coupling matrices
  generate_cluster_bases(A, domain, interactions, err_tol, max_rank, is_rel_tol);
  generate_far_coupling_matrices(A, domain);
  generate_near_coupling_matrices(A, domain);
}

Matrix get_Ubig(const SymmetricSharedBasisMatrix& A,
                const int64_t i, const int64_t level) {
  if (level == A.max_level) {
    return A.U(i, level);
  }
  const int64_t child1 = i * 2 + 0;
  const int64_t child2 = i * 2 + 1;
  const Matrix Ubig_child1 = get_Ubig(A, child1, level + 1);
  const Matrix Ubig_child2 = get_Ubig(A, child2, level + 1);

  const int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, A.U(i, level).cols);
  auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
  auto U_splits = A.U(i, level).split(vec{Ubig_child1.cols}, vec{});
  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

double construction_error(const SymmetricSharedBasisMatrix& A,
                          const Domain& domain, const bool relative = false) {
  double dense_norm = 0;
  double diff_norm = 0;
  // Inadmissible blocks (only at leaf level)
  {
    const int64_t level = A.max_level;
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: A.inadmissible_cols(i, level)) {
        const Matrix Dij = generate_p2p_matrix(domain,
                                               domain.get_cell_index(i, level),
                                               domain.get_cell_index(j, level));
        const Matrix& Aij = A.D(i, j, level);
        const auto d_norm = norm(Dij);
        const auto diff = norm(Aij - Dij);
        dense_norm += d_norm * d_norm;
        diff_norm += diff * diff;
      }
    }
  }
  // Admissible blocks
  for (int64_t level = A.max_level; level >= A.min_adm_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      for (int64_t j: A.admissible_cols(i, level)) {
        const Matrix Dij = generate_p2p_matrix(domain,
                                               domain.get_cell_index(i, level),
                                               domain.get_cell_index(j, level));
        const Matrix Ubig = get_Ubig(A, i, level);
        const Matrix Vbig = get_Ubig(A, j, level);
        const Matrix Aij = matmul(matmul(Ubig, A.S(i, j, level)), Vbig, false, true);
        const auto d_norm = norm(Dij);
        const auto diff = norm(Aij - Dij);
        dense_norm += d_norm * d_norm;
        diff_norm += diff * diff;
      }
    }
  }
  return (relative ? std::sqrt(diff_norm / dense_norm) : std::sqrt(diff_norm));
}

int64_t get_basis_min_rank(const SymmetricSharedBasisMatrix& A,
                           int64_t level_begin = 0,
                           int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  int64_t min_rank = std::numeric_limits<int64_t>::max();
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        min_rank = std::min(min_rank, A.U(i, level).cols);
      }
    }
  }
  return (min_rank == std::numeric_limits<int64_t>::max() ? -1 : min_rank);
}

int64_t get_basis_max_rank(const SymmetricSharedBasisMatrix& A,
                           int64_t level_begin = 0,
                           int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  int64_t max_rank = -1;
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        max_rank = std::max(max_rank, A.U(i, level).cols);
      }
    }
  }
  return max_rank;
}

double get_basis_avg_rank(const SymmetricSharedBasisMatrix& A,
                          int64_t level_begin = 0,
                          int64_t level_end = 0) {
  if (level_begin == 0) level_begin = A.min_level;
  if (level_end == 0)   level_end = A.max_level;
  double sum_rank = 0;
  double num_bases = 0;
  for (int64_t level = level_begin; level <= level_end; level++) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        sum_rank += static_cast<double>(A.U(i, level).cols);
        num_bases += 1.;
      }
    }
  }
  return sum_rank / num_bases;
}

// Return memory usage (in bytes)
int64_t get_memory_usage(const SymmetricSharedBasisMatrix& A) {
  int64_t mem = 0;
  for (int64_t level = A.max_level; level >= A.min_level; level--) {
    for (int64_t i = 0; i < A.level_nblocks[level]; i++) {
      if (A.U.exists(i, level)) {
        mem += A.U(i, level).memory_used();
      }
      if (A.US_row.exists(i, level)) {
        mem += A.US_row(i, level).memory_used();
      }
      for (auto j: A.inadmissible_cols(i, level)) {
        if (A.D.exists(i, j, level)) {
          mem += A.D(i, j, level).memory_used();
        }
      }
      for (auto j: A.admissible_cols(i, level)) {
        if (A.S.exists(i, j, level)) {
          mem += A.S(i, j, level).memory_used();
        }
      }
    }
  }
  return mem;
}

}  // namespace

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  // err_tol == 0 means fixed rank
  const double err_tol = argc > 3 ? atof(argv[3]) : 1.e-8;
  // Use relative or absolute error threshold for LRA
  const bool is_rel_tol = argc > 4 ? (atol(argv[4]) == 1) : false;
  // Fixed accuracy with bounded rank
  const int64_t max_rank = argc > 5 ? atol(argv[5]) : 20;
  const double admis = argc > 6 ? atof(argv[6]) : 2;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: ELSES Dense Matrix
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  // 4: Random Uniform Grid
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim  = argc > 9 ? atol(argv[9]) : 1;

  // Specify sampling technique
  // 0: Choose bodies with equally spaced indices
  // 1: Choose bodies random indices
  // 2: Farthest Point Sampling
  // 3: Anchor Net method
  const int64_t sampling_algo = argc > 10 ? atol(argv[10]) : 3;
  // Specify maximum number of sample points that represents a cluster
  // For anchor-net method: size of grid
  int64_t sample_local_size = argc > 11 ? atol(argv[11]) :
                              (ndim == 1 ? 10 * leaf_size : 30);
  // Specify maximum number of sample points that represents a cluster's far-field
  // For anchor-net method: size of grid
  int64_t sample_far_size = argc > 12 ? atol(argv[12]) :
                            (ndim == 1 ? 10 * leaf_size + 10: sample_local_size + 5);

  // ELSES Input Files
  const std::string file_name = argc > 13 ? std::string(argv[13]) : "";

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
      kernel_name = "ELSES-dense-file";
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
      const auto prefix_end = file_name.find_last_of("/\\");
      geom_name = file_name.substr(prefix_end + 1);
      break;
    }
    case 4: {
      domain.initialize_random_uniform_grid();
      geom_name += "random_uniform_grid";
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
    default: {
      sampling_algo_name = "no_sample";
      break;
    }
  }
  // Pre-processing step for ELSES geometry
  if (geom_type == 3) {
    const int64_t num_atoms_per_molecule = 60;
    const int64_t num_electrons_per_atom = kernel_type == 2 ? 4 : 1;
    const int64_t molecule_size = num_atoms_per_molecule * num_electrons_per_atom;
    assert(file_name.length() > 0);
    domain.read_bodies_ELSES(file_name + ".xyz", num_electrons_per_atom);
    assert(N == domain.N);

    domain.sort_bodies_ELSES(molecule_size);
    domain.build_tree_from_sorted_bodies(leaf_size, std::vector<int64_t>(N / leaf_size, leaf_size));
    if (kernel_type == 2) {
      domain.read_p2p_matrix_ELSES(file_name + ".dat");
    }
  }
  else {
    domain.build_tree(leaf_size);
  }

  SymmetricSharedBasisMatrix A;
  Admissibility::CellInteractionLists interactions;
  const auto start_construct = std::chrono::system_clock::now();
  construct_H2(A, interactions, domain, admis, err_tol, max_rank,
               sampling_algo, sample_local_size, sample_far_size, is_rel_tol);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  // Count maximum farfield size in interaction lists
  int64_t max_farfield_size = -1;
  for (const auto& farfield: interactions.far_particles) {
    max_farfield_size = std::max(max_farfield_size, (int64_t)farfield.size());
  }
  const auto construct_error = construction_error(A, domain, is_rel_tol);
  const auto construct_min_rank = get_basis_min_rank(A);
  const auto construct_max_rank = get_basis_max_rank(A);
  const auto construct_avg_rank = get_basis_avg_rank(A);
  const auto construct_mem = static_cast<double>(get_memory_usage(A)) * 1e-9;

  const std::string err_prefix = (is_rel_tol ? "rel" : "abs");
  printf("N=%" PRId64 " leaf_size=%d %s_err_tol=%.1e max_rank=%d admis=%.2lf kernel=%s geometry=%s\n"
         "sampling_algo=%s sample_local_size=%d sample_far_size=%d max_farfield_size=%" PRId64 "\n"
         "h2_height=%d construct_min_rank=%d construct_max_rank=%d construct_avg_rank=%.2lf "
         "construct_time=%e[ms] construct_mem=%e[GB] construct_%s_err=%e\n",
         N, (int)leaf_size, err_prefix.c_str(), err_tol, (int)max_rank, admis,
         kernel_name.c_str(), geom_name.c_str(),
         sampling_algo_name.c_str(), (int)sample_local_size, (int)sample_far_size, max_farfield_size,
         (int)A.max_level, (int)construct_min_rank, (int)construct_max_rank, construct_avg_rank,
         construct_time, construct_mem, err_prefix.c_str(), construct_error);

  return 0;
}
