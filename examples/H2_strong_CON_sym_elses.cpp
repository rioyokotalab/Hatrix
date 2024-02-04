#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

extern "C"
{
#include "elses.h"
}

using namespace Hatrix;
using vec = std::vector<int64_t>;

/*
  H2-Construction with SVD. Has O(N^2) complexity due to the explicit upper level bases formation
*/

namespace
{

  void generate_cluster_bases(SymmetricSharedBasisMatrix &A, RowLevelMap &Ubig,
                              const Domain &domain, const Admissibility::CellInteractionLists &interactions,
                              const double err_tol, const int64_t max_rank,
                              const bool is_rel_tol)
  {
    // Bottom up pass
    for (int64_t level = A.max_level; level >= A.min_level; level--)
    {
#pragma omp parallel for
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        const auto ii = domain.get_cell_index(i, level);
        if (interactions.far_particles[ii].size() > 0)
        { // If row has admissible blocks
          const auto &Ci = domain.cells[ii];
          Matrix far_blocks = generate_p2p_matrix(domain,
                                                  Ci.get_bodies(),
                                                  interactions.far_particles[ii]);
          if (level == A.max_level)
          {
            // Leaf level: direct SVD
            Matrix Ui, Si, Vi;
            int64_t rank;
            std::tie(Ui, Si, Vi, rank) = error_svd(far_blocks, err_tol, is_rel_tol, false);
            // Fixed-accuracy with bounded rank
            rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
            // Separate U into original and complement part
            auto Ui_splits = Ui.split(vec{}, vec{rank});
            Matrix Uo(Ui_splits[0], true); // Deep-copy
            Matrix Uc = rank < Ui.rows ? Matrix(Ui_splits[1], true) : Matrix(Ui.rows, 0);
            // Save actual basis (Ubig) for upper level bases construction
            Matrix Ubig_i(Uo);
// Insert
#pragma omp critical
            {
              A.U.insert(i, level, std::move(Uo));
              A.Uc.insert(i, level, std::move(Uc));
              A.US_row.insert(i, level, matmul(Ui, Si)); // Save full basis for ULV update basis operation
              Ubig.insert(i, level, std::move(Ubig_i));
            }
          }
          else
          {
            // Non-leaf level: project with children's bases then SVD to generate transfer matrix
            // Note: this assumes balanced binary tree of cells
            const auto child_level = level + 1;
            const auto child1 = 2 * i + 0;
            const auto child2 = 2 * i + 1;
            const auto &Ubig_child1 = Ubig(child1, child_level);
            const auto &Ubig_child2 = Ubig(child2, child_level);
            Matrix proj_far_blocks(Ubig_child1.cols + Ubig_child2.cols, far_blocks.cols);
            auto far_blocks_splits = far_blocks.split(vec{Ubig_child1.rows}, {});
            auto proj_far_blocks_splits = proj_far_blocks.split(vec{Ubig_child1.cols}, {});
            matmul(Ubig_child1, far_blocks_splits[0], proj_far_blocks_splits[0], true, false, 1, 0);
            matmul(Ubig_child2, far_blocks_splits[1], proj_far_blocks_splits[1], true, false, 1, 0);
            Matrix Ui, Si, Vi;
            int64_t rank;
            std::tie(Ui, Si, Vi, rank) = error_svd(proj_far_blocks, err_tol, is_rel_tol, false);
            // Fixed-accuracy with bounded rank
            rank = max_rank > 0 ? std::min(max_rank, rank) : rank;
            // Separate U into original and complement part
            auto Ui_splits = Ui.split(vec{}, vec{rank});
            Matrix Uo(Ui_splits[0], true); // Deep-copy
            Matrix Uc = rank < Ui.rows ? Matrix(Ui_splits[1], true) : Matrix(Ui.rows, 0);
            // Save actual basis (Ubig) for upper level bases construction
            Matrix Ubig_i(Ubig_child1.rows + Ubig_child2.rows, Uo.cols);
            auto Uo_splits = Uo.split(vec{Ubig_child1.cols}, {});
            auto Ubig_i_splits = Ubig_i.split(vec{Ubig_child1.rows}, {});
            matmul(Ubig_child1, Uo_splits[0], Ubig_i_splits[0]);
            matmul(Ubig_child2, Uo_splits[1], Ubig_i_splits[1]);
// Insert
#pragma omp critical
            {
              A.U.insert(i, level, std::move(Uo));
              A.Uc.insert(i, level, std::move(Uc));
              A.US_row.insert(i, level, matmul(Ui, Si)); // Save full basis for ULV update basis operation
              Ubig.insert(i, level, std::move(Ubig_i));
            }
          }
        }
      }
    }
  }

  void generate_far_coupling_matrices(SymmetricSharedBasisMatrix &A, const RowLevelMap &Ubig,
                                      const Domain &domain)
  {
    for (int64_t level = A.max_level; level >= A.min_adm_level; level--)
    {
#pragma omp parallel for
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        for (int64_t j : A.admissible_cols(i, level))
        {
          const Matrix Dij = generate_p2p_matrix(domain,
                                                 domain.get_cell_index(i, level),
                                                 domain.get_cell_index(j, level));
          Matrix Sij = matmul(matmul(Ubig(i, level), Dij, true, false), Ubig(j, level));
#pragma omp critical
          {
            A.S.insert(i, j, level, std::move(Sij));
          }
        }
      }
    }
  }

  void generate_near_coupling_matrices(SymmetricSharedBasisMatrix &A,
                                       const Domain &domain)
  {
    const int64_t level = A.max_level; // Only generate inadmissible leaf blocks
#pragma omp parallel for
    for (int64_t i = 0; i < A.level_nblocks[level]; i++)
    {
      for (int64_t j : A.inadmissible_cols(i, level))
      {
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

  void construct_H2(SymmetricSharedBasisMatrix &A,
                    const Domain &domain, const double admis,
                    const double err_tol, const int64_t max_rank,
                    const bool is_rel_tol = false)
  {
    // Initialize cell interactions for admissibility
    Admissibility::CellInteractionLists interactions;
    Admissibility::build_cell_interactions(interactions, domain, admis);
    Admissibility::assemble_farfields(interactions, domain);
    // Initialize matrix block structure and admissibility
    Admissibility::init_block_structure(A, domain);
    Admissibility::init_geometry_admissibility(A, interactions, domain, admis);
    // Generate cluster bases and coupling matrices
    RowLevelMap Ubig;
    generate_cluster_bases(A, Ubig, domain, interactions, err_tol, max_rank, is_rel_tol);
    generate_far_coupling_matrices(A, Ubig, domain);
    generate_near_coupling_matrices(A, domain);
  }

  Matrix get_Ubig(const SymmetricSharedBasisMatrix &A,
                  const int64_t i, const int64_t level)
  {
    if (level == A.max_level)
    {
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

  double construction_error(const SymmetricSharedBasisMatrix &A,
                            const Domain &domain, const bool relative = false)
  {
    double dense_norm = 0;
    double diff_norm = 0;
    // Inadmissible blocks (only at leaf level)
    {
      const int64_t level = A.max_level;
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        for (int64_t j : A.inadmissible_cols(i, level))
        {
          const Matrix Dij = generate_p2p_matrix(domain,
                                                 domain.get_cell_index(i, level),
                                                 domain.get_cell_index(j, level));
          const Matrix &Aij = A.D(i, j, level);
          const auto d_norm = norm(Dij);
          const auto diff = norm(Aij - Dij);
          dense_norm += d_norm * d_norm;
          diff_norm += diff * diff;
        }
      }
    }
    // Admissible blocks
    for (int64_t level = A.max_level; level >= A.min_adm_level; level--)
    {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        for (int64_t j : A.admissible_cols(i, level))
        {
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

  int64_t get_basis_min_rank(const SymmetricSharedBasisMatrix &A,
                             int64_t level_begin = 0,
                             int64_t level_end = 0)
  {
    if (level_begin == 0)
      level_begin = A.min_level;
    if (level_end == 0)
      level_end = A.max_level;
    int64_t min_rank = std::numeric_limits<int64_t>::max();
    for (int64_t level = level_begin; level <= level_end; level++)
    {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        if (A.U.exists(i, level))
        {
          min_rank = std::min(min_rank, A.U(i, level).cols);
        }
      }
    }
    return (min_rank == std::numeric_limits<int64_t>::max() ? -1 : min_rank);
  }

  int64_t get_basis_max_rank(const SymmetricSharedBasisMatrix &A,
                             int64_t level_begin = 0,
                             int64_t level_end = 0)
  {
    if (level_begin == 0)
      level_begin = A.min_level;
    if (level_end == 0)
      level_end = A.max_level;
    int64_t max_rank = -1;
    for (int64_t level = level_begin; level <= level_end; level++)
    {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        if (A.U.exists(i, level))
        {
          max_rank = std::max(max_rank, A.U(i, level).cols);
        }
      }
    }
    return max_rank;
  }

  double get_basis_avg_rank(const SymmetricSharedBasisMatrix &A,
                            int64_t level_begin = 0,
                            int64_t level_end = 0)
  {
    if (level_begin == 0)
      level_begin = A.min_level;
    if (level_end == 0)
      level_end = A.max_level;
    double sum_rank = 0;
    double num_bases = 0;
    for (int64_t level = level_begin; level <= level_end; level++)
    {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        if (A.U.exists(i, level))
        {
          sum_rank += static_cast<double>(A.U(i, level).cols);
          num_bases += 1.;
        }
      }
    }
    return sum_rank / num_bases;
  }

  // Return memory usage (in bytes)
  int64_t get_memory_usage(const SymmetricSharedBasisMatrix &A)
  {
    int64_t mem = 0;
    for (int64_t level = A.max_level; level >= A.min_level; level--)
    {
      for (int64_t i = 0; i < A.level_nblocks[level]; i++)
      {
        if (A.U.exists(i, level))
        {
          mem += A.U(i, level).memory_used();
        }
        if (A.US_row.exists(i, level))
        {
          mem += A.US_row(i, level).memory_used();
        }
        for (auto j : A.inadmissible_cols(i, level))
        {
          if (A.D.exists(i, j, level))
          {
            mem += A.D(i, j, level).memory_used();
          }
        }
        for (auto j : A.admissible_cols(i, level))
        {
          if (A.S.exists(i, j, level))
          {
            mem += A.S(i, j, level).memory_used();
          }
        }
      }
    }
    return mem;
  }

} // namespace

int main(int argc, char **argv)
{
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
  // 3: ELSES Wrapper kernel
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  // 4: Random Uniform Grid
  // 5: ELSES Wrapper
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim = argc > 9 ? atol(argv[9]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 10 ? std::string(argv[10]) : "";

  const auto start_overall = std::chrono::system_clock::now();

  Hatrix::set_kernel_constants(1.e-3, 1.);
  std::string kernel_name = "";
  switch (kernel_type)
  {
  case 0:
  {
    Hatrix::set_kernel_function(Hatrix::laplace_kernel);
    kernel_name = "laplace";
    break;
  }
  case 1:
  {
    Hatrix::set_kernel_function(Hatrix::yukawa_kernel);
    kernel_name = "yukawa";
    break;
  }
  case 2:
  {
    Hatrix::set_kernel_function(Hatrix::ELSES_dense_input);
    kernel_name = "ELSES-dense-file";
    break;
  }
  case 3:
  {
    init_elses_state();
    Hatrix::set_kernel_function(Hatrix::ELSES_kernel);
    kernel_name = "ELSES-kernel";
    break;
  }
  default:
  {
    Hatrix::set_kernel_function(Hatrix::laplace_kernel);
    kernel_name = "laplace";
  }
  }

  Hatrix::Domain domain(N, ndim);
  std::string geom_name = std::to_string(ndim) + "d-";
  switch (geom_type)
  {
  case 0:
  {
    domain.initialize_unit_circular_mesh();
    geom_name += "circular_mesh";
    break;
  }
  case 1:
  {
    domain.initialize_unit_cubical_mesh();
    geom_name += "cubical_mesh";
    break;
  }
  case 2:
  {
    domain.initialize_starsh_uniform_grid();
    geom_name += "starsh_uniform_grid";
    break;
  }
  case 3:
  {
    domain.ndim = 3;
    const auto prefix_end = file_name.find_last_of("/\\");
    geom_name = file_name.substr(prefix_end + 1);
    break;
  }
  case 4:
  {
    domain.initialize_random_uniform_grid();
    geom_name += "random_uniform_grid";
    break;
  }
  default:
  {
    domain.initialize_unit_circular_mesh();
    geom_name += "circular_mesh";
  }
  }
  // Pre-processing step for ELSES geometry
  if (geom_type == 3)
  {
    const int64_t num_atoms_per_molecule = 60;
    const int64_t num_electrons_per_atom = (kernel_type == 2 || kernel_type == 3) ? 4 : 1;
    const int64_t molecule_size = num_atoms_per_molecule * num_electrons_per_atom;
    assert(file_name.length() > 0);
    domain.read_bodies_ELSES(file_name + ".xyz", num_electrons_per_atom);
    assert(N == domain.N);

    domain.sort_bodies_ELSES(molecule_size);
    domain.build_tree_from_sorted_bodies(leaf_size, std::vector<int64_t>(N / leaf_size, leaf_size));
    if (kernel_type == 2)
    {
      domain.read_p2p_matrix_ELSES(file_name + ".dat");
    }
  }
  else
  {
    domain.build_tree(leaf_size);
  }

  SymmetricSharedBasisMatrix A;
  const auto start_construct = std::chrono::system_clock::now();
  construct_H2(A, domain, admis, err_tol, max_rank, is_rel_tol);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_construct - start_construct).count();

  const auto construct_abs_error = construction_error(A, domain, false);
  const auto construct_rel_error = construction_error(A, domain, true);
  const auto construct_min_rank = get_basis_min_rank(A);
  const auto construct_max_rank = get_basis_max_rank(A);
  const auto construct_avg_rank = get_basis_avg_rank(A);
  const auto construct_mem = static_cast<double>(get_memory_usage(A)) * 1e-9;

  const auto stop_overall = std::chrono::system_clock::now();
  const double overall_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_overall - start_overall).count();

  const std::string err_prefix = (is_rel_tol ? "rel" : "abs");
  printf("N=%" PRId64 " leaf_size=%d %s_err_tol=%.1e max_rank=%d admis=%.2lf kernel=%s geometry=%s\n"
         "h2_height=%d construct_min_rank=%d construct_max_rank=%d construct_avg_rank=%.2lf "
         "construct_time=%e[ms] construct_mem=%e[GB] construct_abs_err=%e construct_rel_err=%e\n"
         "overall_time=%e[ms]\n",
         N, (int)leaf_size, err_prefix.c_str(), err_tol, (int)max_rank, admis,
         kernel_name.c_str(), geom_name.c_str(),
         (int)A.max_level, (int)construct_min_rank, (int)construct_max_rank, construct_avg_rank,
         construct_time, construct_mem, err_prefix.c_str(), construct_abs_error, construct_rel_error,
         overall_time);

  return 0;
}
