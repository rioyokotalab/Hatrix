#pragma once

#include <cstdint>
#include <vector>

#include "Cell.hpp"
#include "Domain.hpp"
#include "functions.hpp"
#include "Hatrix/Hatrix.hpp"

namespace Hatrix {

typedef struct CellInteractionLists {
  std::vector<std::vector<int64_t>> near_cells;
  std::vector<std::vector<int64_t>> far_cells;
  std::vector<std::vector<int64_t>> far_particles;
} CellInteractionLists;

namespace Admissibility {

bool is_well_separated(const Cell& source, const Cell& target,
                       const double theta) {
  const int64_t ndim = MAX_NDIM;
  double source_size = 0, target_size = 0;
  for (int64_t d = 0; d < ndim; d++) {
    source_size += source.radius[d] * source.radius[d];
    target_size += target.radius[d] * target.radius[d];
  }
  const auto distance = Domain::dist2(ndim, source.center, target.center);
  return (distance > (theta * (source_size + target_size)));
}

void dual_tree_traversal(CellInteractionLists& interactions,
                         const Domain& domain, const double theta,
                         const int64_t i, const int64_t j) {
  const auto& Ci = domain.cells[i];
  const auto& Cj = domain.cells[j];
  const auto i_level = Ci.level;
  const auto j_level = Cj.level;
  bool admissible = false;
  if (i_level == j_level) {
    admissible = is_well_separated(Ci, Cj, theta);
    if (admissible)
      interactions.far_cells[i].push_back(j);
    else
      interactions.near_cells[i].push_back(j);
  }
  if (!admissible) {
    if (i_level <= j_level && !Ci.is_leaf()) {
      dual_tree_traversal(interactions, domain, theta, Ci.child + 0, j);
      dual_tree_traversal(interactions, domain, theta, Ci.child + 1, j);
    }
    else if (j_level <= i_level && !Cj.is_leaf()) {
      dual_tree_traversal(interactions, domain, theta, i, Cj.child + 0);
      dual_tree_traversal(interactions, domain, theta, i, Cj.child + 1);
    }
  }
}

void build_cell_interactions(CellInteractionLists& interactions,
                             const Domain& domain, const double theta,
                             const MatrixType matrix_type = MatrixType::H2_MATRIX) {
  const int64_t ncells = domain.cells.size();
  interactions.near_cells.assign(ncells, std::vector<int64_t>());
  interactions.far_cells.assign(ncells, std::vector<int64_t>());
  if (matrix_type == MatrixType::H2_MATRIX) {
    dual_tree_traversal(interactions, domain, theta, 0, 0);
  }
  else {
    // BLR2: only initialize leaf level cell interactions
    const auto level = domain.tree_height;
    const auto level_begin = domain.level_offset[level];
    const auto level_end = domain.level_offset[level+1];
    for (int64_t i = level_begin; i < level_end; i++) {
      for (int64_t j = level_begin; j < level_end; j++) {
        const auto& Ci = domain.cells[i];
        const auto& Cj = domain.cells[j];
        bool admissible = is_well_separated(Ci, Cj, theta);
        if (admissible)
          interactions.far_cells[i].push_back(j);
        else
          interactions.near_cells[i].push_back(j);
      }
    }
  }
}

// Note: This utility function incurs O(N log N) cost
void assemble_farfields(CellInteractionLists& interactions, const Domain& domain) {
  const auto ncells = domain.cells.size();
  interactions.far_particles.assign(ncells, std::vector<int64_t>());
  // Top-down pass
  for (int64_t level = 1; level <= domain.tree_height; level++) {
    const auto level_begin = domain.level_offset[level];
    const auto level_end = domain.level_offset[level+1];
    for (int64_t i = level_begin; i < level_end; i++) {
      const auto& Ci = domain.cells[i];
      // Gather far particles
      for (const auto j: interactions.far_cells[i]) {
        const auto& Cj = domain.cells[j];
        const auto j_particles = Cj.get_bodies();
        interactions.far_particles[i].insert(interactions.far_particles[i].end(),
                                             j_particles.begin(), j_particles.end());
      }
      // Also pass to children
      for (int64_t j = 0; j < Ci.nchilds; j++) {
        const auto child = Ci.child + j;
        interactions.far_particles[child].insert(interactions.far_particles[child].end(),
                                                 interactions.far_particles[i].begin(),
                                                 interactions.far_particles[i].end());
      }
    }
  }
}

void init_block_structure(SymmetricSharedBasisMatrix& A,
                          const Domain& domain, const MatrixType matrix_type = MatrixType::H2_MATRIX) {
  // Initialize block structure
  A.min_level = matrix_type == MatrixType::H2_MATRIX ? 1 : domain.tree_height;
  A.max_level = domain.tree_height;
  A.level_nblocks.assign(A.max_level + 1, 0);
  A.level_nblocks[A.min_level-1] = 1;  // Root level block
  for (int64_t level = A.min_level; level <= A.max_level; level++) {
    A.level_nblocks[level] = domain.level_offset[level+1] - domain.level_offset[level];
  }
}

void init_geometry_admissibility(SymmetricSharedBasisMatrix& A,
                                 const CellInteractionLists& interactions,
                                 const Domain& domain, const double theta,
                                 const MatrixType matrix_type = MatrixType::H2_MATRIX) {
  // Assemble blockwise is_admissible structure from cells' interaction lists
  A.inadmissible_cols.insert(0, A.min_level-1, std::vector<int64_t>());
  A.admissible_cols.insert(0, A.min_level-1, std::vector<int64_t>());
  // Root level is always inadmissible
  A.is_admissible.insert(0, 0, A.min_level-1, false);
  A.inadmissible_cols(0, A.min_level-1).push_back(0);
  A.min_adm_level = 0;
  for (int64_t level = A.min_level; level <= A.max_level; level++) {
    const auto level_begin = domain.level_offset[level];
    const auto level_end = domain.level_offset[level+1];
    for (int64_t i = level_begin; i < level_end; i++) {
      // Near interaction list: inadmissible dense blocks
      const auto block_i = i - level_begin;
      A.inadmissible_cols.insert(block_i, level, std::vector<int64_t>());
      for (const auto j: interactions.near_cells[i]) {
        const auto block_j = j - level_begin;
        A.is_admissible.insert(block_i, block_j, level, false);
        A.inadmissible_cols(block_i, level).push_back(block_j);
      }
      // Far interaction list: admissible low-rank blocks
      A.admissible_cols.insert(block_i, level, std::vector<int64_t>());
      for (const auto j: interactions.far_cells[i]) {
        const auto block_j = j - level_begin;
        A.is_admissible.insert(block_i, block_j, level, true);
        A.admissible_cols(block_i, level).push_back(block_j);
      }
      if ((A.min_adm_level == 0) && (interactions.far_cells[i].size() > 0)) {
        A.min_adm_level = level;
      }
    }
  }
}

}  // namespace Admissibility
}  // namespace Hatrix
