#pragma once

#include <cstdint>
#include <vector>

#include "Cell.hpp"
#include "Domain.hpp"
#include "functions.hpp"
#include "sample_point.hpp"
#include "Hatrix/Hatrix.hpp"

namespace Hatrix {
namespace Admissibility {

typedef struct CellInteractionLists {
  std::vector<std::vector<int64_t>> near_cells;
  std::vector<std::vector<int64_t>> far_cells;
  std::vector<std::vector<int64_t>> far_particles;
} CellInteractionLists;

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

void assemble_farfields_sample(CellInteractionLists& interactions,
                               const Domain& domain, const int64_t sampling_algo,
                               const int64_t sample_local_size, const int64_t sample_far_size) {
  const int64_t ncells = domain.cells.size();
  // Bottom-up pass to sample each cell
  std::vector<std::vector<int64_t>> local_sample(ncells);
  for (int64_t level = domain.tree_height; level > 0; level--) {
    const auto level_begin = domain.level_offset[level];
    const auto level_end = domain.level_offset[level+1];
    for (int64_t k = level_begin; k < level_end; k++) {
      const auto& cell = domain.cells[k];
      std::vector<int64_t> initial_sample;
      if (level == domain.tree_height) {
        // Leaf level: use all particles as initial sample
        initial_sample = cell.get_bodies();
      }
      else {
        // Non-leaf level: gather children's samples
        for (int64_t i = 0; i < cell.nchilds; i++) {
          initial_sample.insert(initial_sample.end(),
                                local_sample[cell.child + i].begin(),
                                local_sample[cell.child + i].end());
        }
      }
      local_sample[k] = Hatrix::Utility::select_sample_points(
          domain.ndim, domain.bodies, initial_sample, sample_local_size, sampling_algo, 0);
    }
  }
  // Top-down pass to sample cell's farfield
  interactions.far_particles.assign(ncells, std::vector<int64_t>());
  std::vector<std::vector<int64_t>> initial_far_sample(ncells);
  for (int64_t level = 1; level <= domain.tree_height; level++) {
    const auto level_begin = domain.level_offset[level];
    const auto level_end   = domain.level_offset[level+1];
    for (int64_t k = level_begin; k < level_end; k++) {
      const auto& far_cells = interactions.far_cells[k];
      auto& initial_sample = initial_far_sample[k];  // Initially contains farfield from parent cell
      // Skip if current cell has no far cell and no farfields obtained from parent cell
      if ((far_cells.size() == 0) && (initial_sample.size() == 0)) continue;

      int64_t far_nbodies = 0;
      for (int64_t far_cell_idx: far_cells) {
        far_nbodies += local_sample[far_cell_idx].size();
      }
      if ((sampling_algo != 3) ||  // Not anchor-net sampling
          (initial_sample.size() > far_nbodies)) {
        // Put all samples of far particles into initial sample
        for (int64_t far_cell_idx: far_cells) {
          initial_sample.insert(initial_sample.end(),
                                local_sample[far_cell_idx].begin(),
                                local_sample[far_cell_idx].end());
        }
      }
      else {
        // Put only samples of far nodes into initial sample
        // So that the initial sample contain an equal proportion of
        // the current cell's far-particles and parent's far-particles
        const int64_t num_far_cells = far_cells.size();
        // 1. Find centroid of each far-cell's sample bodies
        // Store centers in column major, each column is a coordinate
        std::vector<double> centers(domain.ndim * num_far_cells);
        for (int64_t i = 0; i < num_far_cells; i++) {
          const auto far_cell_idx = far_cells[i];
          const auto far_cell_nsamples = local_sample[far_cell_idx].size();
          for (int64_t axis = 0; axis < domain.ndim; axis++) {
            const auto sum = Domain::get_Xsum(domain.bodies, local_sample[far_cell_idx], axis);
            centers[i * domain.ndim + axis] = sum / (double)far_cell_nsamples;
          }
        }
        // 2. Build anchor grid on far-node's sample bodies
        std::vector<double> far_cell_xmin(domain.ndim);
        std::vector<double> far_cell_xmax(domain.ndim);
        for (int64_t i = 0; i < num_far_cells; i++) {
          const auto far_cell_idx = far_cells[i];
          for (int64_t axis = 0; axis < domain.ndim; axis++) {
            const auto Xmin = Domain::get_Xmin(domain.bodies, local_sample[far_cell_idx], axis);
            const auto Xmax = Domain::get_Xmax(domain.bodies, local_sample[far_cell_idx], axis);
            far_cell_xmin[axis] = i == 0 ? Xmin :
                                  std::min(far_cell_xmin[axis], Xmin);
            far_cell_xmax[axis] = i == 0 ? Xmax :
                                  std::max(far_cell_xmax[axis], Xmax);
          }
        }
        std::vector<double> far_cell_box_size(domain.ndim);
        for (int64_t axis = 0; axis < domain.ndim; axis++) {
          far_cell_box_size[axis] = far_cell_xmax[axis] - far_cell_xmin[axis];
        }
        std::vector<int64_t> far_cell_grid_size(domain.ndim);
        proportional_int_decompose(domain.ndim, sample_far_size,
                                   far_cell_box_size.data(), far_cell_grid_size.data());
        const auto anchor_bodies = build_anchor_grid(domain.ndim,
                                                     far_cell_xmin.data(), far_cell_xmax.data(),
                                                     far_cell_box_size.data(), far_cell_grid_size.data(),
                                                     1);
        // 3. For each anchor point, assign it to the closest center
        const int64_t anchor_npt = anchor_bodies.size();
        std::vector<int64_t> closest_center(anchor_npt, 0);
        std::vector<int64_t> center_group_cnt(num_far_cells, 0);
        for (int64_t j = 0; j < anchor_npt; j++) {
          int64_t min_idx = -1;
          double min_dist = std::numeric_limits<double>::max();
          for (int64_t i = 0; i < num_far_cells; i++) {
            const auto dist2_i = Domain::dist2(domain.ndim,
                                               anchor_bodies[j].X,
                                               centers.data() + i * domain.ndim);
            if (dist2_i < min_dist) {
              min_dist = dist2_i;
              min_idx = i;
            }
          }
          closest_center[j] = min_idx;
          center_group_cnt[min_idx]++;
        }
        // 4. For each anchor point, select sample point from it's closest center group (far cell)
        std::set<int64_t> far_cells_sample;
        for (int64_t j = 0; j < anchor_npt; j++) {
          const auto closest_cell_idx = far_cells[closest_center[j]];
          int64_t min_idx = -1;
          double min_dist = std::numeric_limits<double>::max();
          for (int64_t i: local_sample[closest_cell_idx]) {
            const auto dist2_i = Domain::dist2(domain.ndim,
                                               anchor_bodies[j].X,
                                               domain.bodies[i].X);
            if (dist2_i < min_dist) {
              min_dist = dist2_i;
              min_idx = i;
            }
          }
          far_cells_sample.insert(min_idx);
        }
        // 5. Select one sample body each from cell that has no closest anchor point
        for (int64_t j = 0; j < num_far_cells; j++) {
          if (center_group_cnt[j] > 0) continue;
          const auto far_cell_idx = far_cells[j];
          int64_t min_idx = -1;
          double min_dist = std::numeric_limits<double>::max();
          for (int64_t i: local_sample[far_cell_idx]) {
            const auto dist2_i = Domain::dist2(domain.ndim,
                                               centers.data() + j * domain.ndim,
                                               domain.bodies[i].X);
            if (dist2_i < min_dist) {
              min_dist = dist2_i;
              min_idx = i;
            }
          }
          far_cells_sample.insert(min_idx);
        }
        // 6. Insert to initial sample
        initial_sample.insert(initial_sample.end(),
                              far_cells_sample.begin(), far_cells_sample.end());
      }
      // Select farfield samples
      interactions.far_particles[k] = Hatrix::Utility::select_sample_points(
          domain.ndim, domain.bodies, initial_sample, sample_far_size, sampling_algo, 1);
      // Non-leaf cell: propagate farfield to children
      if (level < domain.tree_height) {
        const auto& cell = domain.cells[k];
        for (int64_t i = 0; i < cell.nchilds; i++) {
          initial_far_sample[cell.child + i].insert(initial_far_sample[cell.child + i].end(),
                                                    interactions.far_particles[k].begin(),
                                                    interactions.far_particles[k].end());
        }
      }
    }
  }
}

}  // namespace Admissibility
}  // namespace Hatrix
