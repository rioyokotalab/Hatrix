#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "Body.hpp"
#include "Cell.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Hatrix {

class Domain {
 public:
  int64_t N, ndim;
  int64_t ncells, tree_height;
  double X0[MAX_NDIM], X0_min[MAX_NDIM], X0_max[MAX_NDIM], R0;  // Root level bounding box
  std::vector<Body> bodies;
  std::vector<Cell> cells;
  Matrix p2p_matrix;

  static int64_t get_cell_idx(const int64_t block_index, const int64_t level) {
    return (1 << level) - 1 + block_index;
  }

  static double get_Xmin(const std::vector<Body>& bodies_arr,
                         const std::vector<int64_t>& bodies_idx,
                         const int64_t axis) {
    assert(axis < MAX_NDIM);
    double Xmin = bodies_arr[bodies_idx[0]].X[axis];
    for (int64_t i = 1; i < bodies_idx.size(); i++) {
      Xmin = std::min(Xmin, bodies_arr[bodies_idx[i]].X[axis]);
    }
    return Xmin;
  }

  static double get_Xmax(const std::vector<Body>& bodies_arr,
                         const std::vector<int64_t>& bodies_idx,
                         const int64_t axis) {
    assert(axis < MAX_NDIM);
    double Xmax = bodies_arr[bodies_idx[0]].X[axis];
    for (int64_t i = 1; i < bodies_idx.size(); i++) {
      Xmax = std::max(Xmax, bodies_arr[bodies_idx[i]].X[axis]);
    }
    return Xmax;
  }

  static double get_Xsum(const std::vector<Body>& bodies_arr,
                         const std::vector<int64_t>& bodies_idx,
                         const int64_t axis) {
    assert(axis < MAX_NDIM);
    double Xsum = 0;
    for (int64_t i = 0; i < bodies_idx.size(); i++) {
      Xsum += bodies_arr[bodies_idx[i]].X[axis];
    }
    return Xsum;
  }

  // Compute squared euclidean distance between two coordinates
  static double dist2(const int64_t ndim,
                      const double a_X[/* ndim */],
                      const double b_X[/* ndim */]) {
    double dist = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      dist += (a_X[axis] - b_X[axis]) *
              (a_X[axis] - b_X[axis]);
    }
    return dist;
  }

  static bool is_well_separated(const Cell& source, const Cell& target,
                                const double theta) {
    const auto distance = std::sqrt(dist2(MAX_NDIM, source.center, target.center));
    const auto source_diam = source.get_diameter();
    const auto target_diam = target.get_diameter();
    return (distance > (theta * std::max(source_diam, target_diam)));
  }

  // Taken from: H2Pack GitHub
  // Decompose an integer into the sum of multiple integers which are approximately proportional
  // to another floating point array.
  // Input parameters:
  //   nelem      : Number of integers after decomposition
  //   decomp_sum : The target number to be decomposed
  //   prop       : Floating point array as the proportions
  // Output parameter:
  //   decomp : Decomposed values, decomp[i] ~= prop[i] / sum(prop[0:nelem-1]) * decomp_sum
  // Return value: product(decomp[0:nelem-1])
  static int64_t proportional_int_decompose(const int64_t nelem,
                                            const int64_t decomp_sum,
                                            const double* prop,
                                            int64_t* decomp) {
    double sum_prop = 0.0;
    for (int64_t i = 0; i < nelem; i++) {
      sum_prop += prop[i];
    }
    std::vector<double> decomp_prop(nelem, 0);
    int decomp_sum0 = 0;
    for (int64_t i = 0; i < nelem; i++) {
      decomp_prop[i] = (double)decomp_sum * prop[i] / sum_prop;
      decomp[i] = (int64_t)std::floor(decomp_prop[i]);
      decomp_sum0 += decomp[i];
    }
    for (int64_t k = decomp_sum0; k < decomp_sum; k++) {
      // Add 1 to the position that got hit most by floor
      int64_t min_idx = 0;
      double max_diff = decomp_prop[0] - (double)decomp[0];
      for (int64_t i = 1; i < nelem; i++) {
        const auto diff = decomp_prop[i] - (double)decomp[i];
        if (diff > max_diff) {
          max_diff = diff;
          min_idx = i;
        }
      }
      decomp[min_idx]++;
    }
    int64_t prod1 = 1;
    for (int64_t i = 0; i < nelem; i++) prod1 *= (decomp[i] + 1);
    return prod1;
  }

  // Source: https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c
  static std::vector<Body>
  build_anchor_grid(const int64_t ndim,
                    const double coord_min[/* ndim */],
                    const double coord_max[/* ndim */],
                    const double enbox_size[/* ndim */],
                    const int64_t grid_size[/* ndim */],
                    const int64_t grid_algo = 0) {
    int64_t max_grid_size = 0;
    int64_t anchor_npt = 1;
    for (int64_t i = 0; i < ndim; i++) {
      max_grid_size =
          (grid_size[i] > max_grid_size) ? grid_size[i] : max_grid_size;
      anchor_npt *= (grid_size[i] + 1);
    }
    max_grid_size++;

    // 1. Assign anchor points in each dimension
    std::vector<double> anchor_dim(ndim * max_grid_size);
    for (int64_t i = 0; i < ndim; i++) {
      if (grid_size[i] == 0) {
        const auto offset_i = i * max_grid_size;
        anchor_dim[offset_i] = (coord_min[i] + coord_max[i]) * 0.5;
      }
    }
    if (grid_algo == 0) {  // Default: grid_algo == 6 in H2Pack code
      const double c0 = 1.0;
      const double c1 = 0.5;
      const double c2 = 0.25;
      for (int64_t i = 0; i < ndim; i++) {
        const auto size_i = c0 * enbox_size[i] / ((double)grid_size[i] + c1);
        const auto offset_i = i * max_grid_size;
        for (int64_t j = 0; j <= grid_size[i]; j++) {
          anchor_dim[offset_i + j] =
              coord_min[i] + c2 * size_i + size_i * (double)j;
        }
      }
    }
    else {  // Chebyshev anchor points: grid_algo == 2 in H2Pack code
      for (int64_t i = 0; i < ndim; i++) {
        if (grid_size[i] == 0) continue;
        const auto offset_i = i * max_grid_size;
        const auto s0 = 0.5 * (coord_max[i] + coord_min[i]);
        const auto s1 = 0.5 * (coord_max[i] - coord_min[i]);
        const auto s2 = M_PI / (2.0 * (double)grid_size[i] + 2);
        for (int64_t j = 0; j <= grid_size[i]; j++) {
          const double v0 = 2.0 * (double)j + 1.0;
          const double v1 = std::cos(v0 * s2);
          anchor_dim[offset_i + j] = s0 + s1 * v1;
        }
      }
    }
    // 2. Do a tensor product to get all anchor points
    std::vector<Body> anchor_coord;
    anchor_coord.resize(anchor_npt);
    int64_t stride[MAX_NDIM + 1];
    stride[0] = 1;
    for (int64_t i = 0; i < ndim; i++) {
      stride[i + 1] = stride[i] * (grid_size[i] + 1);
    }
    for (int64_t i = 0; i < anchor_npt; i++) {
      for (int64_t j = 0; j < ndim; j++) {
        const int64_t dim_idx = (i / stride[j]) % (grid_size[j] + 1);
        anchor_coord[i].X[j] = anchor_dim[j * max_grid_size + dim_idx];
      }
    }
    return anchor_coord;
  }

  static std::vector<int64_t>
  select_sample_bodies(const int64_t ndim,
                       const std::vector<Body>& bodies_arr,
                       const std::vector<int64_t>& bodies_idx,
                       const int64_t sample_size,
                       const int64_t sampling_algo,
                       const int64_t grid_algo = 0,
                       const bool ELSES_GEOM = false) {
    const int64_t nbodies = bodies_idx.size();
    if (sample_size >= nbodies) {
      return bodies_idx;
    }
    std::vector<int64_t> sample_idx;
    switch (sampling_algo) {
      case 0: {  // Select based on equally spaced indices
        const int64_t d =
            (int64_t)std::floor((double)nbodies / (double)sample_size);
        int64_t k = 0;
        sample_idx.reserve(sample_size);
        for (int64_t i = 0; i < sample_size; i++) {
          sample_idx.push_back(bodies_idx[k]);
          k += d;
        }
        break;
      }
      case 1: {  // Random sampling
        static std::mt19937 g(1234);  // Use fixed seed for reproducibility
        std::vector<int64_t> random_indices(nbodies, 0);
        for (int64_t i = 0; i < nbodies; i++) {
          random_indices[i] = i;
        }
        // Random shuffle 3 times
        for (int64_t i = 0; i < 3; i++) {
          std::shuffle(random_indices.begin(), random_indices.end(), g);
        }
        // Choose sample based on random indices
        sample_idx.reserve(sample_size);
        for (int64_t i = 0; i < sample_size; i++) {
          const auto ridx = random_indices[i];
          sample_idx.push_back(bodies_idx[ridx]);
        }
        break;
      }
      case 2: {  // Farthest Point Sampling (FPS)
        std::vector<bool> chosen(nbodies, false);
        // Find centroid of bodies
        double center[MAX_NDIM];
        for (int64_t axis = 0; axis < ndim; axis++) {
          const auto Xsum = get_Xsum(bodies_arr, bodies_idx, axis);
          center[axis] = Xsum / (double)bodies_idx.size();
        }
        // Start with point closest to the centroid as pivot
        int64_t pivot = -1;
        double min_dist2 = std::numeric_limits<double>::max();
        for (int64_t i = 0; i < nbodies; i++) {
          const auto dist2_i =
              dist2(ndim, center, bodies_arr[bodies_idx[i]].X);
          if (dist2_i < min_dist2) {
            min_dist2 = dist2_i;
            pivot = i;
          }
        }
        chosen[pivot] = true;
        for (int64_t k = 1; k < sample_size; k++) {
          // Add the farthest body from pivot into sample
          double max_dist2 = -1.;
          int64_t farthest_idx = -1;
          for (int64_t i = 0; i < nbodies; i++) {
            if(!chosen[i]) {
              const auto dist2_i = dist2(ndim,
                                         bodies_arr[bodies_idx[pivot]].X,
                                         bodies_arr[bodies_idx[i]].X);
              if (dist2_i > max_dist2) {
                max_dist2 = dist2_i;
                farthest_idx = i;
              }
            }
          }
          chosen[farthest_idx] = true;
          pivot = farthest_idx;
        }
        sample_idx.reserve(sample_size);
        for (int64_t i = 0; i < nbodies; i++) {
          if (chosen[i]) {
            sample_idx.push_back(bodies_idx[i]);
          }
        }
        break;
      }
      case 3: {  // Anchor Net Method
        double coord_max[MAX_NDIM], coord_min[MAX_NDIM], diameter[MAX_NDIM];
        for (int64_t axis = 0; axis < ndim; axis++) {
          coord_min[axis] = get_Xmin(bodies_arr, bodies_idx, axis);
          coord_max[axis] = get_Xmax(bodies_arr, bodies_idx, axis);
          diameter[axis] = coord_max[axis] - coord_min[axis];
        }
        int64_t grid_size[MAX_NDIM];
        const auto anchor_npt = proportional_int_decompose(ndim, sample_size,
                                                           diameter, grid_size);
        if (anchor_npt < nbodies) {
          const auto anchor_bodies =
              build_anchor_grid(ndim, coord_min, coord_max,
                                diameter, grid_size, grid_algo);
          std::vector<bool> chosen(nbodies, false);
          for (int64_t i = 0; i < anchor_npt; i++) {
            int64_t min_idx = -1;
            double min_dist2 = std::numeric_limits<double>::max();
            for (int64_t j = 0; j < nbodies; j++) {
              const auto dist2_ij = dist2(ndim,
                                          anchor_bodies[i].X,
                                          bodies_arr[bodies_idx[j]].X);
              if (dist2_ij < min_dist2) {
                min_dist2 = dist2_ij;
                min_idx = j;
              }
            }
            chosen[min_idx] = true;
          }
          sample_idx.reserve(anchor_npt);
          std::vector<bool> is_chosen_particle(bodies_arr.size() / 4, false);
          for (int64_t i = 0; i < nbodies; i++) {
            // Check if atom from the same particle has been selected before
            bool ELSES_cond = true;
            if (ELSES_GEOM) {
              const auto particle_number = (int64_t)bodies_arr[bodies_idx[i]].value / 4;
              ELSES_cond = !is_chosen_particle[particle_number];
            }
            if (chosen[i] && ELSES_cond) {
              sample_idx.push_back(bodies_idx[i]);
              if (ELSES_GEOM) {
                const auto particle_number = (int64_t)bodies_arr[bodies_idx[i]].value / 4;
                is_chosen_particle[particle_number] = true;
              }
            }
          }
        }
        else {
          sample_idx = bodies_idx;
        }
        break;
      }
      default: { // No sampling
        sample_idx = bodies_idx;
        break;
      }
    }
    return sample_idx;
  }

 private:
  void orthogonal_recursive_bisection(
      const int64_t left, const int64_t right, const int64_t leaf_size,
      const int64_t level, const int64_t block_index) {
    // Initialize cell
    const auto cell_idx = get_cell_idx(block_index, level);
    auto& cell = cells[cell_idx];
    cell.body_offset = left;
    cell.nbodies = right - left;
    cell.level = level;
    cell.block_index = block_index;
    double radius_max = 0;
    int64_t sort_axis = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      const auto Xmin = get_Xmin(bodies, cell.get_bodies(), axis);
      const auto Xmax = get_Xmax(bodies, cell.get_bodies(), axis);
      const auto Xsum = get_Xsum(bodies, cell.get_bodies(), axis);
      cell.center[axis] = (Xmin + Xmax) / 2.;  // Midpoint
      cell.radius[axis] = (Xmax - Xmin) / 2.;

      if (cell.radius[axis] > radius_max) {
        radius_max = cell.radius[axis];
        sort_axis = axis;
      }
    }

    if (cell.nbodies <= leaf_size) {  // Leaf level is reached
      cell.child = -1;
      cell.nchilds = 0;
      return;
    }

    // Sort bodies based on axis with largest radius
    std::sort(bodies.begin() + left, bodies.begin() + right,
              [sort_axis](const Body& lhs, const Body& rhs) {
                return lhs.X[sort_axis] < rhs.X[sort_axis];
              });
    // Split into two equal parts
    const auto mid = (left + right) / 2;
    cell.child = get_cell_idx(block_index << 1, level + 1);
    cell.nchilds = 2;
    cells[cell.child].parent = cell_idx;
    cells[cell.child + 1].parent = cell_idx;
    orthogonal_recursive_bisection(left, mid, leaf_size,
                                   level + 1, block_index << 1);
    orthogonal_recursive_bisection(mid, right, leaf_size,
                                   level + 1, (block_index << 1) + 1);
  }

  void dual_tree_traversal(Cell& Ci, Cell& Cj, const double theta) {
    const auto i_level = Ci.level;
    const auto j_level = Cj.level;
    bool admissible = false;
    if (i_level == j_level) {
      admissible = is_well_separated(Ci, Cj, theta);
      if (admissible) {
        Ci.far_list.push_back(get_cell_idx(Cj.block_index, Cj.level));
      }
      else {
        Ci.near_list.push_back(get_cell_idx(Cj.block_index, Cj.level));
      }
    }
    if (!admissible) {
      if (i_level <= j_level && !Ci.is_leaf()) {
        dual_tree_traversal(cells[Ci.child], Cj, theta);
        dual_tree_traversal(cells[Ci.child + 1], Cj, theta);
      }
      else if (j_level <= i_level && !Cj.is_leaf()) {
        dual_tree_traversal(Ci, cells[Cj.child], theta);
        dual_tree_traversal(Ci, cells[Cj.child + 1], theta);
      }
    }
  }

  // Sort indices within cell's interaction lists
  void sort_interaction_lists() {
    for (auto& cell: cells) {
      std::sort(cell.near_list.begin(), cell.near_list.end());
      std::sort(cell.far_list.begin(), cell.far_list.end());
    }
  }

  // Levelwise offset of Hilbert key
  int64_t levelOffset(int64_t level) {
    return (((int64_t)1 << 3 * level) - 1) / 7;
  }

  // Get 3-D Hilbert index from coordinates
  void get3DIndex(const double X[MAX_NDIM], const int64_t level,
                  int64_t iX[MAX_NDIM]) {
    const double dx = 2 * R0 / (1 << level);
    for (int64_t axis = 0; axis < 3; axis++) {
      iX[axis] = floor((X[axis] - X0_min[axis]) / dx);
    }
  }

  int64_t getKey(int64_t iX[MAX_NDIM], const int64_t level,
                 const bool offset = true) {
    // Preprocess
    int64_t M = 1 << (level - 1);
    for (int64_t Q=M; Q>1; Q>>=1) {
      int64_t R = Q - 1;
      for (int64_t d=0; d<3; d++) {
        if (iX[d] & Q) iX[0] ^= R;
        else {
          int64_t t = (iX[0] ^ iX[d]) & R;
          iX[0] ^= t;
          iX[d] ^= t;
        }
      }
    }
    for (int64_t d=1; d<3; d++) iX[d] ^= iX[d-1];
    int64_t t = 0;
    for (int64_t Q=M; Q>1; Q>>=1)
      if (iX[2] & Q) t ^= Q - 1;
    for (int64_t d=0; d<3; d++) iX[d] ^= t;

    int64_t i = 0;
    for (int64_t l = 0; l < level; l++) {
      i |= (iX[2] & (int64_t)1 << l) << 2*l;
      i |= (iX[1] & (int64_t)1 << l) << (2*l + 1);
      i |= (iX[0] & (int64_t)1 << l) << (2*l + 2);
    }
    if (offset) i += levelOffset(level);
    return i;
  }

 public:
  Domain(const int64_t _N, const int64_t _ndim)
      : N(_N), ndim(_ndim) {
    if (ndim < 1 || ndim > MAX_NDIM) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void build_tree(const int64_t leaf_size) {
    // Assume balanced binary tree
    tree_height = (int64_t)std::log2((double)N / (double)leaf_size);
    const int64_t nleaf_cells = (int64_t)1 << tree_height;
    ncells = 2 * nleaf_cells - 1;
    // Initialize empty cells
    cells.resize(ncells);
    // Partition
    orthogonal_recursive_bisection(0, N, leaf_size, 0, 0);
  }

  void build_interactions(const double theta) {
    dual_tree_traversal(cells[0], cells[0], theta);
    sort_interaction_lists();
  }

  /*
    Refine interactions resulting from non-optimal partitioning
    This basically merge 2x2 admissible block into a single upper level admissible block.
  */
  void refine_interactions() {
    // Remove element by value from STL container
    auto erase_by_value = [](std::vector<int64_t>& v, const int64_t value) {
      v.erase(std::remove(v.begin(), v.end(), value), v.end());
    };
    for (int64_t level = tree_height; level > 0; level--) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t i = 0; i < level_ncells; i += 2) {
        for (int64_t j = i + 2; j < level_ncells; j += 2) {
          const auto i1 = level_offset + i;
          const auto i2 = level_offset + i + 1;
          const auto j1 = level_offset + j;
          const auto j2 = level_offset + j + 1;
          const bool i1_j1_found =
              std::find(cells[i1].far_list.begin(),
                        cells[i1].far_list.end(), j1) != cells[i1].far_list.end();
          const bool i1_j2_found =
              std::find(cells[i1].far_list.begin(),
                        cells[i1].far_list.end(), j2) != cells[i1].far_list.end();
          const bool i2_j1_found =
              std::find(cells[i2].far_list.begin(),
                        cells[i2].far_list.end(), j1) != cells[i2].far_list.end();
          const bool i2_j2_found =
              std::find(cells[i2].far_list.begin(),
                        cells[i2].far_list.end(), j2) != cells[i2].far_list.end();
          if (i1_j1_found && i1_j2_found &&
              i2_j1_found && i2_j2_found) {
            // Erase from each other's far_list
            erase_by_value(cells[i1].far_list, j1);
            erase_by_value(cells[i1].far_list, j2);
            erase_by_value(cells[i2].far_list, j1);
            erase_by_value(cells[i2].far_list, j2);
            // Erase from j1 and j2 as well due to symmetricity
            erase_by_value(cells[j1].far_list, i1);
            erase_by_value(cells[j1].far_list, i2);
            erase_by_value(cells[j2].far_list, i1);
            erase_by_value(cells[j2].far_list, i2);

            const auto parent_i = cells[i1].parent;
            const auto parent_j = cells[j1].parent;
            // Erase from parent's near_list
            erase_by_value(cells[parent_i].near_list, parent_j);
            erase_by_value(cells[parent_j].near_list, parent_i);
            // Insert to parent's far_list
            cells[parent_i].far_list.push_back(parent_j);
            cells[parent_j].far_list.push_back(parent_i);
          }
        }
      }
    }
  }

  void build_sample_bodies(const int64_t sample_self_size,
                           const int64_t sample_far_size,
                           const int64_t sampling_algo,
                           const bool ELSES_GEOM = false) {
    const auto sample_near_size = sample_far_size;
    build_sample_bodies(sample_self_size, sample_near_size, sample_far_size,
                        sampling_algo, ELSES_GEOM);
  }

  void build_sample_bodies(const int64_t sample_self_size,
                           const int64_t sample_near_size,
                           const int64_t sample_far_size,
                           const int64_t sampling_algo,
                           const bool ELSES_GEOM = false) {
    // Bottom-up pass to select cell's sample bodies
    for (int64_t level = tree_height; level > 0; level--) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        std::vector<int64_t> initial_sample;
        if (level == tree_height) {
          // Leaf level: use all bodies as initial sample
          initial_sample = cell.get_bodies();
        }
        else {
          // Non-leaf level: gather children's samples
          const auto& child1 = cells[cell.child];
          const auto& child2 = cells[cell.child + 1];
          initial_sample.insert(initial_sample.end(),
                                child1.sample_bodies.begin(),
                                child1.sample_bodies.end());
          initial_sample.insert(initial_sample.end(),
                                child2.sample_bodies.begin(),
                                child2.sample_bodies.end());
        }
        cell.sample_bodies =
            select_sample_bodies(ndim, bodies, initial_sample,
                                 sample_self_size, sampling_algo, 0, ELSES_GEOM);
      }
    }
    // Bottom-up pass to select cell's nearfield sample
    for (int64_t level = tree_height; level > 0; level--) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        const auto cell_idx = level_offset + node;
        auto& cell = cells[cell_idx];
        std::vector<int64_t> initial_sample;
        for (const auto near_idx: cell.near_list) {
          if (near_idx != cell_idx) {  // Exclude self-to-self interaction
            const auto& near_cell = cells[near_idx];
            initial_sample.insert(initial_sample.end(),
                                  near_cell.sample_bodies.begin(),
                                  near_cell.sample_bodies.end());
          }
        }
        cell.sample_nearfield =
            select_sample_bodies(ndim, bodies, initial_sample,
                                 sample_near_size, sampling_algo, 1, ELSES_GEOM);
      }
    }
    // Top-down pass to select cell's farfield sample
    for (int64_t level = 1; level <= tree_height; level++) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        const auto& parent = cells[cell.parent];
        if ((parent.sample_farfield.size() == 0) &&
            (cell.far_list.size() == 0)) continue;

        int64_t far_nbodies = 0;
        for (const auto far_idx: cell.far_list) {
          far_nbodies += cells[far_idx].sample_bodies.size();
        }
        auto initial_sample = parent.sample_farfield;
        if ((sampling_algo != 3) ||
            (parent.sample_farfield.size() > far_nbodies)) {
          // Put all sample_bodies of far nodes into initial sample
          for (const auto far_idx: cell.far_list) {
            initial_sample.insert(initial_sample.end(),
                                  cells[far_idx].sample_bodies.begin(),
                                  cells[far_idx].sample_bodies.end());
          }
        }
        else {
          // Put only samples of far nodes into initial sample
          // So that the initial sample contain an equal proportion of
          // current cell's far-bodies and parent's far-bodies
          const int64_t num_far_nodes = cell.far_list.size();
          // 1. Find centroid of each far-node's sample bodies
          // Store centers in column major, each column is a coordinate
          std::vector<double> centers(ndim * num_far_nodes);
          for (int64_t i = 0; i < num_far_nodes; i++) {
            const auto far_idx = cell.far_list[i];
            const auto& far_cell = cells[far_idx];
            const auto far_cell_nsamples = far_cell.sample_bodies.size();
            for (int64_t axis = 0; axis < ndim; axis++) {
              const auto sum = get_Xsum(bodies, far_cell.sample_bodies, axis);
              centers[i * ndim + axis] = sum / (double)far_cell_nsamples;
            }
          }
          // 2. Build anchor grid on far-node's sample bodies
          double far_node_xmin[MAX_NDIM], far_node_xmax[MAX_NDIM];
          for (int64_t i = 0; i < num_far_nodes; i++) {
            const auto far_idx = cell.far_list[i];
            const auto& far_cell = cells[far_idx];
            for (int64_t axis = 0; axis < ndim; axis++) {
              const auto Xmin = get_Xmin(bodies, far_cell.sample_bodies, axis);
              const auto Xmax = get_Xmax(bodies, far_cell.sample_bodies, axis);
              far_node_xmin[axis] = i == 0 ? Xmin :
                                    std::min(far_node_xmin[axis], Xmin);
              far_node_xmax[axis] = i == 0 ? Xmax :
                                    std::max(far_node_xmax[axis], Xmax);
            }
          }
          double far_node_box_size[MAX_NDIM];
          for (int64_t axis = 0; axis < ndim; axis++) {
            far_node_box_size[axis] = far_node_xmax[axis] - far_node_xmin[axis];
          }
          int64_t far_node_grid_size[MAX_NDIM];
          proportional_int_decompose(ndim, sample_far_size,
                                     far_node_box_size, far_node_grid_size);
          const auto anchor_bodies = build_anchor_grid(ndim, far_node_xmin,
                                                       far_node_xmax,
                                                       far_node_box_size,
                                                       far_node_grid_size,
                                                       1);
          // 3. For each anchor point, assign it to the closest center
          const int64_t anchor_npt = anchor_bodies.size();
          std::vector<int64_t> closest_center(anchor_npt, 0);
          std::vector<int64_t> center_group_cnt(num_far_nodes, 0);
          for (int64_t k = 0; k < anchor_npt; k++) {
            int64_t min_idx = 0;
            double min_dist = dist2(ndim, anchor_bodies[k].X, centers.data());
            for (int64_t i = 1; i < num_far_nodes; i++) {
              const auto dist2_i = dist2(ndim, anchor_bodies[k].X,
                                         centers.data() + i * ndim);
              if (dist2_i < min_dist) {
                min_dist = dist2_i;
                min_idx = i;
              }
            }
            closest_center[k] = min_idx;
            center_group_cnt[min_idx]++;
          }
          // 4. For each anchor point, select sample point from it's closest center group (far node)
          std::set<int64_t> far_nodes_sample;
          for (int64_t k = 0; k < anchor_npt; k++) {
            const auto closest_cell_idx = cell.far_list[closest_center[k]];
            const auto& closest_cell = cells[closest_cell_idx];
            int64_t min_body_idx = -1;
            double min_dist = std::numeric_limits<double>::max();
            for (const auto body_idx: closest_cell.sample_bodies) {
              const auto dist_idx = dist2(ndim, anchor_bodies[k].X,
                                          bodies[body_idx].X);
              if (dist_idx < min_dist) {
                min_dist = dist_idx;
                min_body_idx = body_idx;
              }
            }
            far_nodes_sample.insert(min_body_idx);
          }
          // 5. Select one sample body each from cell that has no closest anchor point
          for (int64_t i = 0; i < num_far_nodes; i++) {
            if (center_group_cnt[i] > 0) continue;
            const auto far_idx = cell.far_list[i];
            const auto& far_cell = cells[far_idx];
            int64_t min_body_idx = -1;
            double min_dist = std::numeric_limits<double>::max();
            for (const auto body_idx: far_cell.sample_bodies) {
              const auto dist_idx = dist2(ndim, centers.data() + i * ndim,
                                          bodies[body_idx].X);
              if (dist_idx < min_dist) {
                min_dist = dist_idx;
                min_body_idx = body_idx;
              }
            }
            far_nodes_sample.insert(min_body_idx);
          }
          // 6. Insert to initial sample
          for (const auto body_idx: far_nodes_sample) {
            initial_sample.push_back(body_idx);
          }
        }
        cell.sample_farfield =
            select_sample_bodies(ndim, bodies, initial_sample,
                                 sample_far_size, sampling_algo, 1, ELSES_GEOM);
      }
    }
  }

  int64_t get_max_farfield_size() const {
    int64_t max_farfield_size = 0;
    for (int64_t i = 0; i < ncells; i++) {
      max_farfield_size = std::max(max_farfield_size,
                                   (int64_t)cells[i].sample_farfield.size());
    }
    return max_farfield_size;
  }

  void initialize_unit_circular_mesh() {
    if (ndim == 1) {
      // Generate uniform points within a unit straight line
      const auto x = equally_spaced_vector(N, 0, 1);
      for (int64_t i = 0; i < N; i++) {
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x[i], value));
      }
    }
    else if (ndim == 2) {
      // Generate a unit circle with N points on the circumference.
      for (int64_t i = 0; i < N; i++) {
        const double theta = (i * 2.0 * M_PI) / (double)N;
        const double x = cos(theta);
        const double y = sin(theta);
        const double value = (double)i / (double)N;

        bodies.emplace_back(Body(x, y, value));
      }
    }
    else if (ndim == 3) {
      // Generate a unit sphere mesh with N uniformly spaced points on the surface
      // https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
      const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
      for (int64_t i = 0; i < N; i++) {
        const double y = 1. - ((double)i / ((double)N - 1)) * 2.;  // y goes from 1 to -1

        // Note: setting constant radius = 1 will produce a cylindrical shape
        const double radius = std::sqrt(1. - y * y);  // radius at y
        const double theta = (double)i * phi;

        const double x = radius * std::cos(theta);
        const double z = radius * std::sin(theta);
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x, y, z, value));
      }
    }
  }

  void initialize_unit_cubical_mesh() {
    if (ndim == 1) {
      // Generate uniform points within a unit straight line
      const auto x = equally_spaced_vector(N, 0, 1);
      for (int64_t i = 0; i < N; i++) {
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x[i], value));
      }
    }
    else if (ndim == 2) {
      // Generate a unit square with N points on the sides
      if (N < 4) {
        std::cout << "N has to be >=4 for unit square mesh" << std::endl;
        exit(EXIT_FAILURE);
      }
      // Taken from H2Lib: Library/curve2d.c
      const double a = 0.5;
      const int64_t top = N / 4;
      const int64_t left = N / 2;
      const int64_t bottom = 3 * N / 4;
      int64_t i = 0;
      for (i = 0; i < top; i++) {
        const double x = a - 2.0 * a * i / top;
        const double y = a;
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x, y, value));
      }
      for (; i < left; i++) {
        const double x = -a;
        const double y = a - 2.0 * a * (i - top) / (left - top);
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x, y, value));
      }
      for (; i < bottom; i++) {
        const double x = -a + 2.0 * a * (i - left) / (bottom - left);
        const double y = -a;
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x, y, value));
      }
      for (; i < N; i++) {
        const double x = a;
        const double y = -a + 2.0 * a * (i - bottom) / (N - bottom);
        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(x, y, value));
      }
    }
    else if (ndim == 3) {
      // Generate a unit cube mesh with N points around the surface
      const int64_t mlen = (int64_t)ceil((double)N / 6.);
      const double alen = std::sqrt((double)mlen);
      const int64_t m = (int64_t)std::ceil(alen);
      const int64_t n = (int64_t)std::ceil((double)mlen / m);

      const double seg_fv = 1. / ((double)m - 1);
      const double seg_fu = 1. / (double)n;
      const double seg_sv = 1. / ((double)m + 1);
      const double seg_su = 1. / ((double)n + 1);

      for (int64_t i = 0; i < N; i++) {
        const int64_t face = i / mlen;
        const int64_t ii = i - face * mlen;
        const int64_t x = ii / m;
        const int64_t y = ii - x * m;
        const int64_t x2 = y & 1;

        double u, v;
        double px, py, pz;

        switch (face) {
          case 0: // POSITIVE X
            v = y * seg_fv;
            u = (0.5 * x2 + x) * seg_fu;
            px = 1.;
            py = 2. * v - 1.;
            pz = -2. * u + 1.;
            break;
          case 1: // NEGATIVE X
            v = y * seg_fv;
            u = (0.5 * x2 + x) * seg_fu;
            px = -1.;
            py = 2. * v - 1.;
            pz = 2. * u - 1.;
            break;
          case 2: // POSITIVE Y
            v = (y + 1) * seg_sv;
            u = (0.5 * x2 + x + 1) * seg_su;
            px = 2. * u - 1.;
            py = 1.;
            pz = -2. * v + 1.;
            break;
          case 3: // NEGATIVE Y
            v = (y + 1) * seg_sv;
            u = (0.5 * x2 + x + 1) * seg_su;
            px = 2. * u - 1.;
            py = -1.;
            pz = 2. * v - 1.;
            break;
          case 4: // POSITIVE Z
            v = y * seg_fv;
            u = (0.5 * x2 + x) * seg_fu;
            px = 2. * u - 1.;
            py = 2. * v - 1.;
            pz = 1.;
            break;
          case 5: // NEGATIVE Z
            v = y * seg_fv;
            u = (0.5 * x2 + x) * seg_fu;
            px = -2. * u + 1.;
            py = 2. * v - 1.;
            pz = -1.;
            break;
        }

        const double value = (double)i / (double)N;
        bodies.emplace_back(Body(px, py, pz, value));
      }
    }
  }

  void initialize_cube_uniform_grid() {
    for (int64_t i = 0; i < N; i++) {
      const double px = ndim > 0 ? ((double)rand() / RAND_MAX) : 0.;
      const double py = ndim > 1 ? ((double)rand() / RAND_MAX) : 0.;
      const double pz = ndim > 2 ? ((double)rand() / RAND_MAX) : 0.;
      const double value = (double)i / (double)N;
      bodies.emplace_back(Body(px, py, pz, value));
    }
  }

  void initialize_starsh_uniform_grid() {
    std::vector<int64_t> sides(ndim, 0);
    sides[0] = ceil(pow((double)N, 1.0 / ndim));
    int64_t total = sides[0];
    int64_t temp_N = N;
    for (int k = 1; k < ndim; ++k) {
      sides[k] = temp_N / sides[k-1];
      temp_N = sides[k];
    }
    bodies.resize(N, Body(std::vector<double>(ndim), 0));

    if (ndim == 1) {
      double space_0 = 1.0 / N;
      for (int64_t i = 0; i < sides[0]; ++i) {
        std::vector<double> point(ndim);
        point[0] = i * space_0;
        bodies[i] = Body(point, 0);
      }
    }
    else if (ndim == 2) {
      for (int k = 1; k < ndim; ++k) { total += sides[k]; }

      double space_0 = 1.0 / sides[0], space_1 = 1.0 / sides[1];
      for (int64_t i = 0; i < sides[0]; ++i) {
        for (int64_t j = 0; j < sides[1]; ++j) {
          std::vector<double> point(ndim);
          point[0] = i * space_0;
          point[1] = j * space_1;
          bodies[i + j * sides[0]] = Body(point, 0);
        }
      }
    }
    else {
      abort();
    }
  }

  void calculate_bounding_box() {
    Cell root;
    root.body_offset = 0;
    root.nbodies = N;
    R0 = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      const auto Xmin = get_Xmin(bodies, root.get_bodies(), axis);
      const auto Xmax = get_Xmax(bodies, root.get_bodies(), axis);
      X0[axis] = (Xmin + Xmax) / 2.;
      R0 = std::max(R0, (Xmax - Xmin) / 2.);
    }
    for (int64_t axis = 0; axis < ndim; axis++) {
      X0_min[axis] = X0[axis] - R0;
      X0_max[axis] = X0[axis] + R0;
    }
  }

  void read_bodies_ELSES(const std::string& file_name) {
    std::ifstream file;
    file.open(file_name);
    int64_t num_particles;
    file >> num_particles;
    constexpr int64_t num_atoms_per_particle = 4;
    ndim = 3;
    N = num_particles * num_atoms_per_particle;

    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore the rest of line after num_particles
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore line before atom positions
    int64_t body_idx = 0;
    for(int64_t i = 0; i < num_particles; i++) {
      std::string pref;
      double x, y, z;
      file >> pref >> x >> y >> z;
      file.ignore(1, '\n'); //Ignore newline
      for (int64_t k = 0; k < num_atoms_per_particle; k++) {
        bodies.emplace_back(Body(x, y, z, (double)body_idx));
        body_idx++;
      }
    }
    file.close();
  }

  // Build cells of tree adaptively using a top-down approach based on recursion
  void build_cells(Body *bodies, Body *buffer,
                   const int64_t begin, const int64_t end,
                   Cell * cell, std::vector<Cell>& cells, const int64_t leaf_size,
                   const double X[MAX_NDIM], const double R,
                   const int64_t level = 0, const bool direction = false) {
    // Create a tree cell
    cell->body_ptr = bodies + begin;
    if (direction) cell->body_ptr = buffer + begin;
    cell->nbodies = end - begin;
    cell->level = level;
    cell->nchilds = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      cell->center[axis] = X[axis];
      cell->radius[axis] = R;
    }
    int64_t iX[MAX_NDIM];
    get3DIndex(X, level, iX);
    cell->key = getKey(iX, level);
    // Count number of bodies in each octant
    int64_t size[8] = {0,0,0,0,0,0,0,0};
    for (int64_t i = begin; i < end; i++) {
      int64_t octant = ( bodies[i].X[0] > X[0]) +
                       ((bodies[i].X[1] > X[1]) << 1) +
                       ((bodies[i].X[2] > X[2]) << 2);
      size[octant]++;
    }
    // Exclusive scan to get offsets
    int64_t offset = begin;
    int64_t offsets[8], counter[8];
    for (int64_t i = 0; i < 8; i++) {
      offsets[i] = offset;
      offset += size[i];
      if (size[i]) cell->nchilds++;
    }
    // If cell is a leaf
    if (end - begin <= leaf_size) {
      cell->nchilds = 0;
      if (direction) {
        for (int64_t i = begin; i < end; i++) {
          buffer[i] = bodies[i];
        }
      }
      return;
    }
    // Sort bodies by octant
    for (int64_t i = 0; i < 8; i++) counter[i] = offsets[i];
    for (int64_t i = begin; i < end; i++) {
      int64_t octant = ( bodies[i].X[0] > X[0]) +
                       ((bodies[i].X[1] > X[1]) << 1) +
                       ((bodies[i].X[2] > X[2]) << 2);
      buffer[counter[octant]] = bodies[i];
      counter[octant]++;
    }
    // Loop over children and recurse
    assert(cells.capacity() >= cells.size()+cell->nchilds);
    cells.resize(cells.size()+cell->nchilds);
    Cell *child = &cells.back() - cell->nchilds + 1;
    cell->child_ptr = child;
    for (int64_t i = 0, c = 0; i < 8; i++) {
      double Xchild[MAX_NDIM];
      for (int64_t axis = 0; axis < MAX_NDIM; axis++) {
        Xchild[axis] = X[axis];
      }
      double Rchild = R / 2.;
      for (int64_t d = 0; d < 3; d++) {
        Xchild[d] += Rchild * (((i & 1 << d) >> d) * 2 - 1);
      }
      if (size[i]) {
        build_cells(buffer, bodies, offsets[i], offsets[i] + size[i],
                    &child[c++], cells, leaf_size, Xchild, Rchild, level+1, !direction);
      }
    }
  }

  void sort_bodies_ELSES() {
    // Calculate root level bounding box
    calculate_bounding_box();
    // Every consecutive 240 bodies (atoms) comprise a molecule
    const int64_t atom_leaf_size = 240;
    const auto nmols = N / atom_leaf_size;
    std::vector<Body> mol_centers(nmols);  // Center of each molecule
    for (int64_t i = 0; i < nmols; i++) {
      Cell cell;
      cell.body_offset = i * atom_leaf_size;
      cell.nbodies = atom_leaf_size;
      for (int64_t axis = 0; axis < ndim; axis++) {
        const auto Xmin = get_Xmin(bodies, cell.get_bodies(), axis);
        const auto Xmax = get_Xmax(bodies, cell.get_bodies(), axis);
        mol_centers[i].X[axis] = (Xmin + Xmax) / 2.;
      }
      mol_centers[i].value = (double)i;
    }
    // Partition until each box contain only one molecule
    std::vector<Body> buffer = mol_centers;
    std::vector<Cell> mol_cells(1);
    const int64_t mol_leaf_size = 1;
    mol_cells.reserve(nmols*(32/mol_leaf_size+1));
    build_cells(&mol_centers[0], &buffer[0], 0, nmols, &mol_cells[0], mol_cells, mol_leaf_size, X0, R0);
    for (int64_t i = 0; i < mol_cells.size(); i++) {
      if (mol_cells[i].nchilds == 0) {
        for (int64_t b = 0; b < mol_cells[i].nbodies; b++) {
          auto& bi = mol_cells[i].body_ptr[b];
          bi.key = mol_cells[i].key;
        }
      }
    }
    // Sort molecules based on hilbert index
    std::vector<Body> mol_centers_sorted = mol_centers;
    std::sort(mol_centers_sorted.begin(), mol_centers_sorted.end(),
              [](const Body& a, const Body& b) {
                return a.key < b.key;
              });
    // Sort bodies based on molecule hilbert index
    std::vector<Body> temp = bodies;
    int64_t count = 0;
    for (int64_t i = 0; i < nmols; i++) {
      const auto& mol_i = mol_centers_sorted[i];
      const auto srcBegin = (int64_t)mol_i.value * atom_leaf_size;
      const auto dstBegin = count;
      for (int64_t k = 0; k < atom_leaf_size; k++) {
        bodies[dstBegin + k] = temp[srcBegin + k];
      }
      count += atom_leaf_size;
    }
  }

  void build_tree_from_sorted_bodies(const int64_t leaf_size,
                                     const std::vector<int64_t>& buckets) {
    tree_height = (int64_t)std::log2((double)N / (double)leaf_size);
    const int64_t nleaf_cells = (int64_t)1 << tree_height;
    assert(nleaf_cells == buckets.size());

    ncells = 2 * nleaf_cells - 1;
    // Initialize empty cells
    cells.resize(ncells);
    // Build cell tree using bottom up traversal
    int64_t count = 0;
    for (int64_t level = tree_height; level >= 0; level--) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        cell.level = level;
        cell.block_index = node;
        if (level == tree_height) {
          // Construct leaf node
          cell.child = -1;
          cell.nchilds = 0;
          cell.body_offset = count;
          cell.nbodies = buckets[node];
          count += buckets[node];
        }
        else {
          // Construct non-leaf nodes from adjacent lower nodes
          const auto child1_idx = get_cell_idx((node << 1) + 0, level + 1);
          const auto child2_idx = get_cell_idx((node << 1) + 1, level + 1);
          auto& child1 = cells[child1_idx];
          auto& child2 = cells[child2_idx];
          cell.child = child1_idx;
          cell.nchilds = 2;
          cell.body_offset = child1.body_offset;
          cell.nbodies = child1.nbodies + child2.nbodies;
          // Set parent
          child1.parent = level_offset + node;
          child2.parent = level_offset + node;
        }
        // Calculate cell center and radius
        for (int64_t axis = 0; axis < ndim; axis++) {
          const auto Xmin = get_Xmin(bodies, cell.get_bodies(), axis);
          const auto Xmax = get_Xmax(bodies, cell.get_bodies(), axis);
          const auto Xsum = get_Xsum(bodies, cell.get_bodies(), axis);
          const auto diam = Xmax - Xmin;
          cell.center[axis] = (Xmin + Xmax) / 2.;  // Midpoint
          cell.radius[axis] = (diam == 0. && Xmin == 0.) ? 0. : (1.e-8 + diam / 2.);
        }
      }
    }
    cells[0].parent = -1; //  Root has no parent
  }

  void read_p2p_matrix_ELSES(const std::string& file_name) {
    std::ifstream file;
    file.open(file_name);
    // Ignore first two lines
    const int64_t nskip_lines = 2;
    for (int64_t i = 0; i < nskip_lines; i++) {
      file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    int64_t nrows, ncols, nnz;
    file >> nrows >> ncols >> nnz;
    Matrix D(nrows, ncols);
    D = 0; // Initialize with zero entries
    int64_t row, col;
    double val;
    for(int64_t k = 0; k < nnz; k++) {
      file >> col >> row >> val;
      D(row - 1, col - 1) = val;
      D(col - 1, row - 1) = val; // Symmetric
    }
    file.close();
    p2p_matrix = std::move(D);
  }

  void write_bodies(const std::string& file_name) const {
    const std::vector<char> axis{'x', 'y', 'z'};

    std::ofstream file;
    file.open(file_name, std::ios::out);
    for (int64_t k = 0; k < ndim; k++) {
      if (k > 0) file << ",";
      file << axis[k];
    }
    file << ",value" << std::endl;

    for (int64_t i = 0; i < N; i++) {
      for (int64_t k = 0; k < ndim; k++) {
        if (k > 0) file << ",";
        file << bodies[i].X[k];
      }
      file << "," << bodies[i].value << std::endl;
    }

    file.close();
  }
};

} // namespace Hatrix
