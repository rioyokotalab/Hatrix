#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

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
  std::vector<Body> bodies;
  std::vector<Cell> cells;

 private:
  double get_Xmin(const std::vector<int64_t>& loc, const int64_t axis) const {
    assert(axis < ndim);
    double Xmin = bodies[loc[0]].X[axis];
    for (int64_t i = 1; i < loc.size(); i++) {
      Xmin = std::min(Xmin, bodies[loc[i]].X[axis]);
    }
    return Xmin;
  }
  double get_Xmin(const int64_t body_start, const int64_t body_end,
                  const int64_t axis) const {
    assert(axis < ndim);
    double Xmin = bodies[body_start].X[axis];
    for (int64_t i = body_start + 1; i <= body_end ; i++) {
      Xmin = std::min(Xmin, bodies[i].X[axis]);
    }
    return Xmin;
  }

  double get_Xmax(const std::vector<int64_t>& loc, const int64_t axis) const {
    assert(axis < ndim);
    double Xmax = bodies[loc[0]].X[axis];
    for (int64_t i = 1; i < loc.size(); i++) {
      Xmax = std::max(Xmax, bodies[loc[i]].X[axis]);
    }
    return Xmax;
  }
  double get_Xmax(const int64_t body_start, const int64_t body_end,
                  const int64_t axis) const {
    assert(axis < ndim);
    double Xmax = bodies[body_start].X[axis];
    for (int64_t i = body_start + 1; i <= body_end ; i++) {
      Xmax = std::max(Xmax, bodies[i].X[axis]);
    }
    return Xmax;
  }

  void orthogonal_recursive_bisection(
      const int64_t left, const int64_t right, const int64_t leaf_size,
      const int64_t level, const int64_t index) {
    // Initialize cell
    const auto loc = get_cell_loc(index, level);
    cells[loc].body = left;
    cells[loc].nbodies = right - left;
    cells[loc].level = level;
    cells[loc].index = index;
    double radius_max = 0.;
    int64_t sort_axis = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      const auto Xmin = get_Xmin(left, right - 1, axis);
      const auto Xmax = get_Xmax(left, right - 1, axis);
      const auto diam = Xmax - Xmin;
      cells[loc].center[axis] = (Xmin + Xmax) / 2.;
      cells[loc].radius[axis] = (diam == 0. && Xmin == 0.) ? 0. : (1.e-8 + diam / 2.);

      if (cells[loc].radius[axis] > radius_max) {
        radius_max = cells[loc].radius[axis];
        sort_axis = axis;
      }
    }

    if (cells[loc].nbodies <= leaf_size) {  // Leaf level is reached
      cells[loc].child = -1;
      cells[loc].nchilds = 0;
      return;
    }

    // Sort bodies based on axis with largest radius
    std::sort(bodies.begin() + left, bodies.begin() + right,
              [sort_axis](const Body& lhs, const Body& rhs) {
                return lhs.X[sort_axis] < rhs.X[sort_axis];
              });
    // Split into two equal parts
    const auto mid = (left + right) / 2;
    cells[loc].child = get_cell_loc(index << 1, level + 1);
    cells[loc].nchilds = 2;
    cells[cells[loc].child].parent = loc;
    cells[cells[loc].child + 1].parent = loc;
    orthogonal_recursive_bisection(left, mid, leaf_size, level + 1, index << 1);
    orthogonal_recursive_bisection(mid, right, leaf_size, level + 1, (index << 1) + 1);
  }

  bool is_well_separated(const Cell& source, const Cell& target,
                         const double theta) const {
    const auto distance = source.distance_from(target);
    const auto diameter = std::min(source.get_diameter(), target.get_diameter());
    return distance > (theta * diameter);
  }

  void dual_tree_traversal(Cell& Ci, Cell& Cj, const double theta) {
    const auto i_level = Ci.level;
    const auto j_level = Cj.level;
    bool admissible = false;
    if (i_level == j_level) {
      admissible = is_well_separated(Ci, Cj, theta);
      if (admissible) {
        Ci.far_list.push_back(get_cell_loc(Cj.index, Cj.level));
      }
      else {
        Ci.near_list.push_back(get_cell_loc(Cj.index, Cj.level));
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

  // Sort locations within cell's interaction lists
  void sort_interaction_lists() {
    for (auto& cell: cells) {
      std::sort(cell.near_list.begin(), cell.near_list.end());
      std::sort(cell.far_list.begin(), cell.far_list.end());
    }
  }

  // Remove element by value from STL container
  void erase_by_value(std::vector<int64_t>& vec, const int64_t value) {
    vec.erase(std::remove(vec.begin(), vec.end(), value), vec.end());
  }

  // Compute squared euclidean distance between two coordinates
  double dist2(const double* a_X, const double* b_X) const {
    double dist = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      dist += (a_X[axis] - b_X[axis]) *
              (a_X[axis] - b_X[axis]);
    }
    return dist;
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
  int64_t proportional_int_decompose(const int64_t nelem, const int64_t decomp_sum,
                                     const double* prop, int64_t* decomp) const {
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
  std::vector<Body> build_anchor_grid(const int64_t pt_dim,
                                      const double coord_min[MAX_NDIM],
                                      const double coord_max[MAX_NDIM],
                                      const double enbox_size[MAX_NDIM],
                                      const int64_t grid_size[MAX_NDIM],
                                      const int64_t grid_algo = 0) const {
    int64_t max_grid_size = 0;
    int64_t anchor_npt = 1;
    for (int64_t i = 0; i < pt_dim; i++) {
      max_grid_size = (grid_size[i] > max_grid_size) ? grid_size[i] : max_grid_size;
      anchor_npt *= (grid_size[i] + 1);
    }
    max_grid_size++;

    // 1. Assign anchor points in each dimension
    std::vector<double> anchor_dim(pt_dim * max_grid_size);
    for (int64_t i = 0; i < pt_dim; i++) {
      if (grid_size[i] == 0) {
        const auto offset_i = i * max_grid_size;
        anchor_dim[offset_i] = (coord_min[i] + coord_max[i]) * 0.5;
      }
    }
    if (grid_algo == 0) {  // Default
      const double c0 = 1.0;
      const double c1 = 0.5;
      const double c2 = 0.25;
      for (int64_t i = 0; i < pt_dim; i++) {
        const auto size_i = c0 * enbox_size[i] / ((double)grid_size[i] + c1);
        const auto offset_i = i * max_grid_size;
        for (int64_t j = 0; j <= grid_size[i]; j++) {
          anchor_dim[offset_i + j] = coord_min[i] + c2 * size_i + size_i * (double)j;
        }
      }
    }
    else {  // Chebyshev anchor points
      for (int64_t i = 0; i < pt_dim; i++) {
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
    for (int64_t i = 0; i < pt_dim; i++) {
      stride[i + 1] = stride[i] * (grid_size[i] + 1);
    }
    for (int64_t i = 0; i < anchor_npt; i++) {
      for (int64_t j = 0; j < pt_dim; j++) {
        const int64_t dim_idx = (i / stride[j]) % (grid_size[j] + 1);
        anchor_coord[i].X[j] = anchor_dim[j * max_grid_size + dim_idx];
      }
    }
    return anchor_coord;
  }

  std::vector<int64_t> select_cluster_sample_bodies(
      const std::vector<int64_t>& bodies_loc,
      const int64_t _sample_size, const int64_t sampling_alg) const {
    const int64_t nbodies = bodies_loc.size();
    const int64_t sample_size = std::min(nbodies, _sample_size);
    std::vector<int64_t> samples_loc;
    samples_loc.reserve(sample_size);

    switch (sampling_alg) {
      case 0: {  // Select based on equally spaced indices
        const int64_t d = (int64_t)std::floor((double)nbodies / (double)sample_size);
        int64_t k = 0;
        for (int64_t i = 0; i < sample_size; i++) {
          samples_loc.push_back(bodies_loc[k]);
          k += d;
        }
        break;
      }
      case 1: {  // Random sampling
        static std::mt19937 g(0);  // Use fixed seed for reproducibility
        std::vector<int64_t> random_indices(nbodies, 0);
        for (int64_t i = 0; i < nbodies; i++) {
          random_indices[i] = i;
        }
        // Random shuffle 3 times
        for (int64_t i = 0; i < 3; i++) {
          std::shuffle(random_indices.begin(), random_indices.end(), g);
        }
        // Choose sample based on random indices
        for (int64_t i = 0; i < sample_size; i++) {
          const auto ridx = random_indices[i];
          samples_loc.push_back(bodies_loc[ridx]);
        }
        break;
      }
      case 2: {  // Farthest Point Sampling (FPS)
        std::vector<bool> chosen(nbodies, false);
        // Find center of enclosing box
        double center[MAX_NDIM];
        for (int64_t axis = 0; axis < ndim; axis++) {
          const auto Xmin = get_Xmin(bodies_loc, axis);
          const auto Xmax = get_Xmax(bodies_loc, axis);
          center[axis] = (Xmax + Xmin) / 2.;
        }
        // Start with point closest to the center as pivot
        int64_t pivot = -1;
        double min_dist2 = std::numeric_limits<double>::max();
        for (int64_t i = 0; i < nbodies; i++) {
          const auto dist2_i = dist2(center, bodies[bodies_loc[i]].X);
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
              const auto dist2_i = dist2(bodies[bodies_loc[pivot]].X,
                                         bodies[bodies_loc[i]].X);
              if (dist2_i > max_dist2) {
                max_dist2 = dist2_i;
                farthest_idx = i;
              }
            }
          }
          chosen[farthest_idx] = true;
          pivot = farthest_idx;
        }
        for (int64_t i = 0; i < nbodies; i++) {
          if (chosen[i]) {
            samples_loc.push_back(bodies_loc[i]);
          }
        }
        break;
      }
      case 3: {  // Anchor Net Method
        double Xmax[MAX_NDIM], Xmin[MAX_NDIM], diameter[MAX_NDIM];
        for (int64_t axis = 0; axis < ndim; axis++) {
          Xmin[axis] = get_Xmin(bodies_loc, axis);
          Xmax[axis] = get_Xmax(bodies_loc, axis);
          diameter[axis] = Xmax[axis] - Xmin[axis];
        }
        int64_t grid_size[MAX_NDIM];
        const auto anchor_npt = proportional_int_decompose(ndim, sample_size, diameter, grid_size);
        if (anchor_npt < nbodies) {
          const auto anchor_bodies = build_anchor_grid(ndim, Xmin, Xmax, diameter, grid_size, 0);
          assert(anchor_npt == anchor_bodies.size());
          std::vector<bool> chosen(nbodies, false);
          for (int64_t i = 0; i < anchor_npt; i++) {
            int64_t min_idx = -1;
            double min_dist2 = std::numeric_limits<double>::max();
            for (int64_t j = 0; j < nbodies; j++) {
              const auto dist2_ij = dist2(anchor_bodies[i].X, bodies[bodies_loc[j]].X);
              if (dist2_ij < min_dist2) {
                min_dist2 = dist2_ij;
                min_idx = j;
              }
            }
            chosen[min_idx] = true;
          }
          for (int64_t i = 0; i < nbodies; i++) {
            if (chosen[i]) {
              samples_loc.push_back(bodies_loc[i]);
            }
          }
        }
        else {
          samples_loc = bodies_loc;
        }
        break;
      }
    }
    return samples_loc;
  }

 public:
  Domain(const int64_t _N, const int64_t _ndim)
      : N(_N), ndim(_ndim) {
    if (ndim < 1 || ndim > MAX_NDIM) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int64_t get_cell_loc(const int64_t index, const int64_t level) const {
    return (1 << level) - 1 + index;
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
              std::find(cells[i1].far_list.begin(), cells[i1].far_list.end(), j1) !=
              cells[i1].far_list.end();
          const bool i1_j2_found =
              std::find(cells[i1].far_list.begin(), cells[i1].far_list.end(), j2) !=
              cells[i1].far_list.end();
          const bool i2_j1_found =
              std::find(cells[i2].far_list.begin(), cells[i2].far_list.end(), j1) !=
              cells[i2].far_list.end();
          const bool i2_j2_found =
              std::find(cells[i2].far_list.begin(), cells[i2].far_list.end(), j2) !=
              cells[i2].far_list.end();
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
    sort_interaction_lists();
  }

  void select_sample_bodies(const int64_t sample_self_size,
                            const int64_t sample_far_size,
                            const int64_t sampling_alg = 0) {
    // Bottom-up pass to select cell's sample bodies
    for (int64_t level = tree_height; level > 0; level--) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        std::vector<int64_t> sample_candidates;
        if (level == tree_height) {
          // Leaf level: candidates are from its bodies
          sample_candidates = cell.get_bodies();
        }
        else {
          // Non-leaf level: candidates are from children's samples
          const auto& child1 = cells[cell.child];
          const auto& child2 = cells[cell.child + 1];
          sample_candidates.insert(sample_candidates.end(),
                                   child1.sample_bodies.begin(),
                                   child1.sample_bodies.end());
          sample_candidates.insert(sample_candidates.end(),
                                   child2.sample_bodies.begin(),
                                   child2.sample_bodies.end());
        }
        cell.sample_bodies = select_cluster_sample_bodies(sample_candidates,
                                                          sample_self_size,
                                                          sampling_alg);
      }
    }
    // Top-down pass to select cell's farfield sample
    for (int64_t level = 1; level <= tree_height; level++) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        const auto& parent = cells[cell.parent];
        if (cell.far_list.size() > 0 || parent.sample_farfield.size() > 0) {
          // If node has non-empty farfield
          auto farfield = parent.sample_farfield;
          for (auto far_loc: cell.far_list) {
            farfield.insert(farfield.end(),
                            cells[far_loc].sample_bodies.begin(),
                            cells[far_loc].sample_bodies.end());
          }
          cell.sample_farfield = select_cluster_sample_bodies(farfield,
                                                              sample_far_size,
                                                              sampling_alg);
          std::sort(cell.sample_farfield.begin(), cell.sample_farfield.end());
        }
      }
    }
  }

  void initialize_unit_circular_mesh() {
    if (ndim == 2) {
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
    if (ndim == 2) {
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

  void initialize_starsh_uniform_grid() {
    const int64_t side = std::ceil(
        std::pow((double)N, 1. / (double)ndim)); // size of each side of the grid
    int64_t total = side;
    for (int64_t i = 1; i < ndim; i++) {
      total *= side;
    }

    const int64_t ncoords = ndim * side;
    std::vector<double> coord(ncoords);
    for (int64_t i = 0; i < side; i++) {
      const double val = (double)i / side;
      for (int64_t j = 0; j < ndim; j++) {
        coord[j * side + i] = val;
      }
    }

    std::vector<int64_t> pivot(ndim, 0);
    int64_t k = 0;
    for (int64_t i = 0; i < N; i++) {
      std::vector<double> points(ndim);
      for (k = 0; k < ndim; k++) {
        points[k] = coord[pivot[k] + k * side];
      }
      bodies.emplace_back(Body(points, 0));

      k = ndim - 1;
      pivot[k]++;
      while(pivot[k] == side) {
        pivot[k] = 0;
        if (k > 0) {
          k--;
          pivot[k]++;
        }
      }
    }
  }

  void print_bodies_to_file(const std::string& file_name) const {
    const std::vector<char> axis{'x', 'y', 'z'};

    std::ofstream file;
    file.open(file_name, std::ios::out);
    for (int64_t k = 0; k < ndim; k++) {
      if (k > 0) file << ",";
      file << axis[k];
    }
    file << std::endl;

    for (int64_t i = 0; i < N; i++) {
      for (int64_t k = 0; k < ndim; k++) {
        if (k > 0) file << ",";
        file << bodies[i].X[k];
      }
      file << std::endl;
    }

    file.close();
  }
};

} // namespace Hatrix

