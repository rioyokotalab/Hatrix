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

using kernel_func_t =
    std::function<double(const Body& source, const Body& target)>;

class Domain {
 public:
  int64_t N, ndim;
  int64_t ncells, tree_height;
  std::vector<Body> bodies;
  std::vector<Cell> cells;

 private:
  double get_Xmin(const std::vector<Body>& bodies_arr,
                  const std::vector<int64_t>& bodies_idx,
                  const int64_t axis) const {
    assert(axis < ndim);
    double Xmin = bodies_arr[bodies_idx[0]].X[axis];
    for (int64_t i = 1; i < bodies_idx.size(); i++) {
      Xmin = std::min(Xmin, bodies_arr[bodies_idx[i]].X[axis]);
    }
    return Xmin;
  }

  double get_Xmax(const std::vector<Body>& bodies_arr,
                  const std::vector<int64_t>& bodies_idx,
                  const int64_t axis) const {
    assert(axis < ndim);
    double Xmax = bodies_arr[bodies_idx[0]].X[axis];
    for (int64_t i = 1; i < bodies_idx.size(); i++) {
      Xmax = std::max(Xmax, bodies_arr[bodies_idx[i]].X[axis]);
    }
    return Xmax;
  }

  double get_Xsum(const std::vector<Body>& bodies_arr,
                  const std::vector<int64_t>& bodies_idx,
                  const int64_t axis) const {
    assert(axis < ndim);
    double sum = 0;
    for (int64_t i = 0; i < bodies_idx.size(); i++) {
      sum += bodies_arr[bodies_idx[i]].X[axis];
    }
    return sum;
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
    double radius_max = 0.;
    int64_t sort_axis = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      const auto Xmin = get_Xmin(bodies, cell.get_bodies(), axis);
      const auto Xmax = get_Xmax(bodies, cell.get_bodies(), axis);
      const auto Xsum = get_Xsum(bodies, cell.get_bodies(), axis);
      const auto diam = Xmax - Xmin;
      cell.center[axis] = Xsum / (double)cell.nbodies;
      cell.radius[axis] = diam / 2.;

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
    orthogonal_recursive_bisection(left, mid, leaf_size, level + 1, block_index << 1);
    orthogonal_recursive_bisection(mid, right, leaf_size, level + 1, (block_index << 1) + 1);
  }

  bool is_well_separated(const Cell& source, const Cell& target,
                         const double theta) const {
    const auto distance = dist2(source.center, target.center);
    const auto source_size = source.get_radius();
    const auto target_size = target.get_radius();
    return (distance > (theta * (source_size + target_size)));
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

  // Remove element by value from STL container
  void erase_by_value(std::vector<int64_t>& vec, const int64_t value) {
    vec.erase(std::remove(vec.begin(), vec.end(), value), vec.end());
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
  std::vector<Body> build_anchor_grid(const double coord_min[MAX_NDIM],
                                      const double coord_max[MAX_NDIM],
                                      const double enbox_size[MAX_NDIM],
                                      const int64_t grid_size[MAX_NDIM],
                                      const int64_t grid_algo = 0) const {
    int64_t max_grid_size = 0;
    int64_t anchor_npt = 1;
    for (int64_t i = 0; i < ndim; i++) {
      max_grid_size = (grid_size[i] > max_grid_size) ? grid_size[i] : max_grid_size;
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
          anchor_dim[offset_i + j] = coord_min[i] + c2 * size_i + size_i * (double)j;
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

  std::vector<int64_t> select_sample_bodies(const std::vector<Body>& bodies_arr,
                                            const std::vector<int64_t>& bodies_idx,
                                            const int64_t sample_size,
                                            const int64_t sampling_algo,
                                            const int64_t grid_algo = 0) const {
    const int64_t nbodies = bodies_idx.size();
    if (sample_size >= nbodies) {
      return bodies_idx;
    }
    std::vector<int64_t> sample_idx;
    switch (sampling_algo) {
      case 0: {  // Select based on equally spaced indices
        const int64_t d = (int64_t)std::floor((double)nbodies / (double)sample_size);
        int64_t k = 0;
        sample_idx.reserve(sample_size);
        for (int64_t i = 0; i < sample_size; i++) {
          sample_idx.push_back(bodies_idx[k]);
          k += d;
        }
        break;
      }
      case 1: {  // Random sampling
        static std::mt19937 g(N);  // Use N as seed for reproducibility
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
          const auto Xsum = get_Xsum(bodies, bodies_idx, axis);
          center[axis] = Xsum / (double)bodies_idx.size();
        }
        // Start with point closest to the centroid as pivot
        int64_t pivot = -1;
        double min_dist2 = std::numeric_limits<double>::max();
        for (int64_t i = 0; i < nbodies; i++) {
          const auto dist2_i = dist2(center, bodies_arr[bodies_idx[i]].X);
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
              const auto dist2_i = dist2(bodies_arr[bodies_idx[pivot]].X,
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
          coord_min[axis] = get_Xmin(bodies, bodies_idx, axis);
          coord_max[axis] = get_Xmax(bodies, bodies_idx, axis);
          diameter[axis] = coord_max[axis] - coord_min[axis];
        }
        int64_t grid_size[MAX_NDIM];
        const auto anchor_npt = proportional_int_decompose(ndim, sample_size, diameter, grid_size);
        if (anchor_npt < nbodies) {
          const auto anchor_bodies = build_anchor_grid(coord_min, coord_max, diameter, grid_size, grid_algo);
          std::vector<bool> chosen(nbodies, false);
          for (int64_t i = 0; i < anchor_npt; i++) {
            int64_t min_idx = -1;
            double min_dist2 = std::numeric_limits<double>::max();
            for (int64_t j = 0; j < nbodies; j++) {
              const auto dist2_ij = dist2(anchor_bodies[i].X, bodies_arr[bodies_idx[j]].X);
              if (dist2_ij < min_dist2) {
                min_dist2 = dist2_ij;
                min_idx = j;
              }
            }
            chosen[min_idx] = true;
          }
          sample_idx.reserve(anchor_npt);
          for (int64_t i = 0; i < nbodies; i++) {
            if (chosen[i]) {
              sample_idx.push_back(bodies_idx[i]);
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

 public:
  Domain(const int64_t _N, const int64_t _ndim)
      : N(_N), ndim(_ndim) {
    if (ndim < 1 || ndim > MAX_NDIM) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  int64_t get_cell_idx(const int64_t block_index, const int64_t level) const {
    return (1 << level) - 1 + block_index;
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

  int64_t adaptive_anchor_grid_size(kernel_func_t kernel_function,
                                    const int64_t leaf_size,
                                    const double theta,
                                    const double ID_tol,
                                    const double stop_tol) {
    double L = 0;
    for (int64_t axis = 0; axis < ndim; axis++) {
      const auto Xmin = get_Xmin(bodies, cells[0].get_bodies(), axis);
      const auto Xmax = get_Xmax(bodies, cells[0].get_bodies(), axis);
      L = std::max(L, Xmax - Xmin);
    }
    L /= 6.0; // Extra step taken from H2Pack MATLAB reference

    // Create two sets of points in unit boxes
    const auto box_nbodies = 2 * leaf_size;
    const auto shift = L * (theta == 0 ? 1 : theta);
    std::vector<Body> box1, box2;
    std::vector<int64_t> box_idx;
    box_idx.resize(box_nbodies);
    box1.resize(box_nbodies);
    box2.resize(box_nbodies);
    std::mt19937 gen(1234); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0, 1);
    for (int64_t i = 0; i < box_nbodies; i++) {
      box_idx[i] = i;
      for (int64_t axis = 0; axis < ndim; axis++) {
        box1[i].X[axis] = L * dist(gen);
        box2[i].X[axis] = L * dist(gen) + shift;
      }
    }

    auto generate_matrix = [kernel_function](const std::vector<Body>& source,
                                             const std::vector<Body>& target,
                                             const std::vector<int64_t>& source_idx,
                                             const std::vector<int64_t>& target_idx) {
      Matrix out(source_idx.size(), target_idx.size());
      for (int64_t i = 0; i < out.rows; i++) {
        for (int64_t j = 0; j < out.cols; j++) {
          out(i, j) = kernel_function(source[source_idx[i]], target[target_idx[j]]);
        }
      }
      return out;
    };
    Matrix A = generate_matrix(box1, box2, box_idx, box_idx);
    // Find anchor grid size r by checking approximation error to A
    double box2_xmin[MAX_NDIM], box2_xmax[MAX_NDIM];
    for (int64_t i = 0; i < box_nbodies; i++) {
      for (int64_t axis = 0; axis < ndim; axis++) {
        if (i == 0) {
          box2_xmin[axis] = box2[i].X[axis];
          box2_xmax[axis] = box2[i].X[axis];
        }
        else {
          box2_xmin[axis] = std::min(box2_xmin[axis], box2[i].X[axis]);
          box2_xmax[axis] = std::max(box2_xmax[axis], box2[i].X[axis]);
        }
      }
    }
    double box2_size[MAX_NDIM];
    for (int64_t axis = 0; axis < ndim; axis++) {
      box2_size[axis] = box2_xmax[axis] - box2_xmin[axis];
    }
    int64_t r = 1;
    bool stop = false;
    while (!stop) {
      const auto box2_sample = select_sample_bodies(box2, box_idx, r, 3, 0);
      // A1 = kernel(box1, box2_sample)
      Matrix A1 = generate_matrix(box1, box2, box_idx, box2_sample);
      Matrix U;
      std::vector<int64_t> ipiv_rows;
      std::tie(U, ipiv_rows) = error_id_row(A1, ID_tol, false);
      int64_t rank = U.cols;
      // A2 = A(ipiv_rows[:rank], :)
      Matrix A2(rank, A.cols);
      for (int64_t i = 0; i < rank; i++) {
        for (int64_t j = 0; j < A.cols; j++) {
          A2(i, j) = A(ipiv_rows[i], j);
        }
      }
      Matrix UxA2 = matmul(U, A2);
      const double error = norm(A - UxA2);
      if ((error < stop_tol) || ((box_nbodies - box2_sample.size()) < (box_nbodies / 10))) {
        stop = true;
      }
      else {
        r++;
      }
    }
    return r;
  }

  void build_sample_bodies(const int64_t sample_self_size,
                           const int64_t sample_far_size,
                           const int64_t sampling_algo,
                           const int64_t initial_far_sp) {
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
        cell.sample_bodies = select_sample_bodies(bodies, initial_sample,
                                                  sample_self_size,
                                                  sampling_algo, 0);
      }
    }
    // Top-down pass to select cell's farfield sample
    for (int64_t level = 1; level <= tree_height; level++) {
      const auto level_ncells = (int64_t)1 << level;
      const auto level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; node++) {
        auto& cell = cells[level_offset + node];
        if (cell.far_list.size() == 0) continue;

        int64_t far_nbodies = 0;
        for (const auto far_idx: cell.far_list) {
          far_nbodies += cells[far_idx].sample_bodies.size();
        }
        const auto& parent = cells[cell.parent];
        auto initial_sample = parent.sample_farfield;
        if ((sampling_algo != 3) || (initial_far_sp == 0) ||
            (parent.sample_farfield.size() > far_nbodies)) {
          // Put all sample_bodies of far nodes (in the same level) into initial sample
          for (const auto far_idx: cell.far_list) {
            initial_sample.insert(initial_sample.end(),
                                  cells[far_idx].sample_bodies.begin(),
                                  cells[far_idx].sample_bodies.end());
          }
        }
        else {
          // Put only samples of far nodes (in the same level) into initial sample
          // So that the initial sample contain a the same proportion of
          // current level far-bodies and parent's far-bodies
          const int64_t num_far_nodes = cell.far_list.size();
          // 1. Find centroid of each far-node's sample bodies
          std::vector<double> centers(ndim * num_far_nodes);  // column major, each column is a coordinate
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
              far_node_xmin[axis] = i == 0 ? Xmin : std::min(far_node_xmin[axis], Xmin);
              far_node_xmax[axis] = i == 0 ? Xmax : std::max(far_node_xmax[axis], Xmax);
            }
          }
          double far_node_box_size[MAX_NDIM];
          for (int64_t axis = 0; axis < ndim; axis++) {
            far_node_box_size[axis] = far_node_xmax[axis] - far_node_xmin[axis];
          }
          int64_t far_node_grid_size[MAX_NDIM];
          proportional_int_decompose(ndim, sample_far_size, far_node_box_size, far_node_grid_size);
          const auto anchor_bodies = build_anchor_grid(far_node_xmin,
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
            double min_dist = dist2(anchor_bodies[k].X, centers.data());
            for (int64_t i = 1; i < num_far_nodes; i++) {
              const auto dist2_i = dist2(anchor_bodies[k].X, centers.data() + i * ndim);
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
              const auto dist_idx = dist2(anchor_bodies[k].X, bodies[body_idx].X);
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
              const auto dist_idx = dist2(centers.data() + i * ndim, bodies[body_idx].X);
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
        cell.sample_farfield = select_sample_bodies(bodies, initial_sample,
                                                    sample_far_size,
                                                    sampling_algo, 1);
      }
    }
  }

  int64_t get_max_farfield_size() const {
    int64_t max_farfield_size = 0;
    for (int64_t i = 0; i < ncells; i++) {
      max_farfield_size = std::max(max_farfield_size, (int64_t)cells[i].sample_farfield.size());
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

