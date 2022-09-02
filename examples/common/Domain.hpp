#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
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
  uint64_t N, ndim;
  std::vector<Body> bodies;
  uint64_t ncells, tree_height;
  std::vector<Cell> cells;

 private:
  double get_Xmax(const uint64_t body_start, const uint64_t body_end,
                  const uint64_t axis) const {
    assert(axis < ndim);
    double Xmax = bodies[body_start].X[axis];
    for (uint64_t i = body_start + 1; i <= body_end ; i++) {
      Xmax = std::max(Xmax, bodies[i].X[axis]);
    }
    return Xmax;
  }

  double get_Xmin(const uint64_t body_start, const uint64_t body_end,
                  const uint64_t axis) const {
    assert(axis < ndim);
    double Xmin = bodies[body_start].X[axis];
    for (uint64_t i = body_start + 1; i <= body_end ; i++) {
      Xmin = std::min(Xmin, bodies[i].X[axis]);
    }
    return Xmin;
  }

  void orthogonal_recursive_bisection(
      const uint64_t left, const uint64_t right, const uint64_t leaf_size,
      const uint64_t level, const uint64_t index) {
    // Initialize cell
    const auto loc = get_cell_loc(index, level);
    cells[loc].body = left;
    cells[loc].nbodies = right - left;
    cells[loc].level = level;
    cells[loc].index = index;
    double radius_max = 0.;
    uint64_t sort_axis = 0;
    for (uint64_t axis = 0; axis < ndim; axis++) {
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

 public:
  Domain(const uint64_t _N, const uint64_t _ndim)
      : N(_N), ndim(_ndim) {
    if (ndim < 1 || ndim > 3) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  uint64_t get_cell_loc(const uint64_t index, const uint64_t level) const {
    return (1 << level) - 1 + index;
  }

  void build_tree(const uint64_t leaf_size) {
    // Assume balanced binary tree
    tree_height = (uint64_t)std::log2((double)N / (double)leaf_size);
    const uint64_t nleaf_cells = (uint64_t)1 << tree_height;
    ncells = 2 * nleaf_cells - 1;
    // Initialize empty cells
    cells.resize(ncells);
    // Partition
    orthogonal_recursive_bisection(0, N, leaf_size, 0, 0);
  }

  void build_interactions(const double theta) {
    dual_tree_traversal(cells[0], cells[0], theta);
    // Sort cell locations in interacion lists
    for (auto& cell: cells) {
      std::sort(cell.near_list.begin(), cell.near_list.end());
      std::sort(cell.far_list.begin(), cell.far_list.end());
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

