#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Particle.hpp"
#include "Box.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Hatrix {

class Domain {
 public:
  int64_t N, ndim;
  std::vector<Particle> particles;
  std::vector<Box> boxes;

 private:
  // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
  void orthogonal_recursive_bisection_1dim(const int64_t left, const int64_t right,
                                           const std::string morton_index, const int64_t nleaf) {
    // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
    std::sort(particles.begin() + left, particles.begin() + right,
              [](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[0] < rhs.coords[0];
              });

    const int64_t num_particles = right - left;
    if (num_particles <= nleaf) {
      const double start_coord_x = particles[left].coords[0];
      const double end_coord_x = particles[right - 1].coords[0];
      const double center_x = (start_coord_x + end_coord_x) / 2;
      const double diameter = end_coord_x - start_coord_x;
      boxes.emplace_back(Box(num_particles, left, right - 1,
                             diameter, center_x, morton_index));
    }
    else {  // Recurse and split again
      const int64_t mid = (left + right) / 2;
      // First half
      orthogonal_recursive_bisection_1dim(left, mid, morton_index + "0", nleaf);
      // Second half
      orthogonal_recursive_bisection_1dim(mid, right, morton_index + "1", nleaf);
    }
  }

  void orthogonal_recursive_bisection_2dim(const int64_t left, const int64_t right,
                                           const std::string morton_index, const int64_t nleaf,
                                           const int64_t axis) {
    // Sort particles by axis
    std::sort(particles.begin() + left, particles.begin() + right,
              [axis](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[axis] < rhs.coords[axis];
              });

    const int64_t num_particles = right - left;
    if (num_particles <= nleaf) {
      if (axis == ndim-1) {
        const int64_t begin_index = left;
        const int64_t end_index = right - 1;

        double x_max, y_max, x_min, y_min;
        x_min = x_max = particles[begin_index].coords[0];
        y_min = y_max = particles[begin_index].coords[1];
        for (int64_t i = begin_index + 1; i <= end_index; i++) {
          x_min = std::min(x_min, particles[i].coords[0]);
          y_min = std::min(y_min, particles[i].coords[1]);

          x_max = std::max(x_max, particles[i].coords[0]);
          y_max = std::max(y_max, particles[i].coords[1]);
        }
        const double dx = x_max - x_min;
        const double dy = y_max - y_min;
        const double diameter = (dx * dx) + (dy * dy);
        const double center_x = (x_min + x_max) / 2.;
        const double center_y = (y_min + y_max) / 2.;

        boxes.emplace_back(Box(num_particles, begin_index, end_index,
                               diameter, center_x, center_y, morton_index));
      }
      else {
        orthogonal_recursive_bisection_2dim(left, right, morton_index,
                                            nleaf, (axis + 1) % ndim);
      }
    }
    else {
      const int64_t mid = (left + right) / 2;
      orthogonal_recursive_bisection_2dim(left, mid, morton_index + "0",
                                          nleaf, (axis + 1) % ndim);
      orthogonal_recursive_bisection_2dim(mid, right, morton_index + "1",
                                          nleaf, (axis + 1) % ndim);
    }
  }

  void orthogonal_recursive_bisection_3dim(const int64_t left, const int64_t right,
                                           const std::string morton_index, const int64_t nleaf,
                                           const int64_t axis) {
    // Sort particles by axis
    std::sort(particles.begin() + left, particles.begin() + right,
              [axis](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[axis] < rhs.coords[axis];
              });

    const int64_t num_particles = right - left;
    if (num_particles <= nleaf) {
      if (axis == ndim-1) {
        const int64_t begin_index = left;
        const int64_t end_index = right - 1;

        double x_max, y_max, z_max;
        double x_min, y_min, z_min;
        x_min = x_max = particles[begin_index].coords[0];
        y_min = y_max = particles[begin_index].coords[1];
        z_min = z_max = particles[begin_index].coords[2];
        for (int64_t i = begin_index+1; i <= end_index; i++) {
          x_min = std::min(x_min, particles[i].coords[0]);
          y_min = std::min(y_min, particles[i].coords[1]);
          z_min = std::min(z_min, particles[i].coords[2]);

          x_max = std::max(x_max, particles[i].coords[0]);
          y_max = std::max(y_max, particles[i].coords[1]);
          z_max = std::max(z_max, particles[i].coords[2]);
        }
        const double dx = x_max - x_min;
        const double dy = y_max - y_min;
        const double dz = z_max - z_min;
        const double diameter = (dx * dx) + (dy * dy) + (dz * dz);
        const double center_x = (x_min + x_max) / 2.;
        const double center_y = (y_min + y_max) / 2.;
        const double center_z = (z_min + z_max) / 2.;

        boxes.emplace_back(Box(num_particles, begin_index, end_index,
                               diameter, center_x, center_y, center_z,
                               morton_index));
      }
      else {
        // Sort by next axis
        orthogonal_recursive_bisection_3dim(left, right, morton_index,
                                            nleaf, (axis+1) % ndim);
      }
    }
    else {
      const int64_t mid = (left + right) / 2;
      orthogonal_recursive_bisection_3dim(left, mid, morton_index + "0",
                                          nleaf, (axis+1) % ndim);
      orthogonal_recursive_bisection_3dim(mid, right, morton_index + "1",
                                          nleaf, (axis+1) % ndim);
    }
  }

 public:
  Domain(const int64_t _N, const int64_t _ndim)
      : N(_N), ndim(_ndim) {
    if (ndim < 1) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void generate_unit_circular_mesh() {
    if (ndim == 2) {
      // Generate a unit circle with N points on the circumference.
      for (int64_t i = 0; i < N; i++) {
        const double theta = (i * 2.0 * M_PI) / (double)N;
        const double x = cos(theta);
        const double y = sin(theta);
        const double value = (double)i / (double)N;

        particles.emplace_back(Particle(x, y, value));
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
        particles.emplace_back(Particle(x, y, z, value));
      }
    }
  }

  void generate_unit_cubical_mesh() {
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
        particles.emplace_back(Particle(x, y, value));
      }
      for (; i < left; i++) {
        const double x = -a;
        const double y = a - 2.0 * a * (i - top) / (left - top);
        const double value = (double)i / (double)N;
        particles.emplace_back(Particle(x, y, value));
      }
      for (; i < bottom; i++) {
        const double x = -a + 2.0 * a * (i - left) / (bottom - left);
        const double y = -a;
        const double value = (double)i / (double)N;
        particles.emplace_back(Particle(x, y, value));
      }
      for (; i < N; i++) {
        const double x = a;
        const double y = -a + 2.0 * a * (i - bottom) / (N - bottom);
        const double value = (double)i / (double)N;
        particles.emplace_back(Particle(x, y, value));
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
        particles.emplace_back(Particle(px, py, pz, value));
      }
    }
  }

  void generate_starsh_uniform_grid() {
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
      particles.emplace_back(Particle(points, 0));

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

  void divide_domain_and_create_particle_boxes(const int64_t nleaf) {
    if (ndim == 1) {
      orthogonal_recursive_bisection_1dim(0, N, "", nleaf);
    }
    else if (ndim == 2) {
      orthogonal_recursive_bisection_2dim(0, N, "", nleaf, 0);
    }
    else if (ndim == 3) {
      orthogonal_recursive_bisection_3dim(0, N, "", nleaf, 0);
    }
  }

  // Check admissibility on leaf-level boxes
  bool check_admis(const double theta,
                   const int64_t source, const int64_t target) const {
    const auto min_diameter = std::min(boxes[source].diameter, boxes[target].diameter);
    const auto distance = boxes[source].distance_from(boxes[target]);
    return distance > (theta * min_diameter);
  }

  void print_to_file(const std::string& file_name) const {
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
        file << particles[i].coords[k];
      }
      file << std::endl;
    }

    file.close();
  }
};

} // namespace Hatrix

