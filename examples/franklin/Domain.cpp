#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

namespace Hatrix {
  void
  Domain::print_file(std::string file_name) {
    std::vector<char> coords{'x', 'y', 'z'};

    std::ofstream file;
    file.open(file_name, std::ios::app | std::ios::out);
    for (int64_t k = 0; k < ndim; ++k) {
      file << coords[k] << ",";
    }
    file << std::endl;

    for (int64_t i = 0; i < N; ++i) {
      for (int64_t k = 0; k < ndim; ++k) {
        file << particles[i].coords[k] << ",";
      }
      file << std::endl;
    }

    file.close();
  }

  // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
  void
  Domain::orthogonal_recursive_bisection_1dim(int64_t start,
                                              int64_t end,
                                              std::string morton_index,
                                              int64_t nleaf) {
    // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
    std::sort(particles.begin()+start,
              particles.begin()+end, [](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[0] < rhs.coords[0];
              });

    int64_t num_points = end - start;
    // found a box with the correct number of points.
    if (num_points <= nleaf) {
      auto start_coord_x = particles[start].coords[0];
      auto end_coord_x = particles[end-1].coords[0];
      auto center_x = (start_coord_x + end_coord_x) / 2;
      auto diameter = end_coord_x - start_coord_x;
      boxes.push_back(Box(diameter, center_x, start, end-1, morton_index, num_points));
    }
    else {                    // recurse further and split again.
      int64_t middle = (start + end) / 2;
      // first half
      orthogonal_recursive_bisection_1dim(start, middle, morton_index + "0", nleaf);
      // second half
      orthogonal_recursive_bisection_1dim(middle, end, morton_index + "1", nleaf);
    }
  }

  void
  Domain::orthogonal_recursive_bisection_2dim(int64_t start,
                                              int64_t end,
                                              std::string morton_index,
                                              int64_t nleaf,
                                              int64_t axis) {
    std::sort(particles.begin() + start,
              particles.begin() + end, [&](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[axis] < rhs.coords[axis];
              });

    int64_t num_points = end - start;
    if (num_points <= nleaf) {
      if (axis == ndim-1) {
        int64_t start_index = start;
        int64_t end_index = end - 1;

        double diameter = 0;
        for (int64_t k = 0; k < ndim; ++k) {
          diameter += pow(particles[start_index].coords[k] - particles[end_index].coords[k], 2);
        }
        diameter = std::sqrt(diameter);

        double center_x = (particles[start_index].coords[0] + particles[end_index].coords[0]) / 2;
        double center_y = (particles[start_index].coords[1] + particles[end_index].coords[1]) / 2;

        boxes.push_back(Box(diameter,
                            center_x,
                            center_y,
                            start_index,
                            end_index,
                            morton_index,
                            num_points));
      }
      else {
        orthogonal_recursive_bisection_2dim(start, end, morton_index, nleaf, (axis + 1) % ndim);
      }
    }
    else {
      int64_t middle = (start + end) / 2;
      orthogonal_recursive_bisection_2dim(start,
                                          middle,
                                          morton_index + "0",
                                          nleaf,
                                          (axis + 1) % ndim);
      orthogonal_recursive_bisection_2dim(middle, end, morton_index + "1", nleaf, (axis + 1) % ndim);
    }
  }

  void
  Domain::orthogonal_recursive_bisection_3dim(int64_t start, int64_t end, std::string morton_index,
                                              int64_t nleaf, int64_t axis) {
    std::sort(particles.begin() + start,
              particles.begin() + end,
              [&](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[axis] < rhs.coords[axis];
              });
    int64_t num_points = end - start;
    if (num_points <= nleaf) {
      if (axis == ndim-1) {
        int64_t start_index = start;
        int64_t end_index = end - 1;

        double diameter = 0;
        for (int64_t k = 0; k < ndim; ++k) {
          diameter += pow(particles[start_index].coords[k] - particles[end_index].coords[k], 2);
        }
        diameter = std::sqrt(diameter);

        double center_x = (particles[start_index].coords[0] + particles[end_index].coords[0]) / 2;
        double center_y = (particles[start_index].coords[1] + particles[end_index].coords[1]) / 2;
        double center_z = (particles[start_index].coords[2] + particles[end_index].coords[2]) / 2;

        boxes.push_back(Box(diameter,
                            center_x,
                            center_y,
                            center_z,
                            start_index,
                            end_index,
                            morton_index,
                            num_points));
      }
      else {
        orthogonal_recursive_bisection_3dim(start, end, morton_index, nleaf, (axis+1) % ndim);
      }
    }
    else {
      int64_t middle = (start + end) / 2;
      orthogonal_recursive_bisection_3dim(start, middle, morton_index + "0", nleaf, (axis+1)%ndim);
      orthogonal_recursive_bisection_3dim(middle, end, morton_index + "1", nleaf, (axis+1)%ndim);
    }
  }


  Domain::Domain(int64_t N, int64_t ndim) : N(N), ndim(ndim) {
    if (ndim <= 0) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      abort();
    }
  }

  void Domain::generate_grid_particles() {
    int64_t side = ceil(pow(N, 1.0 / ndim)); // size of each size of the grid.
    int64_t total = side;
    for (int64_t i = 1; i < ndim; ++i) { total *= side; }

    int64_t ncoords = ndim * side;
    std::vector<double> coord(ncoords);

    for (int64_t i = 0; i < side; ++i) {
      double val = double(i) / side;
      for (int64_t j = 0; j < ndim; ++j) {
        coord[j * side + i] = val;
      }
    }

    std::vector<int64_t> pivot(ndim, 0);

    int64_t k = 0;
    for (int64_t i = 0; i < N; ++i) {
      std::vector<double> points(ndim);
      for (k = 0; k < ndim; ++k) {
        points[k] = coord[pivot[k] + k * side];
      }
      particles.push_back(Hatrix::Particle(points, 0));

      k = ndim - 1;
      pivot[k]++;

      while(pivot[k] == side) {
        pivot[k] = 0;
        if (k > 0) {
          --k;
          pivot[k]++;
        }
      }
    }
  }

  void Domain::generate_circular_particles(double min_val, double max_val) {
    double range = max_val - min_val;

    if (ndim == 1) {
      auto vec = equally_spaced_vector(N, min_val, max_val);
      for (int64_t i = 0; i < N; ++i) {
        particles.push_back(Hatrix::Particle(vec[i], min_val + (double(i) / double(range))));
      }
    }
    else if (ndim == 2) {
      // Generate a unit circle with N points on the circumference.
      // std::random_device rd;  // Will be used to obtain a seed for the random number engine
      std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with rd()
      std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
      double radius = 1.0;
      for (int64_t i = 1; i < N; ++i) {
        double theta = (i * 2.0 * M_PI) / N ;
        double x = radius * cos(theta);
        double y = radius * sin(theta);

        particles.push_back(Hatrix::Particle(x, y, min_val + (double(i) / double(range))));
      }

      particles.push_back(Hatrix::Particle(1, -1e-8, 2.0));
      // particles.push_bacK(Hatrix::Particle());
    }
    else if (ndim == 3) {
      // Generate a unit sphere geometry with N points on the surface.
      // http://www.cpp.re/forum/windows/262648/
      // https://neil-strickland.staff.shef.ac.uk/courses/MAS243/lectures/handout10.pdf
      // std::random_device rd;  // Will be used to obtain a seed for the random number engine
      std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with 1 every time.
      std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);

      double radius = 1.0;

      for (int64_t i = 0; i < N; ++i) {
        // double phi = dis(gen);
        // double theta = dis(gen);
        double phi = dis(gen);
        double theta = dis(gen);

        double x = radius * sin(phi) * cos(theta);
        double y = radius * sin(phi) * sin(theta);
        double z = radius * cos(phi);

        particles.push_back(Hatrix::Particle(x, y, z,
                                             min_val + (double(i) / double(range))));
      }
    }
  }

  void Domain::divide_domain_and_create_particle_boxes(int64_t nleaf) {
    if (ndim == 1) {
      orthogonal_recursive_bisection_1dim(0, N, std::string(""), nleaf);
    }
    else if (ndim == 2) {
      orthogonal_recursive_bisection_2dim(0, N, std::string(""), nleaf, 0);
    }
    else if (ndim == 3) {
      orthogonal_recursive_bisection_3dim(0, N, std::string(""), nleaf, 0);
    }
  }

  void
  Domain::read_col_file_3d(const std::string& geometry_file) {
    std::ifstream file;
    file.open(geometry_file, std::ios::in);
    double x, y, z, kmeans_index;

    for (int64_t line = 0; line < N; ++line) {
      file >> x >> y >> z >> kmeans_index;
      particles.push_back(Hatrix::Particle(x, y, z, kmeans_index));
    }

    file.close();
  }

  int
  Domain::get_quadrant(std::vector<double>& p_coords,
                       std::vector<double>& c_coords) {
    int quadrant = 0;
    if (ndim == 2) {
      quadrant = (p_coords[0] < c_coords[0]) +
        ((p_coords[1] < c_coords[1]) << 1);
    }
    else if (ndim == 3) {
      quadrant = p_coords[0] < c_coords[0] +
        ((p_coords[1] < c_coords[1]) << 1) +
        ((p_coords[2] < c_coords[2]) << 2);
    }

    return quadrant;
  }

  void
  Domain::split_cell(Cell* cell, int64_t pstart, int64_t pend,
                     const int64_t max_nleaf,
                     std::vector<Hatrix::Particle>& bodies,
                     std::vector<Hatrix::Particle>& buffer,
                     bool direction) {
    // sort particles into quadrants
    int64_t quadrants = pow(2, ndim);
    std::vector<int64_t> sizes(quadrants, 0);
    std::vector<int64_t> offsets(quadrants+1, 0);
    int64_t cell_particles = pend - pstart;

    if (cell_particles <= max_nleaf) {
      if (direction) {
        for (int64_t i = pstart; i < pend; ++i) {
          for (int64_t k = 0; k < ndim; ++k) {
            buffer[i].coords[k] = bodies[i].coords[k];
          }
        }
      }
      return;
    }

    for (int i = 0; i < quadrants; ++i) {
      offsets[i] = i * (cell_particles / quadrants);
    }

    // count particles in quadrants
    for (int64_t i = pstart; i < pend; ++i) {
      int quadrant = get_quadrant(particles[i].coords,
                                  cell->center);
      sizes[quadrant]++;
    }

    int64_t offset = pstart;
    for (int64_t i = 0; i < quadrants; ++i) {
      offsets[i] = offset;
      offset += sizes[i];
    }
    offsets[quadrants] = offset;

    std::vector<int64_t> counter(quadrants, 0); // storage of counters in offsets
    for (int64_t i = 0; i < quadrants; ++i) {
      counter[i] = offsets[i];
    }

    // sort bodies by quadrant
    for (int64_t i = pstart; i < pend; ++i) {
      int quadrant = get_quadrant(particles[i].coords,
                                  cell->center);
      // out-of-place copy of the particles according to quadrant
      for (int64_t k = 0; k < ndim; ++k) {
        buffer[counter[quadrant]].coords[k] = bodies[i].coords[k];
      }

      // buffer[counter[quadrant]].print();
      // std::cout << " " << quadrant << std::endl;
      counter[quadrant]++;      // increment counters for bodies in each quadrant.
    }

    std::cout << "sizes: ";
    for (int i = 0; i < sizes.size(); ++i) {
      std::cout << sizes[i] << " ";
    }
    std::cout << std::endl;


    std::cout << "offsets: ";
    for (int i = 0; i < offsets.size(); ++i) {
      std::cout << offsets[i] << " ";
    }
    std::cout << std::endl;

    // loop over children and recurse
    for (int64_t d = 0; d < quadrants; ++d) {
      std::vector<double> cell_center(ndim, 0),
        Xmin(ndim, std::numeric_limits<double>::max()),
        Xmax(ndim, std::numeric_limits<double>::min());

      for (int64_t i = offsets[d]; i < offsets[d+1]; ++i) {
        for (int64_t k = 0; k < ndim; ++k) {
          if (Xmax[k] < buffer[i].coords[k]) {
            Xmax[k] = buffer[i].coords[k];
          }
          if (Xmin[k] > buffer[i].coords[k]) {
            Xmin[k] = buffer[i].coords[k];
          }
        }
      }

      double radius = 0;
      for (int64_t k = 0; k < ndim; ++k) {
        cell_center[k] = (Xmax[k] + Xmin[k]) / 2;
        radius += pow(Xmax[k] - Xmin[k], 2);
      }
      radius = sqrt(radius);

      Cell child(cell_center, offsets[d], offsets[d+1], radius);
      cell->cells.push_back(child);

      split_cell(&child, offsets[d], offsets[d+1], max_nleaf, buffer, bodies, !direction);
    }
  }

  void
  Domain::build_tree(const int64_t max_nleaf) {
    // find the min index in each dimension
    std::vector<double> Xmin(ndim, std::numeric_limits<double>::max()),
      Xmax(ndim, std::numeric_limits<double>::min()),
      domain_center(ndim);
    double radius = 0;

    // copy particles into a buffer
    std::vector<Hatrix::Particle> buffer = particles;

    for (int64_t i = 0; i < particles.size(); ++i) {
      for (int64_t k = 0; k < ndim; ++k) {
        if (Xmax[k] < particles[i].coords[k]) {
          Xmax[k] = particles[i].coords[k];
        }
        if (Xmin[k] > particles[i].coords[k]) {
          Xmin[k] = particles[i].coords[k];
        }
      }
    }

    // set the center point
    for (int64_t k = 0; k < ndim; ++k) {
      domain_center[k] = (double)(Xmax[k] + Xmin[k]) / 2.0;
      radius += pow(Xmax[k] - Xmin[k], 2);
    }
    radius = sqrt(radius);

    std::cout << Xmin[0] << " " << Xmin[1] << std::endl;
    std::cout << Xmax[0] << " " <<  Xmax[1]  << std::endl;

    // build the largest node of the tree.
    tree = new Cell(domain_center, 0, N, radius);
    split_cell(tree, 0, N, max_nleaf, particles, buffer, false);
  }

  Cell::Cell(std::vector<double> _center, int64_t pstart,
             int64_t pend, double _radius) :
    center(_center), start_index(pstart), end_index(pend), radius(_radius) {}
}
