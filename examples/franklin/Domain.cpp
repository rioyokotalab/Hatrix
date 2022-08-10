#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

namespace Hatrix {
  int64_t
  Domain::cell_size(int64_t level_index, int64_t level) const {
    int64_t pstart, pend;
    search_tree_for_nodes(tree, level_index, level, pstart, pend);

    return pend - pstart;
  }

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
      for (int64_t i = 0; i < N; ++i) {
        double theta = (i * 2.0 * M_PI) / N ;
        double x = radius * cos(theta);
        double y = radius * sin(theta);

        particles.push_back(Hatrix::Particle(x, y, min_val + (double(i) / double(range))));
      }
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

  void Domain::orb_split(Cell& cell,
                         const int64_t pstart,
                         const int64_t pend,
                         const int64_t max_nleaf,
                         const int64_t dim,
                         const int64_t level,
                         uint32_t level_index) {
    std::vector<double> Xmin(ndim, std::numeric_limits<double>::max()),
      Xmax(ndim, std::numeric_limits<double>::min()),
      cell_center(ndim, 0);

    // calculate the min and max points for this cell
    for (int64_t i = pstart; i < pend; ++i) {
      for (int k = 0; k < ndim; ++k) {
        if (Xmax[k] < particles[i].coords[k]) {
          Xmax[k] = particles[i].coords[k];
        }
        if (Xmin[k] > particles[i].coords[k]) {
          Xmin[k] = particles[i].coords[k];
        }
      }
    }

    // set the center point and radius
    cell.radius = 0;
    for (int64_t k = 0; k < ndim; ++k) {
      cell_center[k] = (double)(Xmax[k] + Xmin[k]) / 2.0;
      cell.radius += pow(Xmax[k] - Xmin[k], 2);
    }
    cell.radius = sqrt(cell.radius) / 2;
    cell.center = cell_center;
    cell.start_index = pstart;
    cell.end_index = pend;
    cell.level = level;
    cell.level_index = level_index;

    // we have hit the leaf level. return without sorting.
    if (pend - pstart <= max_nleaf) {
      return;
    }

    // permute the points along dim using the median as the splitting point.
    std::sort(particles.begin() + pstart, // TODO: is there a way to not sort? try k-th largest element.
              particles.begin() + pend,
              [&](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[dim] < rhs.coords[dim];
              });
    int64_t mid = ceil(double(pstart + pend) / 2.0);
    cell.cells.resize(2);
    orb_split(cell.cells[0], pstart, mid, max_nleaf, (dim+1) % ndim, level+1, level_index << 1);
    orb_split(cell.cells[1], mid, pend, max_nleaf, (dim+1) % ndim, level+1, (level_index << 1) + 1);
  }

  void
  Domain::build_tree(const int64_t max_nleaf) {
    orb_split(tree, 0, N, max_nleaf, 0, 0, 0);
  }

  Cell::Cell() : start_index(-1), end_index(-1), radius(-1), level(-1) {}

  Cell::Cell(std::vector<double> _center, int64_t pstart,
             int64_t pend, double _radius) :
    center(_center), start_index(pstart), end_index(pend), radius(_radius) {}

  void
  Cell::print() const {
    std::cout << "level: " << level << std::endl;
    std::cout << "start: " << start_index << " stop: " << end_index
              << " radius: " << radius
              << " index: " << level_index
              << std::endl;
    for (int i = 0; i < cells.size(); ++i) {
      cells[i].print();
    }
  }

  int64_t
  Cell::height() const {
    if (cells.size() > 0) {
      int64_t left = cells[0].height() + 1;
      int64_t right = cells[1].height() + 1;

      return (left > right) ? left : right;
    }
    else {
      return 1;
    }
  }
}
