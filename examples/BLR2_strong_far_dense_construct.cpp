#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <string>

#include "Hatrix/Hatrix.h"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.

namespace Hatrix {
  class Particle {
  public:
    double value;
    std::vector<double> coords;

    Particle(double x, double _value) : value(_value)  {
      coords.push_back(x);
    }

    Particle(double x, double y, double _value) : value(_value) {
      coords.push_back(x);
      coords.push_back(y);
    }

    Particle(double x, double y, double z, double _value) : value(_value)  {
      coords.push_back(x);
      coords.push_back(y);
      coords.push_back(z);
    }
  };

  class Box {
  public:
    double diameter;
    int64_t ndim, num_particles, start_index, stop_index;
    // Store the center, start and end co-ordinates of this box. Each number
    // in corresponds to the x, y, and z co-oridinate.
    std::vector<double> center;

    std::string morton_index;

    Box() {}

    Box(double _diameter, double center_x, int64_t _start_index, int64_t _stop_index,
        std::string _morton_index, int64_t _num_particles) :
      diameter(_diameter),
      ndim(1),
      num_particles(_num_particles),
      start_index(_start_index),
      stop_index(_stop_index),
      morton_index(_morton_index) {
      center.push_back(center_x);
    }

    Box(double diameter, double center_x, double center_y, double start_index,
        double stop_index, std::string _morton_index, int64_t num_particles) :
      diameter(diameter),
      ndim(2),
      num_particles(num_particles),
      start_index(start_index),
      stop_index(stop_index),
      morton_index(_morton_index) {
      center.push_back(center_x);
      center.push_back(center_y);
    }

    double distance_from(const Box& b) const {
      double dist = 0;
      for (int k = 0; k < ndim; ++k) {
        dist += pow(b.center[k] - center[k], 2);
      }
      return std::sqrt(dist);
    }
  };

  class Domain {
  public:
    std::vector<Hatrix::Particle> particles;
    std::vector<Hatrix::Box> boxes;
    int64_t N, ndim;

  private:
    // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
    void orthogonal_recursive_bisection_1dim(int64_t start, int64_t end, std::string morton_index, int64_t nleaf) {
      // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
      std::sort(particles.begin()+start, particles.begin()+end, [](const Particle& lhs, const Particle& rhs) {
        return lhs.coords[0] <= rhs.coords[0];
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

    void orthogonal_recursive_bisection_2dim_yaxis(int64_t start, int64_t end, std::string morton_index, int64_t nleaf) {
      // Sort along the Y co-ordinates for the points that exist between [start, middle).
      std::sort(particles.begin() + start, particles.begin() + end, [](const Particle& lhs, const Particle& rhs) {
        return lhs.coords[1] < rhs.coords[1];
      });

      int64_t num_points = end - start;
      if (num_points <= nleaf) {
        int64_t start_index = start;
        int64_t end_index = end - 1;

        double diameter = 0;
        for (int k = 0; k < ndim; ++k) {
          diameter += pow(particles[start_index].coords[k] - particles[end_index].coords[k], 2);
        }
        diameter = std::sqrt(diameter);

        double center_x = (particles[start_index].coords[0] + particles[end_index].coords[0]) / 2;
        double center_y = (particles[start_index].coords[1] + particles[end_index].coords[1]) / 2;

        boxes.push_back(Box(diameter, center_x, center_y, start_index, end_index, morton_index, num_points));
      }
      else {
        int64_t middle = (start + end) / 2;
        orthogonal_recursive_bisection_2dim_xaxis(start, middle, morton_index + "0", nleaf);
        orthogonal_recursive_bisection_2dim_xaxis(middle, end, morton_index + "1", nleaf);
      }
    }

    void orthogonal_recursive_bisection_2dim_xaxis(int64_t start, int64_t end, std::string morton_index, int64_t nleaf) {
      std::sort(particles.begin() + start, particles.begin() + end, [](const Particle& lhs, const Particle& rhs) {
        return lhs.coords[0] <= rhs.coords[0];
      });

      int64_t num_points = end - start;
      if (num_points <= nleaf) {
        orthogonal_recursive_bisection_2dim_yaxis(start, end, morton_index, nleaf);
      }
      else {
        int64_t middle = (start + end) / 2;
        orthogonal_recursive_bisection_2dim_yaxis(start, middle, morton_index + "0", nleaf);
        orthogonal_recursive_bisection_2dim_yaxis(middle, end, morton_index + "1", nleaf);
      }
    }

  public:

    Domain(int64_t N, int64_t ndim) : N(N), ndim(ndim) {}

    void generate_particles(double min_val, double max_val) {
      double range = max_val - min_val;

      if (ndim == 1) {
        for (int64_t i = 0; i < N; ++i) {
          particles.push_back(Hatrix::Particle(i*0.4, min_val + (double(i) / double(range))));
        }
      }
      else if (ndim == 2) {
        // Generate a unit circle with N points on the circumference.
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
        double radius = 1.0;
        for (int64_t i = 0; i < N; ++i) {
          double theta = dis(gen);
          double x = radius * cos(theta);
          double y = radius * sin(theta);

          particles.push_back(Hatrix::Particle(x, y, min_val + (double(i) / double(range))));
        }
      }
      else if (ndim == 3) {
        // Generate a unit sphere geometry with N points on the surface.
        // http://www.cpp.re/forum/windows/262648/
        // https://neil-strickland.staff.shef.ac.uk/courses/MAS243/lectures/handout10.pdf
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
        double radius = 1.0;
        for (int64_t i = 0; i < N; ++i) {
          double phi = dis(gen);
          double theta = dis(gen);

          double x = radius * sin(phi) * cos(theta);
          double y = radius * sin(phi) * sin(theta);
          double z = radius * cos(phi);

          particles.push_back(Hatrix::Particle(x, y, z, min_val + (double(i) / double(range))));
        }
      }
    }

    void divide_domain_and_create_particle_boxes(int64_t nleaf) {
      if (ndim == 1) {
        orthogonal_recursive_bisection_1dim(0, N, std::string(""), nleaf);
      }
      else if (ndim == 2) {
        orthogonal_recursive_bisection_2dim_xaxis(0, N, std::string(""), nleaf);
      }
      else if (ndim == 3) {

      }
    }
  };


  // Generate a full dense laplacian matrix assuming unit charges.
  Matrix generate_laplacend_matrix(const std::vector<Hatrix::Particle>& particles,
                                   int64_t nrows, int64_t ncols, int64_t ndim) {
    Matrix out(nrows, ncols);

    for (int64_t i = 0; i < nrows; ++i) {
      for (int64_t j = 0; j < ncols; ++j) {
        double rij = 0;
        for (int64_t k = 0; k < ndim; ++k) {
          rij += pow(particles[i].coords[k] - particles[j].coords[k], 2);
        }
        out(i, j) = 1 / (std::sqrt(rij) + 1e-3);
      }
    }
    return out;
  }

  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol, int64_t ndim) {
    Matrix out(domain.boxes[irow].num_particles, domain.boxes[icol].num_particles);

    for (int64_t i = 0; i < domain.boxes[irow].num_particles; ++i) {
      for (int64_t j = 0; j < domain.boxes[icol].num_particles; ++j) {
        double rij = 0;
        int64_t source = domain.boxes[irow].start_index;
        int64_t target = domain.boxes[icol].start_index;

        for (int64_t k = 0; k < ndim; ++k) {
          rij += pow(domain.particles[source+i].coords[k] - domain.particles[target+j].coords[k], 2);
        }


        out(i, j) = 1 / (std::sqrt(rij) + 1e-3);
      }
    }

    return out;
  }

  class BLR2 {
  private:
    // Store the dense blocks in a multimap for faster iteration without hash lookups.
    std::multimap<int64_t, Matrix> D;
    // Map of inadmissible indices.
    RowColMap<bool> is_admissible;
    // Vector of vector for storing the actual indices of all the inadmissible blocks in a given row.
    std::vector<std::vector<int64_t> > inadmissible_row_indices, admissible_row_indices;
    std::vector<std::vector<int64_t> > inadmissible_col_indices, admissible_col_indices;

    RowMap U;
    ColMap V;
    RowColMap<Matrix> S;

    int64_t N, nleaf, rank, ndim, nblocks;
    double admis;

  public:
    BLR2(const Domain& domain, int64_t N,
         int64_t nleaf, int64_t rank, int64_t ndim, double admis) :
      N(N), nleaf(nleaf), rank(rank), ndim(ndim), admis(admis) {
      nblocks = domain.boxes.size();

      inadmissible_row_indices.resize(nblocks);
      admissible_row_indices.resize(nblocks);

      inadmissible_col_indices.resize(nblocks);
      admissible_col_indices.resize(nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::min(domain.boxes[i].diameter, domain.boxes[j].diameter) <=
                               admis * domain.boxes[i].distance_from(domain.boxes[j]));

          if (!is_admissible(i, j)) {
            D.insert({i, generate_p2p_interactions(domain, i, j, ndim)});
          }
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            admissible_row_indices[i].push_back(j);
          }
          else {
            inadmissible_row_indices[i].push_back(j);
          }
        }
      }

      for (int j = 0; j < nblocks; ++j) {
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible(j, i)) {
            admissible_col_indices[j].push_back(i);
          }
          else {
            inadmissible_col_indices[j].push_back(i);
          }
        }
      }

      int64_t oversampling = 5;
      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(generate_random_matrix(domain.boxes[i].num_particles, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Hatrix::Matrix AY(domain.boxes[i].num_particles, rank + oversampling);
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = generate_p2p_interactions(domain, i, jcol, ndim);
          Hatrix::matmul(dense, Y[jcol], AY);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, std::move(Utemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, domain.boxes[j].num_particles);

        for (long unsigned int i = 0; i < admissible_col_indices[j].size(); ++i) {
          int64_t irow = admissible_col_indices[j][i];
          Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, irow, j, ndim);
          Hatrix::matmul(Y[irow], dense, YtA, true);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, std::move(transpose(Vtemp)));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, i, jcol, ndim);
          S.insert(i, jcol, Hatrix::matmul(Hatrix::matmul(U[i], dense, true), V[jcol]));
        }
      }
    }

    double construction_error(const Domain& domain) {
      // Check dense blocks
      double error = 0; double dense_norm = 0;

      for (int i = 0; i < nblocks; ++i) {
        std::pair<std::multimap<int64_t, Matrix>::iterator,
                  std::multimap<int64_t, Matrix>::iterator> row_dense_blocks = D.equal_range(i);

        int j = 0;
        for (std::multimap<int64_t, Matrix>::iterator it = row_dense_blocks.first; it != row_dense_blocks.second; ++it) {
          int64_t jcol = inadmissible_row_indices[i][j];
          auto dense = Hatrix::generate_p2p_interactions(domain, i, jcol, ndim);
          dense_norm += pow(norm(dense), 2);
          error += pow(norm(it->second -  dense), 2);
          j++;
        }
      }

      for (unsigned i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          auto dense = generate_p2p_interactions(domain, i, jcol, ndim);
          Matrix& Ubig = U(i);
          Matrix& Vbig = V(jcol);
          Matrix expected = matmul(matmul(Ubig, S(i, jcol)), Vbig, false, true);
          Matrix actual = generate_p2p_interactions(domain, i, jcol, ndim);
          error += pow(norm(expected - actual), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }

      return std::sqrt(error / dense_norm);
    }

    void print_structure() {
      std::cout << "BLR " << nblocks << " x " << nblocks << std::endl;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          std::cout << " | " << is_admissible(i, j);
        }
        std::cout << " | \n";
      }
    }
  };
}


int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);

  Hatrix::Context::init();
  Hatrix::Domain domain(N, ndim);
  domain.generate_particles(0.0, 1.0 * N);
  domain.divide_domain_and_create_particle_boxes(nleaf);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(domain, N, nleaf, rank, ndim, admis);
  A.print_structure();
  double construct_error = A.construction_error(domain);

  Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(domain.particles, N, N, 1);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nleaf: " << nleaf << " admis: " <<  admis
            << " construct error: " << construct_error << "\n";
}
