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

#include "Hatrix/Hatrix.hpp"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.

namespace Hatrix {
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

    RowMap<Hatrix::Matrix> U;
    ColMap<Hatrix::Matrix> V;
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

    double low_rank_block_ratio() {
      double total = 0, low_rank = 0;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            low_rank += 1;
          }
          total += 1;
        }
      }

      return low_rank / total;
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

  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(domain, N, nleaf, rank, ndim, admis);
  A.print_structure();
  double construct_error = A.construction_error(domain);

  Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(domain.particles, N, N, 1);

  std::cout << "N: " << N << " rank: " << rank << " nleaf: " << nleaf << " admis: " <<  admis
            << " construct error: " << construct_error << " LR ratio: " << A.low_rank_block_ratio() <<  "\n";

  return 0;
}
