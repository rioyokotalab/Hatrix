#include <exception>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"
#include "matrix_construction.hpp"
#include "MPIWrapper.hpp"

#include "slate/slate.hh"

std::mt19937 random_generator;
std::uniform_real_distribution<double> uniform_distribution(0, 1.0);

static void coarsen_blocks(MPISymmSharedBasisMatrix& A, int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j <= i; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (A.is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !A.is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            A.is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      A.is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

static int64_t diagonal_admis_init(MPISymmSharedBasisMatrix& A, const Hatrix::Args& opts, int64_t level) {
  int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
  if (level == 0) { return 0; }
  if (level == A.max_level) {
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j <= i; ++j) {
        A.is_admissible.insert(i, j, level, std::abs(i - j) > opts.admis);
      }
    }
  }
  else {
    coarsen_blocks(A, level);
  }

  diagonal_admis_init(A, opts, level-1);
  return level;
}

void init_diagonal_admis(MPISymmSharedBasisMatrix& A, const Hatrix::Args& opts) {
  A.max_level = int64_t(log2(opts.N / opts.nleaf));
  A.min_level = diagonal_admis_init(A, opts, A.max_level);
  A.is_admissible.insert(0, 0, 0, false);
}

static void
generate_leaf_nodes(const Hatrix::Domain& domain, MPISymmSharedBasisMatrix& A, slate::Matrix<double>& dense,
                    const slate::Matrix<double>& rand, slate::Matrix<double>& product, const Hatrix::Args& opts) {
  int64_t nblocks = pow(2, A.max_level);
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level) && A.rank_1d(i) == mpi_world.MPIRANK) {
        A.D.insert(i, j, A.max_level,
                   generate_p2p_interactions(domain, i, j, opts.kernel));
      }
    }
  }
}

void random_matrix(int64_t nrows, int64_t ncols, double* data, int64_t lda) {
  for (int64_t i = 0; i < nrows; ++i) {
    for (int64_t j = 0; j < ncols; ++j) {
      data[i + j * lda] = uniform_distribution(random_generator);
    }
  }
}

void random_matrix(slate::Matrix<double>& rand) {
  rand.insertLocalTiles();
  for (int64_t i = 0; i < rand.mt(); ++i) {
    for (int64_t j = 0; j < rand.nt(); ++j) {
      if (rand.tileIsLocal(i, j)) {
        slate::Tile<double> tile = rand(i, j);
        random_matrix(tile.mb(), tile.nb(), tile.data(), tile.stride());
      }
    }
  }
}

void construct_h2_mpi_miro(MPISymmSharedBasisMatrix& A, const Hatrix::Domain& domain,
                           const Hatrix::Args& opts) {
  const int64_t P = 100;
  slate::Matrix<double> dense(opts.N, opts.N, opts.nleaf, opts.nleaf, mpi_world.MPISIZE, 1, mpi_world.COMM);
  slate::Matrix<double> rand(opts.N, P, opts.nleaf, P, mpi_world.MPISIZE, 1, mpi_world.COMM);
  slate::Matrix<double> product(opts.N, P, opts.nleaf, P, mpi_world.MPISIZE, 1, mpi_world.COMM);

  random_matrix(rand);

  generate_leaf_nodes(domain, A, dense, rand, product, opts);
}
