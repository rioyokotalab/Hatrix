#include <exception>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"
#include "matrix_construction.hpp"
#include "MPIWrapper.hpp"
#include "MPISymmSharedBasisMatrix.hpp"

std::mt19937 random_generator;
std::uniform_real_distribution<double> uniform_distribution(0, 1.0);

static const int64_t P = 100;

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

static Hatrix::Matrix
generate_column_bases(int64_t i, int64_t block_size, int64_t level,
                      const Hatrix::RowMap<Hatrix::Matrix>& rand, MPISymmSharedBasisMatrix& A,
                      const Hatrix::Args& opts) {

}

static void
generate_random_blocks(const Hatrix::Domain& domain,
                         MPISymmSharedBasisMatrix& A,
                         const Hatrix::RowMap<Hatrix::Matrix>& rand,
                         Hatrix::RowMap<Hatrix::Matrix>& product,
                         const Hatrix::Args& opts) {
  int64_t nblocks = pow(2, A.max_level);
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t block_size = domain.boxes[block].num_particles;
    Hatrix::Matrix random_block(block_size, P);

    if (A.rank_1d(block) == mpi_world.MPIRANK) {
      random_block = rand(block);
    }

    MPI_Bcast(&random_block, block_size * P, MPI_DOUBLE, A.rank_1d(block), MPI_COMM_WORLD);

    for (int64_t i = mpi_world.MPIRANK; i < nblocks; i += mpi_world.MPISIZE) {
      Hatrix::Matrix Ai_block = generate_p2p_interactions(domain, i, block, opts.kernel);
      matmul(Ai_block, random_block, product(i), false, false, 1.0, 1.0);
    }
  }
}

static void
generate_leaf_nodes(const Hatrix::Domain& domain, MPISymmSharedBasisMatrix& A,
                    const Hatrix::RowMap<Hatrix::Matrix>& rand, Hatrix::RowMap<Hatrix::Matrix>& product,
                    const Hatrix::Args& opts) {
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

  // A.rank_map.insert(A.max_level, std::vector<int64_t>(nblocks));

  generate_random_blocks(domain, A, rand, product, opts);

  // generate the leaf basis.
  for (int64_t i = mpi_world.MPIRANK; i < nblocks; i += mpi_world.MPISIZE) {
    Hatrix::Matrix Ui;
    std::vector<int64_t> pivots;
    int64_t rank;

    if (opts.accuracy == -1) {
      rank = opts.max_rank;
      std::tie(Ui, pivots) = pivoted_qr(product(i), rank);
    }
    else {
      std::tie(Ui, pivots, rank) = error_pivoted_qr(product(i), opts.accuracy);
    }
    A.U.insert(i,
               A.max_level,
               std::move(Ui));
    // A.rank_map(A.max_level)[i] = rank;
  }

  // generate the S blocks
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        if (A.rank_1d(i) == mpi_world.MPIRANK) {
          Hatrix::Matrix Aij = generate_p2p_interactions(domain, i, j, opts.kernel);
          // A.S.insert(i, j, A.max_level,
          //            )
        }
      }
    }
  }
}

void construct_h2_miro(MPISymmSharedBasisMatrix& A, const Hatrix::Domain& domain,
                       const Hatrix::Args& opts) {
  // init random matrix
  Hatrix::RowMap<Hatrix::Matrix> rand, product;
  int64_t nblocks = pow(2, A.max_level);

  for (int64_t block = 0; block < nblocks; ++block) {
    if (A.rank_1d(block) == mpi_world.MPIRANK) {
      Hatrix::Matrix random_block(domain.boxes[block].num_particles, P);

      for (int64_t i = 0; i < opts.nleaf; ++i) {
        for (int64_t j = 0; j < P; ++j) {
          random_block(i, j) = uniform_distribution(random_generator);
        }
      }
      rand.insert(block, std::move(random_block));
      product.insert(block, Hatrix::Matrix(domain.boxes[block].num_particles,
                                           P));
    }
  }
  generate_leaf_nodes(domain, A, rand, product, opts);
}
