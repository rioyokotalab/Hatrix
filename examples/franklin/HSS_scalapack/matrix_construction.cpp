#include <exception>
#include <random>
#include <cassert>

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

static int64_t diagonal_admis_init(MPISymmSharedBasisMatrix& A,
                                   const Hatrix::Args& opts, int64_t level) {
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
      if (A.is_admissible.exists(i, block, A.max_level) &&
          !A.is_admissible(i, block, A.max_level)) { continue; }
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

  std::vector<int64_t> leaf_ranks(nblocks);

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
      std::tie(Ui, pivots, rank) = error_pivoted_qr(product(i), opts.accuracy, opts.max_rank);
    }
    A.U.insert(i,
               A.max_level,
               std::move(Ui));

    leaf_ranks[i] = rank;
  }

  // TODO: this is really clumsy. Find a way to all gather a distributed strided array.
  for (int64_t i = 0; i < nblocks; ++i) {
    MPI_Bcast(&leaf_ranks[i], 1, MPI_INT64_T, i % mpi_world.MPISIZE, MPI_COMM_WORLD);
  }

  // generate the S blocks
  for (int64_t i = 0; i < nblocks; ++i) {
    int64_t block_size = domain.boxes[i].num_particles;
    for (int64_t j = 0; j < i; ++j) {
      MPI_Comm MPI_COMM_ROW;
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        MPI_Comm_split(MPI_COMM_WORLD, i, j, &MPI_COMM_ROW);
      }

      Hatrix::Matrix Ui(block_size, leaf_ranks[i]);
      if (A.rank_1d(i) == mpi_world.MPIRANK) {
        Ui = A.U(i, A.max_level);
      }
      MPI_Bcast(&Ui, block_size * leaf_ranks[i], MPI_DOUBLE,
                A.rank_1d(i), MPI_COMM_WORLD);

    }
  }

  A.rank_map.insert(A.max_level, std::move(leaf_ranks));
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
