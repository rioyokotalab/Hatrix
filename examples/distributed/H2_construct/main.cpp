#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <random>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "mpi.h"

extern "C" {
#include "elses.h"
}

using namespace Hatrix;

static const int BEGIN_PROW = 0, BEGIN_PCOL = 0;

int BLACS_CONTEXT;

int
indxl2g(int indxloc, int nb, int iproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc) % nprocs) * nb + 1;
}

int
translate_rank_comm_world(int num_ranks, int right_comm_rank, MPI_Comm right_comm) {
  std::vector<int> left_rank(num_ranks), right_rank(num_ranks);
  MPI_Group left_group, right_group;

  MPI_Comm_group(MPI_COMM_WORLD, &left_group);
  MPI_Comm_group(right_comm, &right_group);
  for (int i = 0; i < num_ranks; ++i) {
    left_rank[i] = i;
  }


  MPI_Group_translate_ranks(left_group, num_ranks, left_rank.data(),
                            right_group, right_rank.data());

  if (MPIRANK == 0) {
    std::cout << "ranks: " << right_comm_rank <<  std::endl;
    for (int i = 0; i < num_ranks; ++i) {
      std::cout << right_rank[i] << " " << std::endl;
    }
  }

  return right_rank[right_comm_rank];
}

class ScaLAPACK_dist_matrix_t {
public:
  // scalapack storage for matrix descriptor.
  int nrows, ncols, block_nrows, block_ncols, local_stride;
  std::vector<double> data;
  std::vector<int> DESC;
  int local_nrows, local_ncols;

  ScaLAPACK_dist_matrix_t(int nrows, int ncols,
                          int block_nrows, int block_ncols,
                          int begin_prow, int begin_pcol,
                          int BLACS_CONTEXT) :
    nrows(nrows), ncols(ncols), block_nrows(block_nrows), block_ncols(block_ncols)
  {
    local_nrows = numroc_(&nrows, &block_nrows, &MYROW, &begin_prow, &MPIGRID[0]);
    local_ncols = numroc_(&ncols, &block_ncols, &MYCOL, &begin_pcol, &MPIGRID[1]);
    local_stride = local_nrows;

    int INFO;
    DESC.resize(9);
    descinit_(DESC.data(), &nrows, &ncols, &block_nrows, &block_ncols,
              &begin_prow, &begin_pcol, &BLACS_CONTEXT, &local_nrows, &INFO);

    try {
      data.resize((size_t)local_nrows * (size_t)local_ncols, 0);
    }
    catch (std::bad_alloc & exception) {
      std::cerr << "tried to allocate memory of size:  "
                << (size_t)local_nrows * (size_t)local_ncols
                << " " << exception.what() << std::endl;
    }

  }

  int glob_row(int local_row) {
    return indxl2g(local_row + 1, block_nrows, MYROW, MPIGRID[0]) - 1;
  }

  int glob_col(int local_col) {
    return indxl2g(local_col + 1, block_ncols, MYCOL, MPIGRID[1]) - 1;
  }

  void set_local(size_t local_row, size_t local_col, double value) {
    data[local_row + local_col * (size_t)local_nrows] = value;
  }
};

void dist_matvec(ScaLAPACK_dist_matrix_t& A, int A_row_offset, int A_col_offset, double alpha,
                 ScaLAPACK_dist_matrix_t& X, int X_row_offset, int X_col_offset,
                 double beta,
                 ScaLAPACK_dist_matrix_t& B, int B_row_offset, int B_col_offset) {
  const char TRANSA = 'N';
  int INCX = 1, INCB = 1;
  pdgemv_(&TRANSA, &A.nrows, &A.ncols, &alpha,
          A.data.data(), &A_row_offset, &A_col_offset, A.DESC.data(),
          X.data.data(), &X_row_offset, &X_col_offset, X.DESC.data(),
          &INCX,
          &beta,
          B.data.data(), &B_row_offset, &B_col_offset, B.DESC.data(),
          &INCB);
}

// i, j, level -> block numbers.
Matrix
generate_p2p_interactions(int64_t i, int64_t j, int64_t level, const Args& opts,
                          const SymmetricSharedBasisMatrix& A) {
  int64_t block_size = opts.N / A.num_blocks[level];
  Matrix dense(block_size, block_size);

#pragma omp parallel for collapse(2)
  for (int64_t local_i = 0; local_i < block_size; ++local_i) {
    for (int64_t local_j = 0; local_j < block_size; ++local_j) {
      long int global_i = i * block_size + local_i;
      long int global_j = j * block_size + local_j;
      double value;
      get_elses_matrix_value(&global_i, &global_j, &value);

      dense(local_i, local_j) = value;
    }
  }

  return dense;
}

void generate_leaf_nodes(SymmetricSharedBasisMatrix& A, const Args& opts) {
  int64_t nblocks = A.num_blocks[A.max_level];
  int64_t block_size = opts.N / nblocks;

  // Generate dense blocks and store them in the appropriate structure.
  for (int64_t i = 0; i < nblocks; ++i) {
    if (mpi_rank(i) == MPIRANK) {
      Matrix Aij = generate_p2p_interactions(i, i, A.max_level, opts, A);
      A.D.insert(i, i, A.max_level, std::move(Aij));
    }
  }

  double ALPHA = 1.0, BETA = 1.0;

  // Accumulate admissible blocks from the large dist matrix.
  for (int64_t i = MYROW; i < nblocks; i += MPIGRID[0]) { // row cylic distribution.
    Matrix AY(opts.nleaf, opts.nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {

      if (i != j) {
#pragma omp parallel for collapse(2)
        for (int64_t local_i = 0; local_i < block_size; ++local_i) {
          for (int64_t local_j = 0; local_j < block_size; ++local_j) {
            long int global_i = i * block_size + local_i + 1;
            long int global_j = j * block_size + local_j + 1;
            double value;
            get_elses_matrix_value(&global_i, &global_j, &value);

            AY(local_i, local_j) += value;
          }
        }
      }
    }

    Matrix Ui, Si, _V; double error;
    std::tie(Ui, Si, _V, error) = truncated_svd(AY, opts.max_rank);

    A.U.insert(i, A.max_level, std::move(Ui));
    A.US.insert(i, A.max_level, std::move(Si));
  }

  // Calculate the skeleton blocks.
  for (int i = 0; i < nblocks; ++i) {
    std::vector<double> Utemp_buffer(opts.nleaf * opts.max_rank * MPISIZE);

    int p_blocks = nblocks / MPISIZE;

    for (int p_block = 0; p_block < p_blocks; ++p_block) {
      int j = MPISIZE * p_block + MYROW;

      // send, recv
      MPI_Gather(&A.U(j, A.max_level),
                 A.U(j, A.max_level).numel(),
                 MPI_DOUBLE,
                 Utemp_buffer.data(),
                 opts.nleaf * opts.max_rank,
                 MPI_DOUBLE,
                 mpi_rank(i),
                 MPI_COMM_WORLD);

      if (mpi_rank(i) == MPIRANK) {
        for (int j = 0; j < MPISIZE; ++j) {
          int real_j = p_block * MPISIZE + j;
          if (i != j) {
            Matrix temp = matmul(A.U(i, A.max_level),
                                 generate_p2p_interactions(i, real_j, A.max_level, opts, A), true, false);

            Matrix Uj(opts.nleaf, opts.max_rank);
            array_copy(&Utemp_buffer[(opts.nleaf * opts.max_rank) * j], &Uj, Uj.numel());

            auto S_block = matmul(temp, Uj);

            A.S.insert(i, real_j, A.max_level, std::move(S_block));
          }
        }
      }
    }

    int leftover_nprocs = nblocks - p_blocks * MPISIZE;

    // Leftover block group communication. Second condition is needed to limit
    // the root process calculation to within the leftover processes?
    if (p_blocks * MPISIZE < nblocks && mpi_rank(i) < leftover_nprocs) {
      int j = MPISIZE * p_blocks + MYROW;

      MPI_Comm MPI_ROW_I_PROCESSES;
      int color = j < nblocks ? 1 : 0;
      int key = MYROW;

      MPI_Comm_split(MPI_COMM_WORLD,
                     color, key,
                     &MPI_ROW_I_PROCESSES);

      if (j < nblocks && mpi_rank(i) < leftover_nprocs) {
        int root_proc = translate_rank_comm_world(leftover_nprocs, mpi_rank(i), MPI_ROW_I_PROCESSES);
        MPI_Gather(&A.U(j, A.max_level),
                   A.U(j, A.max_level).numel(),
                   MPI_DOUBLE,
                   Utemp_buffer.data(),
                   opts.nleaf * opts.max_rank,
                   MPI_DOUBLE,
                   root_proc,
                   MPI_ROW_I_PROCESSES);

        if (mpi_rank(i) == MPIRANK) {
          for (int j = 0; j < leftover_nprocs; ++j) {
            int real_j = p_blocks * MPISIZE + j;
            if (i != real_j) {
              Matrix temp = matmul(A.U(i, A.max_level),
                                   generate_p2p_interactions(i, real_j, A.max_level, opts, A), true, false);

              Matrix Uj(opts.nleaf, opts.max_rank);
              array_copy(&Utemp_buffer[(opts.nleaf * opts.max_rank) * j], &Uj, Uj.numel());

              auto S_block = matmul(temp, Uj);

              A.S.insert(i, real_j, A.max_level, std::move(S_block));
            }
          }
        }
      }
    }
  }
}

static RowLevelMap
generate_transfer_matrices(SymmetricSharedBasisMatrix& A, const Args& opts, const RowLevelMap& Uchild) {
  RowLevelMap Ubig_parent;

  return Ubig_parent;
}

void construct_H2_matrix(SymmetricSharedBasisMatrix& A, const Args& opts) {
  generate_leaf_nodes(A, opts);
  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    Uchild = generate_transfer_matrices(A, opts, Uchild);
  }
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  int N = opts.N;

  assert(opts.N % opts.nleaf == 0);

  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  MPIGRID[0] = MPISIZE; MPIGRID[1] = 1;

  Cblacs_get(-1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit(&BLACS_CONTEXT, "Row", MPIGRID[0], MPIGRID[1]);
  Cblacs_pcoord(BLACS_CONTEXT, MPIRANK, &MYROW, &MYCOL);

  std::mt19937 gen(MPIRANK);
  std::uniform_real_distribution<double> dist(0, 1);

  // Init domain decomposition for H2 matrix using dual tree traversal.
  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();
    domain.build_tree(opts.nleaf);
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
    domain.build_tree(opts.nleaf);
  }
  else if (opts.kind_of_geometry == COL_FILE) {
    domain.read_col_file_3d(opts.geometry_file);
    domain.build_tree(opts.nleaf);
  }
  else if (opts.kind_of_geometry == ELSES_C60_GEOMETRY) {
    const int64_t num_electrons_per_atom = 4;
    const int64_t num_atoms_per_molecule = 60;
    init_elses_state();
    domain.read_xyz_chemical_file(opts.geometry_file, num_electrons_per_atom);
    domain.build_elses_tree(num_electrons_per_atom * num_atoms_per_molecule);
  }

  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  if (!MPIRANK)
    std::cout << "Domain setup time: " << domain_time << "ms"
              << " leaf: " << opts.nleaf
              << " ndim: " << opts.ndim
              << std::endl;

  auto start_construct = std::chrono::system_clock::now();

  int64_t construct_max_rank;
  SymmetricSharedBasisMatrix A;

  // Making BLR for now.
  A.max_level = log2(opts.N/opts.nleaf);
  A.min_level = log2(opts.N/opts.nleaf);
  A.num_blocks.resize(A.max_level+1);
  A.num_blocks[A.max_level] = opts.N/opts.nleaf;

  // if (opts.admis_kind == GEOMETRY) {
  //   init_geometry_admis(A, domain, opts); // init admissiblity conditions with DTT
  // }
  // else if (opts.admis_kind == DIAGONAL) {
  //   init_diagonal_admis(A, domain, opts); // init admissiblity conditions with diagonal condition.
  // }

  construct_H2_matrix(A, opts);

//   ScaLAPACK_dist_matrix_t VECTOR_B(N, 1, SCALAPACK_BLOCK_SIZE, 1, BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT),
//     VECTOR_X(N, 1, SCALAPACK_BLOCK_SIZE, 1, BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT);

//   // scope the construction so the memory is deleted after construction.
//   {
//     ScaLAPACK_dist_matrix_t DENSE(N, N, SCALAPACK_BLOCK_SIZE, SCALAPACK_BLOCK_SIZE,
//                                   BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT);

// #pragma omp parallel for collapse(2)
//     for (size_t i = 0; i < DENSE.local_nrows; ++i) {
//       for (size_t j = 0; j < DENSE.local_ncols; ++j) {
//         long int g_row = indxl2g(i + 1, SCALAPACK_BLOCK_SIZE, MYROW, MPIGRID[0]) + 1;
//         long int g_col = indxl2g(j + 1, SCALAPACK_BLOCK_SIZE, MYCOL, MPIGRID[1]) + 1;
//         double val;
//         get_elses_matrix_value(&g_row, &g_col, &val);
//         DENSE.set_local(i, j, val);
//       }
//     }

// #pragma omp parallel for
//     for (int i = 0; i < VECTOR_X.local_nrows; ++i) {
//       VECTOR_X.set_local(i, 0, dist(gen));
//       VECTOR_B.set_local(i, 0, 0);
//     }

//     dist_matvec(DENSE, 1, 1, 1.0,
//                 VECTOR_X, 1, 1,
//                 0.0,
//                 VECTOR_B, 1, 1);

//   }

  Cblacs_gridexit(BLACS_CONTEXT);
  Cblacs_exit(1);
  MPI_Finalize();

  if (!MPIRANK) {
    std::cout << "Everything finished.\n";
  }

  return 0;
}
