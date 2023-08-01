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

static const int SCALAPACK_BLOCK_SIZE = 60;
static const int BEGIN_PROW = 0, BEGIN_PCOL = 0;

int BLACS_CONTEXT;

int
indxl2g(int indxloc, int nb, int iproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc) % nprocs) * nb + 1;
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

// sub(C) = beta * sub(C) + alpha * sub(A)
void dist_add(ScaLAPACK_dist_matrix_t& A, int A_nrows, int A_ncols,
              int A_row_offset, int A_col_offset,
              const double alpha, const double beta,
              ScaLAPACK_dist_matrix_t& C, int C_row_offset, int C_col_offset) {
  const char TRANSA = 'N';
  pdgeadd_(&TRANSA, &A_nrows, &A_ncols,
           &alpha,
           A.data.data(), &A_row_offset, &A_col_offset, A.DESC.data(),
           &beta,
           C.data.data(), &C_row_offset, &C_col_offset, C.DESC.data());
}

void dist_svd_only_u(ScaLAPACK_dist_matrix_t& A, int A_nrows, int A_ncols,
                     int A_row_offset, int A_col_offset,
                     std::vector<double>& S_row,
                     ScaLAPACK_dist_matrix_t& U, int U_row_offset, int U_col_offset) {
  // Verify requirements for SCALAPACK.
  assert(A.block_nrows == A.block_ncols);
  assert((A_row_offset - 1) % A.block_nrows == 0);
  assert((A_col_offset - 1) % A.block_ncols == 0);

  // Verify memory requirements.
  assert(S_row.size() == A_ncols);

  int LWORK = -1;
  int INFO;
  const char JOB_U = 'V';
  const char JOB_Vt = 'N';
  std::vector<double> WORK(1);

  // workspace query.
  pdgesvd_(&JOB_U, &JOB_Vt,
           &A_nrows, &A_ncols,
           A.data.data(), &A_row_offset, &A_col_offset, A.DESC.data(),
           S_row.data(),
           U.data.data(), &U_row_offset, &U_col_offset, U.DESC.data(),
           NULL, NULL, NULL, NULL,
           WORK.data(), &LWORK, &INFO);
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

void generate_leaf_nodes(SymmetricSharedBasisMatrix& A, ScaLAPACK_dist_matrix_t& DENSE,
                         ScaLAPACK_dist_matrix_t& U, const Args& opts) {
  ScaLAPACK_dist_matrix_t AY(opts.N, opts.nleaf, SCALAPACK_BLOCK_SIZE, SCALAPACK_BLOCK_SIZE,
                             BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT);

  int64_t nblocks = A.num_blocks[A.max_level];

  // Generate dense blocks and store them in the appropriate structure.
  for (int64_t i = 0; i < nblocks; ++i) {
    if (mpi_rank(i) == MPIRANK) {
      Matrix Aij = generate_p2p_interactions(i, i, A.max_level, opts, A);
      A.D.insert(i, i, A.max_level, std::move(Aij));
    }
  }

  double ALPHA = 1.0, BETA = 1.0;

  // Accumulate admissible blocks from the large dist matrix.
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (i != j) {
        int global_i = i * opts.nleaf + 1;
        int global_j = j * opts.nleaf + 1;

        dist_add(AY, opts.nleaf, opts.nleaf,
                 global_i, 1,
                 ALPHA, BETA,
                 DENSE, global_i, global_j);
      }
    }
  }

  // Calculate the shared bases.
  for (int64_t block = 0; block < nblocks; ++block) {
    std::vector<double> S_MEM(opts.nleaf);

    int IOFFSET = opts.nleaf * block + 1;
    dist_svd_only_u(AY, opts.nleaf, opts.nleaf,
                    IOFFSET, 1,
                    S_MEM,
                    U,
                    IOFFSET, 1);
  }
}

void construct_H2_matrix(SymmetricSharedBasisMatrix& A, ScaLAPACK_dist_matrix_t& DENSE, const Args& opts) {
  ScaLAPACK_dist_matrix_t U(opts.N, opts.nleaf, SCALAPACK_BLOCK_SIZE, SCALAPACK_BLOCK_SIZE,
                            BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT);

  generate_leaf_nodes(A, DENSE, U, opts);
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  int N = opts.N;

  assert(opts.nleaf % SCALAPACK_BLOCK_SIZE == 0);
  assert(opts.N % opts.nleaf == 0);
  assert(opts.max_rank % SCALAPACK_BLOCK_SIZE == 0);

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
    //    domain.build_elses_tree(num_electrons_per_atom * num_atoms_per_molecule);
  }

  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  if (!MPIRANK)
    std::cout << "Domain setup time: " << domain_time << "ms" << std::endl;

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

  // scope the construction so the memory is deleted after construction.
  {
    ScaLAPACK_dist_matrix_t DENSE(N, N, SCALAPACK_BLOCK_SIZE, SCALAPACK_BLOCK_SIZE,
                                  0, 0, BLACS_CONTEXT);
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < DENSE.local_nrows; ++i) {
      for (size_t j = 0; j < DENSE.local_ncols; ++j) {
        long int g_row = indxl2g(i + 1, SCALAPACK_BLOCK_SIZE, MYROW, MPIGRID[0]) + 1;
        long int g_col = indxl2g(j + 1, SCALAPACK_BLOCK_SIZE, MYCOL, MPIGRID[1]) + 1;
        double val;
        get_elses_matrix_value(&g_row, &g_col, &val);
        DENSE.set_local(i, j, val);
      }
    }

    construct_H2_matrix(A, DENSE, opts);

    ScaLAPACK_dist_matrix_t VECTOR_B(N, 1, SCALAPACK_BLOCK_SIZE, 1, 0, 0, BLACS_CONTEXT),
      VECTOR_X(N, 1, SCALAPACK_BLOCK_SIZE, 1, 0, 0, BLACS_CONTEXT);

#pragma omp parallel for
    for (int i = 0; i < VECTOR_X.local_nrows; ++i) {
      VECTOR_X.set_local(i, 0, dist(gen));
      VECTOR_B.set_local(i, 0, 0);
    }

    dist_matvec(DENSE, 1, 1, 1.0,
                VECTOR_X, 1, 1,
                0.0,
                VECTOR_B, 1, 1);

  }

  Cblacs_gridexit(BLACS_CONTEXT);
  Cblacs_exit(1);
  MPI_Finalize();

  if (!MPIRANK) {
    std::cout << "Everything finished.\n";
  }

  return 0;
}
