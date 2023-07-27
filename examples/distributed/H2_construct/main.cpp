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

using namespace Hatrix;

static const int SCALAPACK_BLOCK_SIZE = 256;

int indxl2g(int indxloc, int nb, int iproc, int nprocs) {
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

    data.resize((size_t)local_nrows * (size_t)local_ncols);
  }

  int glob_row(int local_row) {
    return indxl2g(local_row + 1, block_nrows, MYROW, MPIGRID[0]) - 1;
  }

  int glob_col(int local_col) {
    return indxl2g(local_col + 1, block_ncols, MYCOL, MPIGRID[1]) - 1;
  }

  void set_local(size_t local_row, size_t local_col, double value) {
    data[local_row + local_col * local_nrows] = value;
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


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  int N = opts.N;
  int BLACS_CONTEXT;

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

  // Init domain decomposition for H2 matrix using DTT.
  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  else if (opts.kind_of_geometry == COL_FILE) {
    domain.read_col_file_3d(opts.geometry_file);
  }
  domain.build_tree(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  int64_t construct_max_rank;
  SymmetricSharedBasisMatrix A;

  auto start_construct = std::chrono::system_clock::now();
  if (opts.admis_kind == GEOMETRY) {
    init_geometry_admis(A, domain, opts); // init admissiblity conditions with DTT
  }
  else if (opts.admis_kind == DIAGONAL) {
    init_diagonal_admis(A, domain, opts); // init admissiblity conditions with diagonal condition.
  }

  ScaLAPACK_dist_matrix_t DENSE(N, N, SCALAPACK_BLOCK_SIZE, SCALAPACK_BLOCK_SIZE, 0, 0, BLACS_CONTEXT);
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < DENSE.local_nrows; ++i) {
    for (size_t j = 0; j < DENSE.local_ncols; ++j) {
      double value = opts.kernel(domain.particles[DENSE.glob_row(i)].coords,
                                 domain.particles[DENSE.glob_col(j)].coords);
      DENSE.set_local(i, j, value);
    }
  }

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


  Cblacs_gridexit(BLACS_CONTEXT);
  Cblacs_exit(1);
  MPI_Finalize();

  if (!MPIRANK) {
    std::cout << "Everything finished.\n";
  }

  return 0;
}
