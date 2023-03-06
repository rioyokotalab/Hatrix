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

#include "parsec.h"

#include "globals.hpp"
#include "h2_construction.hpp"
#include "h2_operations.hpp"

#include "omp.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

using namespace Hatrix;

static void
redistribute_vector2scalapack(std::vector<Matrix>& x,
                              std::vector<double>& x_mem,
                              SymmetricSharedBasisMatrix& A,
                              Hatrix::Args& opts) {
  int nblocks = pow(2, A.max_level);

  // send Matrix blocks to procs where scalapack blocks exist.
  for (int i = 0; i < nblocks; ++i) {
    int x_rank = mpi_rank(i);
    int scalapack_rank = (i % MPIGRID[0]) * MPIGRID[1];
    int index = i / MPISIZE;
    MPI_Request req;

    if (MPIRANK == x_rank) {
      MPI_Isend(&x[index], x[index].numel(), MPI_DOUBLE,
                scalapack_rank, i, MPI_COMM_WORLD, &req);
    }
  }

  // receive blocks into scalapack blocks.
  for (int i = 0; i < nblocks; ++i) {
    MPI_Status status;
    int x_rank = mpi_rank(i);
    int scalapack_rank = (i % MPIGRID[0]) * MPIGRID[1];
    int index = (i / MPIGRID[0]) * DENSE_NBROW;

    if (MPIRANK == scalapack_rank) {
      MPI_Recv(&x_mem[index], DENSE_NBROW, MPI_DOUBLE, x_rank,
               i, MPI_COMM_WORLD, &status);
    }
  }
}

// Distribute the scalapack vector A_mem into the vector of Matrix objects A.
static void
redistribute_scalapack2vector(std::vector<Matrix>& x,
                              std::vector<double>& x_mem,
                              SymmetricSharedBasisMatrix& A,
                              Hatrix::Args& opts) {
  int nblocks = pow(2, A.max_level);

  for (int i = 0; i < nblocks; ++i) {
    MPI_Request req;
    int x_rank = mpi_rank(i);
    int scalapack_rank = (i % MPIGRID[0]) * MPIGRID[1];
    int index = (i / MPIGRID[0]) * DENSE_NBROW;

    if (MPIRANK == scalapack_rank) {
      MPI_Isend(&x_mem[index], DENSE_NBROW, MPI_DOUBLE,
                x_rank, i, MPI_COMM_WORLD, &req);
    }
  }

  for (int i = 0; i < nblocks; ++i) {
    MPI_Status status;
    int x_rank = mpi_rank(i);
    int scalapack_rank = (i % MPIGRID[0]) * MPIGRID[1];
    int index = i / MPISIZE;

    if (MPIRANK == x_rank) {
      MPI_Recv(&x[index], DENSE_NBROW, MPI_DOUBLE, scalapack_rank,
               i, MPI_COMM_WORLD, &status);
    }
  }
}

static double
dist_norm2(std::vector<Matrix>& x) {
  double partial_sum_squares = 0, sum_squares = 0;

  for (int i = 0; i < x.size(); ++i) {
    for (int j = 0; j < x[i].numel(); ++j) {
      partial_sum_squares += pow(x[i](j, 0), 2);
    }
  }

  MPI_Allreduce(&partial_sum_squares, &sum_squares, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);

  return sqrt(sum_squares);
}

static std::vector<Matrix>
cholesky_solve(SymmetricSharedBasisMatrix& A,
               Args& opts,
               std::vector<double>& X_mem, std::vector<int>& DESCX) {
  std::vector<Matrix> dense_solution;
  for (int64_t block = MPIRANK; block < pow(2, A.max_level); block += MPISIZE) {
    dense_solution.push_back(Matrix(opts.nleaf, 1));
  }

  const char UPLO = 'L';
  int IA = 1, JA = 1;
  int info;

  pdpotrf_(&UPLO, &N, DENSE_MEM, &IA, &JA, DENSE.data(), &info);

  const char SIDE = 'L';
  const char DIAG = 'N';
  const double ALPHA = 1.0;
  int IB = 1;
  int JB = 1;

  // x_mem = L^(-1) * x_mem
  pdtrsm_(&SIDE, &UPLO, &NOTRANS, &DIAG,
          &N, &ONE, &ALPHA,
          DENSE_MEM, &IA, &JA, DENSE.data(),
          X_mem.data(), &IB, &JB, DESCX.data());

  // x_mem = L.T^(-1) * x_mem
  pdtrsm_(&SIDE, &UPLO, &TRANS, &DIAG,
          &N, &ONE, &ALPHA,
          DENSE_MEM, &IA, &JA, DENSE.data(),
          X_mem.data(), &IB, &JB, DESCX.data());

  redistribute_scalapack2vector(dense_solution, X_mem, A, opts);

  return dense_solution;
}


int main(int argc, char **argv) {
  Hatrix::Context::init();

  int rc;


  std::cout << "init args.\n";
  Args opts(argc, argv);

  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  MPI_Dims_create(MPISIZE, 2, MPIGRID);
  N = opts.N;
  int cores = opts.parsec_cores;
  if (!MPIRANK) {
    std::cout << "MPIGRID g[0] : " << MPIGRID[0]
              << " g[1]: " << MPIGRID[1]
              << " num args: " << opts.num_args
              << std::endl;
  }

  // Init domain decomposition for H2 matrix using DTT.
  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  else if (opts.kind_of_geometry == COL_FILE_3D) {
    domain.read_col_file_3d(opts.geometry_file);
  }
  domain.build_tree(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();


  int64_t construct_max_rank;
  SymmetricSharedBasisMatrix A;

  auto start_construct = std::chrono::system_clock::now();

  Cblacs_get(-1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit(&BLACS_CONTEXT, "Row", MPIGRID[0], MPIGRID[1]);
  Cblacs_pcoord(BLACS_CONTEXT, MPIRANK, &MYROW, &MYCOL);

  DENSE_NBROW = opts.nleaf;
  DENSE_NBCOL = opts.nleaf;
  DENSE_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  DENSE_local_cols = numroc_(&N, &DENSE_NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);

  descinit_(DENSE.data(), &N, &N, &DENSE_NBROW, &DENSE_NBCOL, &ZERO, &ZERO,
            &BLACS_CONTEXT, &DENSE_local_rows, &info);
  DENSE_MEM = new double[int64_t(DENSE_local_rows) * int64_t(DENSE_local_cols)];

  if (!MPIRANK) {
    std::cout << "begin data init.\n";
  }

  // generate the distributed P2P matrix.
#pragma omp parallel for
  for (int64_t i = 0; i < DENSE_local_rows; ++i) {
#pragma omp parallel for
    for (int64_t j = 0; j < DENSE_local_cols; ++j) {
      int g_row = indxl2g(i + 1, DENSE_NBROW, MYROW, ZERO, MPIGRID[0]) - 1;
      int g_col = indxl2g(j + 1, DENSE_NBCOL, MYCOL, ZERO, MPIGRID[1]) - 1;

      DENSE_MEM[i + j * DENSE_local_rows] =
        opts.kernel(domain.particles[g_row].coords,
                    domain.particles[g_col].coords);
    }
  }

  if (!MPIRANK) {
    std::cout << "begin construction.\n";
  }

  if (opts.admis_kind == GEOMETRY) {
    init_geometry_admis(A, domain, opts); // init admissiblity conditions with DTT
  }
  else if (opts.admis_kind == DIAGONAL) {
    init_diagonal_admis(A, domain, opts); // init admissiblity conditions with diagonal condition.
  }
  if(!MPIRANK) A.print_structure();
  construct_h2_matrix_dtd(A, domain, opts); // construct H2 matrix.
  auto stop_construct =  std::chrono::system_clock::now();
  double construct_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_construct - start_construct).count();

  int64_t dense_blocks = A.leaf_dense_blocks();
  construct_max_rank = A.max_rank(); // get max rank of H2 matrix post construct.

  // Allocate the vectors as a vector of Matrix objects of the form H2_A * x = b,
  // and dense_A * x = b_check.
  std::vector<Matrix> x, b, b_check;
  for (int i = MPIRANK; i < pow(2, A.max_level); i += MPISIZE) {
    x.push_back(Matrix(opts.nleaf, 1));
    b.push_back(Matrix(opts.nleaf, 1));
    b_check.push_back(Matrix(opts.nleaf, 1));
  }

  // scalapack data structures for x and b.
  std::vector<int> DESCB_CHECK(DESC_LEN), DESCX(DESC_LEN);
  int B_CHECK_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int B_CHECK_local_cols = numroc_(&ONE, &ONE, &MYCOL, &ZERO, &MPIGRID[1]);
  std::vector<double> B_CHECK_mem(B_CHECK_local_rows * B_CHECK_local_cols, 0),
    X_mem(B_CHECK_local_rows * B_CHECK_local_cols);

  descinit_(DESCB_CHECK.data(), &N, &ONE, &DENSE_NBROW, &ONE,
            &ZERO, &ZERO, &BLACS_CONTEXT, &B_CHECK_local_rows, &info);
  descinit_(DESCX.data(), &N, &ONE, &DENSE_NBROW, &ONE,
            &ZERO, &ZERO, &BLACS_CONTEXT, &B_CHECK_local_rows, &info);

  std::mt19937 gen(MPIRANK);
  std::uniform_real_distribution<double> dist(0, 1);
  for (int block = MPIRANK; block < pow(2, A.max_level); block += MPISIZE) {
    int numel = domain.cell_size(block, A.max_level);
    int index = block / MPISIZE;

    for (int i = 0; i < numel; ++i) {
      int g_row = block * numel + i + 1;
      double value = dist(gen);

      int l_row = indxg2l(g_row, DENSE_NBROW, MPIGRID[0]) - 1;
      int l_col = 0;

      x[index](i, 0) = value;  // assign random data to x.
      b[index](i, 0) = 0.0;    // set b to 0.
    }
  }

  redistribute_vector2scalapack(x, X_mem, A, opts);

  auto start_matvec = std::chrono::system_clock::now();
  matmul(A, domain, x, b);      // H2 matrix matvec. H2_A * x = b.
  auto stop_matvec = std::chrono::system_clock::now();
  double matvec_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_matvec - start_matvec).count();

  delete[] DENSE_MEM;           // free dense matrix to free space for parsec.

  // H2 matvec verification.
  double construction_error = 0;

  // ---- BEGIN PARSEC ----


  /* Initializing parsec context */
  // parsec = parsec_init( cores, NULL, NULL);
  // if( NULL == parsec ) {
  //   printf("Cannot initialize PaRSEC\n");
  //   exit(-1);
  // }

  // parsec_profiling_start();

  double solve_error = 0, fp_ops = 0, factorize_time = 0;

  // parsec_fini(&parsec);


  if (!MPIRANK) {
    // std::cout << "----------------------------\n";
    // std::cout << "N               : " << opts.N << std::endl;
    // std::cout << "ACCURACY        : " << opts.accuracy << std::endl;
    // std::cout << "OPT MAX RANK    : " << opts.max_rank << std::endl;
    // std::cout << "ADMIS           : " << opts.admis << std::endl;
    // std::cout << "REAL MAX RANK   : " << construct_max_rank << std::endl;
    // std::cout << "NPROCS          : " << MPISIZE << std::endl;
    // std::cout << "NLEAF           : " << opts.nleaf << "\n"
    //           << "CONSTRUCT ERROR : " << construction_error << std::endl
    //           << "H2 SOLVE ERROR  : " << solve_error << std::endl
    //           << "Contruct(ms)    : " << construct_time << std::endl
    //           << "Factorize (ms)  : " << factorize_time << std::endl
    //           << "PAPI FP OPS     : " << fp_ops << std::endl
    //           << "CORES           : " << cores << std::endl
    //           << "\n";
    // std::cout << "----------------------------\n";
    std::cout << "RESULT: "
              << opts.N << ","
              << opts.accuracy << ","
              << opts.max_rank << ","
              << opts.admis << ","
              << construct_max_rank << ","
              << MPISIZE << ","
              << opts.nleaf << ","
              << construction_error <<  ","
              << solve_error << ","
              << construct_time  << ","
              << factorize_time << ","
              << fp_ops << ","
              << cores << ","
              << dense_blocks
              << std::endl;
  }

  Cblacs_gridexit(BLACS_CONTEXT);
  Cblacs_exit(1);
  MPI_Finalize();

  return 0;
}
