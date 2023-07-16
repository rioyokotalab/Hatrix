#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>
#include <mpi.h>

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

#define OUTPUT_CSV

// ScaLAPACK Fortran Interface
extern "C" {
  /* Cblacs declarations: https://netlib.org/blacs/BLACS/QRef.html */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int CONTEXT, int WHAT, int*VALUE);
  void Cblacs_gridinit(int*, const char*, int, int);
  // returns the co-ordinates of the process number PNUM in PROW and PCOL.
  void Cblacs_pcoord(int CONTEXT, int PNUM, int* PROW, int* PCOL);
  void Cblacs_gridexit(int CONTEXT);
  void Cblacs_barrier(int, const char*);
  void Cblacs_exit(int CONTINUE);
  void Cblacs_gridmap(int* CONTEXT, int* USERMAP, const int LDUMAP,
                      const int NPROW, const int NPCOL);
  void Cblacs_gridinfo(int CONTEXT, int *NPROW, int *NPCOL,
                       int *MYROW, int *MYCOL);

  // calculate the number of rows and cols owned by process IPROC.
  // IPROC :: (local input) INTEGER
  //          The coordinate of the process whose local array row or
  //          column is to be determined.
  // ISRCPROC :: The coordinate of the process that possesses the first
  //             row or column of the distributed matrix. Global input.
  int numroc_(const int* N, const int* NB, const int* IPROC, const int* ISRCPROC,
              const int* NPROCS);

  // init descriptor for scalapack matrices.
  void descinit_(int *desc,
                 const int *m,  const int *n, const int *mb, const int *nb,
                 const int *irsrc, const int *icsrc,
                 const int *BLACS_CONTEXT,
                 const int *lld, int *info);

  // set values of the descriptor without error checking.
  void descset_(int *desc, const int *m,  const int *n, const int *mb,
                const int *nb, const int *irsrc, const int *icsrc, const int *BLACS_CONTEXT,
                const int *lld, int *info);
  // Compute reference eigenvalues
  void pdsyev_(char*, char*, int*, double*, int*, int*, int*,
               double*, double*, int*, int*, int*, double*, int*, int*);
  // Compute selected eigenvalues
  void pdsyevx_(char*, char*, char*, int*, double*, int*, int*, int*,
                double*, double*, int*, int*, double*, int*, int*,
                double*, double*, double*, int*, int*, int*,
                double*, int*, int*, int*, int*, int*, double*, int*);
}

// Translate global indices to local indices. INDXGLOB is the global index for the
// row/col. Returns the local FORTRAN-style index. NPROCS is the number of processes
// in that row or col.
int indxg2l(int INDXGLOB, int NB, int NPROCS) {
  return NB * ((INDXGLOB - 1) / ( NB * NPROCS)) + (INDXGLOB - 1) % NB + 1;
}
int indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc - isrcproc) % nprocs) * nb + 1;
}
int indxg2p(int INDXGLOB, int NB, int ISRCPROC, int NPROCS) {
  return (ISRCPROC + (INDXGLOB - 1) / NB) % NPROCS;
}

int main(int argc, char ** argv) {
  int mpi_rank, mpi_nprocs, mpi_grid[2] = {0};
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);
  MPI_Dims_create(mpi_nprocs, 2, mpi_grid);

  // Parse Input
  int N = argc > 1 ? atol(argv[1]) : 256;
  int NB = argc > 2 ? atol(argv[2]) : 32;
  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: ELSES Dense Matrix
  const int64_t kernel_type = argc > 3 ? atol(argv[3]) : 0;
  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  const int64_t geom_type = argc > 4 ? atol(argv[4]) : 0;
  int64_t ndim  = argc > 5 ? atol(argv[5]) : 1;
  // Eigenvalue computation parameters
  double abs_tol = argc > 6 ? atof(argv[6]) : 1e-3;
  const int64_t k_begin = argc > 7 ? atol(argv[7]) : 1;
  const int64_t k_end = argc > 8 ? atol(argv[8]) : k_begin;
  const bool compute_eig_acc = argc > 9 ? (atol(argv[9]) == 1) : true;
  const int64_t print_csv_header = argc > 10 ? atol(argv[10]) : 1;
  // ELSES Input Files
  const std::string file_name = argc > 11 ? std::string(argv[11]) : "";
  Hatrix::Context::init();
  // Choose kernel
  Hatrix::set_kernel_constants(1e-3, 1.);
  std::string kernel_name = "";
  switch (kernel_type) {
    case 0: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
      break;
    }
    case 1: {
      Hatrix::set_kernel_function(Hatrix::yukawa_kernel);
      kernel_name = "yukawa";
      break;
    }
    case 2: {
      Hatrix::set_kernel_function(Hatrix::ELSES_dense_input);
      kernel_name = "ELSES-kernel";
      break;
    }
    default: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
    }
  }
  // Initialize Domain
  Hatrix::Domain domain(N, ndim);
  std::string geom_name = std::to_string(ndim) + "d-";
  switch (geom_type) {
    case 0: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
      break;
    }
    case 1: {
      domain.initialize_unit_cubical_mesh();
      geom_name += "cubical_mesh";
      break;
    }
    case 2: {
      domain.initialize_starsh_uniform_grid();
      geom_name += "starsh_uniform_grid";
      break;
    }
    case 3: {
      domain.ndim = 3;
      geom_name = file_name;
      break;
    }
    default: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
    }
  }
  // Pre-processing step for ELSES geometry
  const bool is_non_synthetic = (geom_type == 3);
  if (is_non_synthetic) {
    domain.read_bodies_ELSES(file_name + ".xyz");
    domain.read_p2p_matrix_ELSES(file_name + ".dat");
    N = domain.N;
  }
  domain.build_tree(N);

  // BLACS Initialization
  int ZERO = 0;
  int ONE = 1;
  int blacs_context, blacs_rank, blacs_nprocs, blacs_prow, blacs_pcol;
  // Get BLACS context
  Cblacs_get(0, 0, &blacs_context);
  // Initialize BLACS grid
  Cblacs_gridinit(&blacs_context, "Row", mpi_grid[0], mpi_grid[1]);
  // Get information about grid and local processes
  Cblacs_pinfo(&blacs_rank, &blacs_nprocs);
  Cblacs_gridinfo(blacs_context, &mpi_grid[0], &mpi_grid[1], &blacs_prow, &blacs_pcol);
  const int local_nrows = numroc_(&N, &NB, &blacs_prow, &ZERO, &mpi_grid[0]);
  const int local_ncols = numroc_(&N, &NB, &blacs_pcol, &ZERO, &mpi_grid[1]);
  const int local_stride = local_nrows;

  // Initialize local matrix
  int info, desc[9];
  descinit_(desc, &N, &N, &NB, &NB, &ZERO, &ZERO, &blacs_context, &local_stride, &info);
  std::vector<double> A(local_nrows * local_ncols);
  for (int64_t i = 0; i < local_nrows; i++) {
    for (int64_t j = 0; j < local_ncols; j++) {
      const int g_row = indxl2g(i + 1, NB, blacs_prow, ZERO, mpi_grid[0]) - 1;
      const int g_col = indxl2g(j + 1, NB, blacs_pcol, ZERO, mpi_grid[1]) - 1;
      A[i + j * local_stride] =
          Hatrix::kernel_function(domain, domain.bodies[g_row], domain.bodies[g_col]);
    }
  }
  // Initialize a copy for accuracy computation with pdsyev
  std::vector<double> A_ref;
  if (compute_eig_acc) {
    A_ref = A;
  }

  // Compute selected eigenvalues with pdsyevx
  char jobz = 'N';
  char range = 'I';
  char uplo = 'L';
  int il = k_begin;
  int iu = k_end;
  int m;
  std::vector<double> actual_eigv(N);
  double DZERO = 0;
  int LWORK; double *WORK;
  int NNP, LIWORK; int *IWORK;
  double actual_eig_time = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  actual_eig_time -= MPI_Wtime();
  // PDSYEVX Work Query
  {
    LWORK = -1;
    LIWORK = -1;
    WORK = new double[1];
    IWORK = new int[1];
    MPI_Barrier(MPI_COMM_WORLD);
    pdsyevx_(&jobz, &range, &uplo, &N, A.data(), &ONE, &ONE, desc, nullptr, nullptr,
             &il, &iu, &abs_tol, &m, nullptr, actual_eigv.data(), nullptr, nullptr, nullptr,
             nullptr, nullptr, WORK, &LWORK, IWORK, &LIWORK, nullptr, nullptr, nullptr, &info);
    MPI_Barrier(MPI_COMM_WORLD);
    if (info != 0) {
      printf("Process-%d: Error in pdsyevx workspace query, info=%d\n", mpi_rank, info);
    }
    LWORK = (int)WORK[0];
    // Manual LIWORK calculation
    // For some reason workspace query gives a very big LIWORK, which requires 9.5 GB of memory
    // Use minimal LIWORK instead based on ScaLAPACK reference:
    // LIWORK >= 6 * NNP
    // Where NNP = MAX( N, NPROW*NPCOL + 1, 4 )
    NNP = N;
    NNP = std::max(NNP, mpi_grid[0] * mpi_grid[1] + 1);
    NNP = std::max(NNP, (int)4);
    LIWORK = std::min(6 * NNP, IWORK[0]);
    delete[] WORK;
    delete[] IWORK;
  }
  // PDSYEVX Computation
  {
    WORK = new double[LWORK];
    IWORK = new int[LIWORK];
    MPI_Barrier(MPI_COMM_WORLD);
    pdsyevx_(&jobz, &range, &uplo, &N, A.data(), &ONE, &ONE, desc, nullptr, nullptr,
             &il, &iu, &abs_tol, &m, nullptr, actual_eigv.data(), nullptr, nullptr, nullptr,
             nullptr, nullptr, WORK, &LWORK, IWORK, &LIWORK, nullptr, nullptr, nullptr, &info);
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] WORK;
    delete[] IWORK;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  actual_eig_time += MPI_Wtime();

  // Check eigenvalue accuracy if needed
  std::vector<double> dense_eigv(N);
  double dense_eig_time = 0;
  if (compute_eig_acc) {
    // Compute reference eigenvalues with pdsyev
    MPI_Barrier(MPI_COMM_WORLD);
    dense_eig_time -= MPI_Wtime();
    // PDSYEV Work Query
    {
      LWORK = -1;
      WORK = new double[1];
      pdsyev_(&jobz, &uplo, &N, A_ref.data(), &ONE, &ONE, desc, dense_eigv.data(),
              nullptr, nullptr, nullptr, nullptr, WORK, &LWORK, &info);
      if (info != 0) {
        printf("Process-%d: Error in pdsyev workspace query, info=%d\n", mpi_rank, info);
      }
      LWORK = (int)WORK[0];
      delete[] WORK;
    }
    // PDSYEV Computation
    {
      WORK = new double[LWORK];
      pdsyev_(&jobz, &uplo, &N, A_ref.data(), &ONE, &ONE, desc, dense_eigv.data(),
              nullptr, nullptr, nullptr, nullptr, WORK, &LWORK, &info);
      delete[] WORK;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    dense_eig_time += MPI_Wtime();
  }
  // Print outputs
  if (mpi_rank == 0) {
#ifndef OUTPUT_CSV
    printf("nprocs=%d mpi_nprocs=%d mpi_grid_x=%d mpi_grid_y=%d"
           " N=%d NB=%d kernel=%s geometry=%s abs_tol=%.1e k_begin=%d k_end=%d"
           " dense_eig_time=%.3lf, actual_eig_time=%.3lf\n",
           blacs_nprocs, mpi_nprocs, mpi_grid[0], mpi_grid[1],
           N, NB, kernel_name.c_str(),
           geom_name.c_str(), abs_tol, (int)k_begin, (int)k_end, dense_eig_time, actual_eig_time);
#else
    if (print_csv_header == 1) {
      printf("nprocs,mpi_nprocs,mpi_grid_x,mpi_grid_y"
             ",N,NB,kernel,geometry,abs_tol,k_begin,k_end,dense_eig_time"
             ",actual_eig_time,k,dense_eigv,actual_eigv,eig_abs_err\n");
    }
#endif
    for (int64_t k = k_begin; k <= k_end; k++) {
      const auto dense_eigv_k = compute_eig_acc ? dense_eigv[k - 1] : -1;
      const auto actual_eigv_k = actual_eigv[k - k_begin];
      const auto eig_abs_err = compute_eig_acc ? std::abs(dense_eigv_k - actual_eigv_k) : -1;
#ifndef OUTPUT_CSV
      printf("k=%d dense_eigv=%.8lf actual_eigv=%.8lf eig_abs_err=%.2e\n",
             (int)k, dense_eigv_k, actual_eigv_k, eig_abs_err);
#else
      printf("%d,%d,%d,%d,%d,%d,%s,%s,%.1e,%d,%d,%.3lf,%.3lf,%d,%.8lf,%.8lf,%.2e\n",
             blacs_nprocs, mpi_nprocs, mpi_grid[0], mpi_grid[1],
             N, NB, kernel_name.c_str(), geom_name.c_str(), abs_tol,
             (int)k_begin, (int)k_end, dense_eig_time, actual_eig_time,
             (int)k, dense_eigv_k, actual_eigv_k, eig_abs_err);
#endif
    }
  }

  Hatrix::Context::finalize();
  Cblacs_gridexit(blacs_context);
  Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}

