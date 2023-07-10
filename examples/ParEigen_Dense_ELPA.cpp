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

#include <elpa/elpa.h>

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

void assert_elpa_ok(const int error, const std::string& error_str,
                    bool& success, std::string& error_msg) {
  if (error != ELPA_OK) {
    success = false;
    error_msg = error_str;
  }
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

  const int64_t print_csv_header = argc > 6 ? atol(argv[6]) : 1;
  // ELSES Input Files
  const std::string file_name = argc > 7 ? std::string(argv[7]) : "";
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

  std::vector<double> ev(N);
  double dense_eig_time = 0;
  // Initialize ELPA
  bool elpa_success = true;
  std::string elpa_error_str = "";
  elpa_t handle;
  int error;
  error = elpa_init(ELPA_API_VERSION);
  assert_elpa_ok(error, "ELPA API version not supported",
                 elpa_success, elpa_error_str);
  handle = elpa_allocate(&error);
  assert_elpa_ok(error, "ELPA instance allocation failed",
                 elpa_success, elpa_error_str);
  /* Set parameters the matrix and it's MPI distribution */
  elpa_set(handle, "na", N, &error);
  elpa_set(handle, "local_nrows", local_nrows, &error);
  elpa_set(handle, "local_ncols", local_ncols, &error);
  elpa_set(handle, "nblk", NB, &error);
  elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
  elpa_set(handle, "process_row", blacs_prow, &error);
  elpa_set(handle, "process_col", blacs_pcol, &error);
  assert_elpa_ok(error, "ELPA matrix initialization failed",
                 elpa_success, elpa_error_str);
  error = elpa_setup(handle);
  assert_elpa_ok(error, "ELPA setup failed",
                 elpa_success, elpa_error_str);
  if (elpa_success) {
    std::vector<double> A_copy(A);
    MPI_Barrier(MPI_COMM_WORLD);
    dense_eig_time -= MPI_Wtime();

    elpa_eigenvalues(handle, A.data(), ev.data(), &error);

    MPI_Barrier(MPI_COMM_WORLD);
    dense_eig_time += MPI_Wtime();
    assert_elpa_ok(error, "Call to elpa_eigenvalues failed",
                   elpa_success, elpa_error_str);
    // Verify against ScaLAPACK result
#if 0
    {
      // Compute all eigenvalues with pdsyev
      char jobz = 'N';
      char uplo = 'L';
      int m;
      std::vector<double> dense_eigv(N);
      int LWORK; double *WORK;
      MPI_Barrier(MPI_COMM_WORLD);
      // PDSYEV Work Query
      {
        LWORK = -1;
        WORK = new double[1];
        pdsyev_(&jobz, &uplo, &N, A_copy.data(), &ONE, &ONE, desc, dense_eigv.data(),
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
        pdsyev_(&jobz, &uplo, &N, A_copy.data(), &ONE, &ONE, desc, dense_eigv.data(),
                nullptr, nullptr, nullptr, nullptr, WORK, &LWORK, &info);
        delete[] WORK;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == 0) {
        double max_abs_err = 0;
        for (int64_t i = 0; i < N; i++) {
          max_abs_err = std::max(max_abs_err, std::abs(ev[i] - dense_eigv[i]));
        }
        printf("Max Abs Error vs. ScaLAPACK pdsyev = %.3e\n", max_abs_err);
      }
    }
#endif
  }

  // Print outputs
  if (mpi_rank == 0) {
    if (elpa_success) {
#ifndef OUTPUT_CSV
      printf("nprocs=%d N=%d NB=%d kernel=%s geometry=%s dense_eig_time=%.5lf\n",
             blacs_nprocs, N, NB, kernel_name.c_str(), geom_name.c_str(), dense_eig_time);
#else
      if (print_csv_header == 1) {
        printf("nprocs,N,NB,kernel,geometry,dense_eig_time\n");
      }
      printf("%d,%d,%d,%s,%s,%.5lf\n",
             blacs_nprocs, N, NB, kernel_name.c_str(), geom_name.c_str(), dense_eig_time);
#endif
    }
    else {
      printf("Error: %s\n", elpa_error_str.c_str());
    }
  }

  elpa_deallocate(handle, &error);
  elpa_uninit(&error);
  Hatrix::Context::finalize();
  Cblacs_gridexit(blacs_context);
  Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}

