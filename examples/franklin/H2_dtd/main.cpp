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

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "parsec.h"


using namespace Hatrix;
parsec_context_t *parsec = NULL;
parsec_taskpool_t *dtd_tp = NULL;
int MPIRANK, MPISIZE;

int main (int argc, char **argv) {
  Hatrix::Context::init();

  int rc;
  int world, mpi_rank;
  int cores = -1;

  Args opts(argc, argv);

  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);

  /* Initializing parsec context */
  int parsec_argc = argc + opts.num_args;
  parsec = parsec_init( cores, &parsec_argc, &argv + opts.num_args);
  if( NULL == parsec ) {
    printf("Cannot initialize PaRSEC\n");
    exit(-1);
  }

  /* Initializing parsec handle(collection of tasks) */
  dtd_tp = parsec_dtd_taskpool_new();

  /* Registering the dtd_handle with PARSEC context */
  rc = parsec_context_add_taskpool( parsec, dtd_tp );

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


  /* Cleaning the parsec handle */
  parsec_taskpool_free( dtd_tp );

  /* Cleaning up parsec context */
  parsec_fini(&parsec);

  MPI_Finalize();

  Hatrix::Context::finalize();

  return 0;
}
