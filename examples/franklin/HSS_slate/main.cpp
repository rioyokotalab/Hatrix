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

#include "MPISymmSharedBasisMatrix.hpp"
#include "matrix_construction.hpp"

#include "mpi.h"
#include <slate/slate.hh>

int main(int argc, char* argv[]) {
  Hatrix::Args opts(argc, argv);

  int provided;
  assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == 0);

  // generate the points on all the processes.
  auto start_domain = std::chrono::system_clock::now();
  Hatrix::Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == Hatrix::GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == Hatrix::CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  domain.divide_domain_and_create_particle_boxes(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  double construct_time;

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();
    MPISymmSharedBasisMatrix A;
    if (opts.admis_kind == Hatrix::DIAGONAL) {
      init_diagonal_admis(A, opts);
    }
    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();
  }

  assert(MPI_Finalize() == 0);

  return 0;
}
