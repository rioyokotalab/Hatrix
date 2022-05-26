#include <chrono>

#include "franklin/franklin.hpp"
#include "MPIWrapper.hpp"
#include "library_decls.hpp"
#include "matrix_construction.hpp"

#include "mpi.h"

int main(int argc, char* argv[]) {
  Hatrix::Args opts(argc, argv);
  mpi_world.init(argc, argv);
  random_generator.seed(mpi_world.MPIRANK);

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
  double construct_time, construct_error, matvec_error;

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();
    MPISymmSharedBasisMatrix A;
    if (opts.admis_kind == Hatrix::DIAGONAL) {
      init_diagonal_admis(A, opts);
    }
    construct_h2_miro(A, domain, opts);
    MPI_Barrier(MPI_COMM_WORLD);
    // Cblacs_barrier(mpi_world.CBLACS_CONTEXT, "All");
    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();
    construct_error = construct_error_mpi(A, domain, opts);

  }

  if (mpi_world.MPIRANK == 0) {
    std::cout << "construct time: " << construct_time << std::endl;
  }

  mpi_world.finish();
}
