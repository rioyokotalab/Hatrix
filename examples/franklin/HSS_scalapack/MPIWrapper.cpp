#include "MPIWrapper.hpp"
#include "library_decls.hpp"

#include "mpi.h"
#include <cassert>

void
MPIWrapper::init(int argc, char* argv[]) {
  int provided;
  assert(MPI_Init_thread(&argc, &argv,
                         MPI_THREAD_MULTIPLE,
                         &provided) == 0);
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  COMM = MPI_COMM_WORLD;

  MPI_Dims_create(MPISIZE, 2, MPIGRID);

  Cblacs_get( -1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit(&BLACS_CONTEXT, "Row", MPIGRID[0],
                   MPIGRID[1]);
  Cblacs_pcoord(BLACS_CONTEXT,
                MPIRANK,
                &ROWRANK, &COLRANK);
}

void
MPIWrapper::finish() {
  assert(MPI_Finalize() == 0);
}

MPIWrapper mpi_world;
