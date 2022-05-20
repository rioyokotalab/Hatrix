#include "MPIWrapper.hpp"

#include "mpi.h"
#include <cassert>

void
MPIWrapper::init(int argc, char* argv[]) {
  int provided;
  assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == 0);
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  COMM = MPI_COMM_WORLD;
}

void
MPIWrapper::finish() {
  assert(MPI_Finalize() == 0);
}
