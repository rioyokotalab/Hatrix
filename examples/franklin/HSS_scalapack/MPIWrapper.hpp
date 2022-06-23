#pragma once

#include "mpi.h"

class MPIWrapper {
public:
  int MPISIZE;
  int MPIRANK;
  MPI_Comm COMM;
  int MPIGRID[2] = {0, 0};
  int CBLACS_CONTEXT;

  int ROWRANK, COLRANK;

  void init(int argc, char* argv[]);
  void finish();
};

extern MPIWrapper mpi_world;
