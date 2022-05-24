#pragma once

#include "mpi.h"

class MPIWrapper {
public:
  int MPISIZE;
  int MPIRANK;
  MPI_Comm COMM;
  int MPIGRID[2];

  void init(int argc, char* argv[]);
  void finish();
};

extern MPIWrapper mpi_world;
