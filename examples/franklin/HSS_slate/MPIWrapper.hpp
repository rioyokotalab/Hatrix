#pragma once

class MPIWrapper {
public:
  int MPISIZE;
  int MPIRANK;
  int COMM;
  int MPIGRID[2];

  void init(int argc, char* argv[]);
  void finish();
};

extern MPIWrapper mpi_world;
