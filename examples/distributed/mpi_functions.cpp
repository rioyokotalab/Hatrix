#include "distributed/distributed.hpp"

int MPIRANK, MPISIZE, MPIGRID[2], MYROW, MYCOL, info;

int
mpi_rank(int i) {
  return (i % MPISIZE);
}

int
mpi_rank(int i, int j) {
  if (i == j) {
    return mpi_rank(i);
  }
  // row major distribution similar to scalapack.
  return (i % MPIGRID[0]) * MPIGRID[1] + (j % MPIGRID[1]);
}
