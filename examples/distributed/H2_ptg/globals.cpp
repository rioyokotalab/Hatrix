#include "Hatrix/Hatrix.h"
#include "globals.hpp"

parsec_context_t *parsec = NULL;
parsec_taskpool_t *dtd_tp = NULL;
int MPIRANK, MPISIZE, MPIGRID[2],
  MYROW, MYCOL, info;
int N;
int BLACS_CONTEXT;

int ZERO = 0;
int ONE = 1;
int MINUS_ONE = -1;

const char NOTRANS = 'N';
const char TRANS = 'T';

double *DENSE_MEM = NULL;
std::vector<int> DENSE(9);
int DENSE_NBROW, DENSE_NBCOL;
int DENSE_local_rows, DENSE_local_cols;

int
indxg2l(int INDXGLOB, int NB, int NPROCS) {
  return NB * ((INDXGLOB - 1) / ( NB * NPROCS)) + (INDXGLOB - 1) % NB + 1;
}

int
indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc - isrcproc) % nprocs) * nb + 1;
}

int
indxg2p(int INDXGLOB, int NB, int ISRCPROC, int NPROCS) {
  return (ISRCPROC + (INDXGLOB - 1) / NB) % NPROCS;
}

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
