#include "Hatrix/Hatrix.h"
#include "distributed/scalapack_functions.hpp"

int BLACS_CONTEXT;
int ZERO = 0;
int ONE = 1;
int MINUS_ONE = -1;

const char NOTRANS = 'N';
const char TRANS = 'T';

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
