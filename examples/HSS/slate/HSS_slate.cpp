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

#include "mpi.h"
#include <slate/slate.hh>

#include "../Args.hpp"

typedef struct MPISymmSharedBasisMatrix {

} MPISymmSharedBasisMatrix;

int main(int argc, char* argv[]) {
  Args args(argc, argv);

  int provided;
  assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == 0);

  assert(MPI_Finalize() == 0);

  return 0;
}