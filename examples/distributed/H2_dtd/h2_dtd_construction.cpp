#include "mpi.h"
#include <random>
#include <algorithm>
#include <set>
#include <cmath>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "globals.hpp"
#include "h2_dtd_construction.hpp"

using namespace Hatrix;

double *AY_MEM = NULL;
int AY[9];
int64_t AY_local_nrows, AY_local_ncols;

void
generate_leaf_nodes(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  int N = opts.N;
  int nleaf = opts.nleaf;
  int64_t AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int64_t AY_local_ncols = numroc_(&N, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);

  AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols];

  delete[] AY_MEM;
}

void
construct_h2_matrix_dtd(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  generate_leaf_nodes(A, domain, opts);

}
