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



void
generate_leaf_nodes(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  int N = opts.N;
  int nleaf = opts.nleaf;
  int AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int AY_local_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);
  double *AY_MEM = NULL;
  int AY[9]; int INFO;

  AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols];
  descinit_(AY, &N, &nleaf, &DENSE_NBROW, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT, &AY_local_nrows, &INFO);

  int64_t nblocks = pow(2, A.max_level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        if (mpi_rank(i) == MPIRANK) {
          // regenerate the dense block to avoid communication.
          Matrix Aij = generate_p2p_interactions(domain,
                                                 i, j, A.max_level,
                                                 opts.kernel);
          A.D.insert(i, j, A.max_level, std::move(Aij));
        }
      }
    }
  }

  double ALPHA = 1.0;
  double BETA = 1.0;
  // Accumulate admissible blocks from the large distributed dense matrix.
  for (int64_t block = 0; block < nblocks; ++block) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (exists_and_inadmissible(A, block, j, A.max_level)) { continue; }

      int IA = nleaf * block + 1;
      int JA = nleaf * j + 1;

      pdgeadd_(&NOTRANS, &nleaf, &nleaf,
               &ALPHA, DENSE_MEM, &IA, &JA, DENSE.data(),
               &BETA,
               AY_MEM, &IA, &ONE, AY);
    }
  }

  delete[] AY_MEM;
}

void
construct_h2_matrix_dtd(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  generate_leaf_nodes(A, domain, opts);

}
