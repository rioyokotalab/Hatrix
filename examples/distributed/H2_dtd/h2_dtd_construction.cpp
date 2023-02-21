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
generate_leaf_nodes(SymmetricSharedBasisMatrix& A, const Domain& domain,
                    const Args& opts) {
  int N = opts.N;
  int nleaf = opts.nleaf;
  int AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int AY_local_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);
  int AY[9]; int INFO;

  double* AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols];
  descinit_(AY, &N, &nleaf, &DENSE_NBROW, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &AY_local_nrows, &INFO);

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

  // init global U matrix
  int U_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int U_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);
  double *U_MEM = new double[(int64_t)U_nrows * (int64_t)U_ncols];
  int U[9];
  descinit_(U, &N, &nleaf, &nleaf, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &U_nrows, &INFO);

  // obtain the shared basis of each row.
  for (int64_t block = 0; block < nblocks; ++block) {
    char JOB_U = 'V';
    char JOB_VT = 'N';
    int IA = nleaf * block + 1;
    int JA = 1;
    int IU = nleaf * block + 1;
    int JU = 1;

    // init global S vector for this block.
    double *S_MEM = new double[(int64_t)nleaf];
    double *WORK = new double[1];
    int LWORK = -1;

    // SVD workspace query.
    pdgesvd_(&JOB_U, &JOB_VT,
             &nleaf, &nleaf,
             AY_MEM, &IA, &JA, AY,
             S_MEM,
             U_MEM, &IU, &JU, U,
             NULL, NULL, NULL, NULL, // not calculating VT so NULL
             WORK, &LWORK,
             &INFO);
    LWORK = WORK[0];
    delete[] WORK;

    // SVD computation.
    WORK = new double[(int64_t)LWORK];
    pdgesvd_(&JOB_U, &JOB_VT,
             &nleaf, &nleaf,
             AY_MEM, &IA, &JA, AY,
             S_MEM,
             U_MEM, &IU, &JU, U,
             NULL, NULL, NULL, NULL, // not calculating VT so NULL
             WORK, &LWORK,
             &INFO);


    // init cblacs info for the local U block.
    int U_LOCAL_CONTEXT;        // local CBLACS context
    int IMAP[1];                // workspace to map the original grid.
    int LOCAL_PNROWS, LOCAL_PNCOLS, LOCAL_PROW, LOCAL_PCOL; // local process grid parameters.
    Cblacs_get(-1, 0, &U_LOCAL_CONTEXT);                    // init the new CBLACS context.
    Cblacs_gridmap(&U_LOCAL_CONTEXT, IMAP, ONE, ONE, ONE);  // init a 1x1 process grid.
    Cblacs_gridinfo(U_LOCAL_CONTEXT,                       // init grid params from the context.
                    &LOCAL_PNROWS, &LOCAL_PNCOLS, &LOCAL_PROW, &LOCAL_PCOL);

    // store opts.max_rank columns of U in the A.U for this process.
    // init local U block for communication.
    int U_LOCAL[9];
    int U_LOCAL_nrows = nleaf, U_LOCAL_ncols = opts.max_rank;

    // descset_(U_LOCAL,
    //          &U_LOCAL_nrows, &U_LOCAL_ncols, &U_LOCAL_nrows, &U_LOCAL_ncols,
    //          &);

    Matrix U(U_LOCAL_nrows, U_LOCAL_ncols);

    delete[] WORK;
    delete[] S_MEM;
  }

  delete[] AY_MEM;
  delete[] U_MEM;
}

void
construct_h2_matrix_dtd(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  generate_leaf_nodes(A, domain, opts);

}
