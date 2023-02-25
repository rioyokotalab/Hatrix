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
generate_leaf_nodes(SymmetricSharedBasisMatrix& A,
                    const Domain& domain,
                    const Args& opts,
                    double* U_MEM, int* U) {
  int N = opts.N;
  int nleaf = opts.nleaf;
  int AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int AY_local_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);
  int AY[9]; int INFO;

  double* AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols]();
  descinit_(AY, &N, &nleaf, &nleaf, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &AY_local_nrows, &INFO);

  int64_t nblocks = pow(2, A.max_level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        if (mpi_rank(i) == MPIRANK) { // row-cyclic process distribution.
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
               &ALPHA,
               DENSE_MEM, &IA, &JA, DENSE.data(),
               &BETA,
               AY_MEM, &IA, &ONE, AY);
    }
  }

  int LWORK; double *WORK;

  // obtain the shared basis of each row.
  for (int64_t block = 0; block < nblocks; ++block) {
    const char JOB_U = 'V';
    const char JOB_VT = 'N';

    // init global S vector for this block.
    double *S_MEM = new double[(int64_t)nleaf];

    // SVD workspace query.
    {
      LWORK = -1;
      int IAY = nleaf * block + 1;
      int JAY = 1;
      int IU = nleaf * block + 1;
      int JU = 1;
      WORK = (double*)calloc(1, sizeof(double));

      pdgesvd_(&JOB_U, &JOB_VT,
               &nleaf, &nleaf,
               AY_MEM, &IAY, &JAY, AY,
               S_MEM,
               U_MEM, &IU, &ONE, U,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);

      LWORK = (int)WORK[0] + nleaf*nleaf; // workspace query throws a weird error so add nleaf*nleaf.
      free(WORK);
    }

    // SVD computation.
    {
      int IAY = nleaf * block + 1;
      int JAY = 1;
      int IU = nleaf * block + 1;
      int JU = 1;
      WORK = (double*)calloc(LWORK, sizeof(double));

      pdgesvd_(&JOB_U, &JOB_VT,
               &nleaf, &nleaf,
               AY_MEM, &IAY, &JAY, AY,
               S_MEM,
               U_MEM, &IU, &ONE, U,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);
      free(WORK);
    }

    // init cblacs info for the local U block.
    int U_LOCAL_CONTEXT;        // local CBLACS context
    int IMAP[1];                // workspace to map the original grid.
    int U_LOCAL_PNROWS, U_LOCAL_PNCOLS, U_LOCAL_PROW, U_LOCAL_PCOL; // local process grid parameters.
    IMAP[0] = mpi_rank(block);           // specify the rank from the global grid for the local grid.
    Cblacs_get(-1, 0, &U_LOCAL_CONTEXT);                    // init the new CBLACS context.
    Cblacs_gridmap(&U_LOCAL_CONTEXT, IMAP, ONE, ONE, ONE);  // init a 1x1 process grid.
    Cblacs_gridinfo(U_LOCAL_CONTEXT,                       // init grid params from the context.
                    &U_LOCAL_PNROWS, &U_LOCAL_PNCOLS, &U_LOCAL_PROW, &U_LOCAL_PCOL);

    // store opts.max_rank columns of U in the A.U for this process.
    // init local U block for communication.
    int U_LOCAL[9];
    int U_LOCAL_nrows = nleaf, U_LOCAL_ncols = opts.max_rank;
    Matrix U_LOCAL_MEM(U_LOCAL_nrows, U_LOCAL_ncols);
    descset_(U_LOCAL,
             &U_LOCAL_nrows, &U_LOCAL_ncols, &U_LOCAL_nrows, &U_LOCAL_ncols,
             &U_LOCAL_PROW, &U_LOCAL_PCOL, &U_LOCAL_CONTEXT, &U_LOCAL_nrows, &INFO);

    int IU = nleaf * block + 1;
    int JU = 1;
    pdgemr2d_(&U_LOCAL_nrows, &U_LOCAL_ncols,
              U_MEM, &IU, &JU, U,
              &U_LOCAL_MEM, &ONE, &ONE, U_LOCAL,
              &BLACS_CONTEXT);

    if (mpi_rank(block) == MPIRANK) {
      A.U.insert(block, A.max_level, std::move(U_LOCAL_MEM)); // store U in A.

      // Init US from the row vector.
      Matrix US(U_LOCAL_ncols, U_LOCAL_ncols);
      for (int64_t i = 0; i < U_LOCAL_ncols; ++i) { US(i,i) = S_MEM[i]; }
      A.US.insert(block, A.max_level, std::move(US));
    }

    int64_t rank = opts.max_rank;
    A.ranks.insert(block, A.max_level, std::move(rank));

    delete[] S_MEM;
  }

  // Generate S blocks for the lower triangle
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, A.max_level)) {
        // send U(j) to where S(i,j) exists.
        if (mpi_rank(j) == MPIRANK) {
          MPI_Request request;
          Matrix& Uj = A.U(j, A.max_level);
          MPI_Isend(&Uj, Uj.rows * Uj.cols, MPI_DOUBLE, mpi_rank(i),
                    j, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(i) == MPIRANK) {
          MPI_Status status;
          Matrix Uj(opts.nleaf, opts.max_rank);
          MPI_Recv(&Uj, Uj.rows * Uj.cols, MPI_DOUBLE, mpi_rank(j),
                   j, MPI_COMM_WORLD, &status);

          Matrix Aij = generate_p2p_interactions(domain,
                                                 i, j, A.max_level,
                                                 opts.kernel);
          Matrix S_block = matmul(matmul(A.U(i, A.max_level), Aij, true, false), Uj);
          A.S.insert(i, j, A.max_level, std::move(S_block));
        }
      }
    }
  }

  delete[] AY_MEM;
}

void
generate_transfer_matrices(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts,
                           int64_t level, double *U_MEM, int* U) {
  int64_t nblocks = pow(2, level);
  int64_t child_level = level + 1;

  // 1. Generate blocks from the current admissible blocks for this level.
  int N = opts.N;
  int level_block_size = N / nblocks;
  double ALPHA = 1.0;
  double BETA = 1.0;

  int AY_local_nrows = numroc_(&N, &level_block_size, &MYROW, &ZERO, &MPIGRID[0]);
  int AY_local_ncols = numroc_(&level_block_size, &level_block_size, &MYCOL, &ZERO,
                               &MPIGRID[1]);
  int AY[9]; int INFO;
  double *AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols];
  std::cout << "level : " << level << std::endl;
  descinit_(AY, &N, &level_block_size, &level_block_size, &level_block_size,
            &ZERO, &ZERO, &BLACS_CONTEXT, &AY_local_nrows, &INFO);

  // Allocate temporary AY matrix for accumulation of admissible blocks at this level.
  for (int64_t block = 0; block < nblocks; ++block) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (exists_and_inadmissible(A, block, j, level)) { continue; }

      int IA = level_block_size * block + 1;
      int JA = level_block_size * j + 1;

      // pdgeadd_(&NOTRANS, &level_block_);
    }
  }

  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t c1 = block * 2;
    int64_t c2 = block * 2 + 1;
    int64_t rank = opts.max_rank;
    int64_t block_size = A.ranks(c1, child_level) + A.ranks(c2, child_level);

    // 2.

    if (mpi_rank(block) == MPIRANK) {
      A.U.insert(block, level, generate_identity_matrix(block_size, opts.max_rank));
      A.US.insert(block, level, generate_identity_matrix(opts.max_rank, opts.max_rank));
    }

    A.ranks.insert(block, level, std::move(rank));
  }

  delete[] AY_MEM;
}

void
construct_h2_matrix_dtd(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  // init global U matrix
  int nleaf = opts.nleaf; int N = opts.N; int INFO;
  int U_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPIGRID[0]);
  int U_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]);
  double *U_MEM = new double[(int64_t)U_nrows * (int64_t)U_ncols];
  int U[9];
  descinit_(U, &N, &nleaf, &nleaf, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &U_nrows, &INFO);

  generate_leaf_nodes(A, domain, opts, U_MEM, U);

  for (int64_t level = A.max_level-1; level >= A.min_level; --level) {
    // int nblocks = pow(2, level);
    // int level_block_size = N / nblocks;
    // U_nrows = numroc_(&N, &level_block_size, &MYROW, &ZERO, &MPIGRID[0]);
    // U_ncols = numroc_(&level_block_size, &level_block_size, &MYCOL, &ZERO, &MPIGRID[1]);

    generate_transfer_matrices(A, domain, opts, level, U_MEM, U);
  }
  delete[] U_MEM;
}
