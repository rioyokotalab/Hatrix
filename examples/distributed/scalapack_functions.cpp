#include "mpi.h"
#include <random>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstring>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

using namespace Hatrix;

double *U_MEM, *U_REAL_MEM;
int U[9];

void
generate_leaf_nodes(SymmetricSharedBasisMatrix& A,
                    const Domain& domain,
                    const Args& opts, double* DENSE_MEM, std::vector<int>& DENSE) {
  if (!MPIRANK) {
    std::cout << "generate leaf nodes.\n";
  }
  int N = opts.N;
  int nleaf = opts.nleaf;
  int AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPISIZE);
  int AY_local_ncols = numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &ONE);
  int AY[9]; int INFO;
  double* AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols]();
  descinit_(AY, &N, &nleaf, &nleaf, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &AY_local_nrows, &INFO);

  int64_t nblocks = pow(2, A.max_level);
  int rank = opts.max_rank;

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
  const char JOB_U = 'V';
  const char JOB_VT = 'N';
  // obtain the shared basis of each row.
  for (int64_t block = 0; block < nblocks; ++block) {
    // init global S vector for this block.
    double *S_MEM = new double[(int64_t)nleaf]();

    // SVD workspace query.
    {
      LWORK = -1;
      int IAY = nleaf * block + 1;
      int JAY = 1;
      int IU = nleaf * block + 1;
      int JU = 1;
      WORK = new double[1]();

      pdgesvd_(&JOB_U, &JOB_VT,
               &nleaf, &nleaf,
               AY_MEM, &IAY, &JAY, AY,
               S_MEM,
               U_MEM, &IU, &ONE, U,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);

      LWORK = WORK[0] + nleaf * (AY_local_nrows + AY_local_ncols + 1) + AY_local_ncols;
      // workspace query throws a weird error so add nleaf*rank.

      delete[] WORK;
    }

    // SVD computation.
    {
      int IAY = nleaf * block + 1;
      int JAY = 1;
      int IU = nleaf * block + 1;
      int JU = 1;
      WORK =  new double[(int64_t)LWORK]();

      // std::cout << "IAY: " << IAY << std::endl;

      pdgesvd_(&JOB_U, &JOB_VT,
               &nleaf, &nleaf,
               AY_MEM, &IAY, &JAY, AY,
               S_MEM,
               U_MEM, &IU, &ONE, U,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);
      delete[] WORK;
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
    if (IMAP[0] != MPIRANK) {
      U_LOCAL_CONTEXT = -1;
    }
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
    if (MPIRANK == IMAP[0]) {
      Cblacs_gridexit(U_LOCAL_CONTEXT);
    }
  }

  // Generate S blocks for the lower triangle
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, A.max_level)) {
        // send U(j) to where S(i,j) exists.
        if (mpi_rank(j) == MPIRANK) {
          MPI_Request request;
          Matrix& Uj = A.U(j, A.max_level);
          // std::cout << "\n\n@@@@ SENT TAG " << j << " @@@@\n\n";
          MPI_Isend(&Uj, Uj.rows * Uj.cols, MPI_DOUBLE, mpi_rank(i),
                    j, MPI_COMM_WORLD, &request);
        }

      }
    }
    for (int j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, A.max_level)) {
        if (mpi_rank(i) == MPIRANK) {
          MPI_Status status;
          Matrix Uj(opts.nleaf, opts.max_rank);
          // std::cout << "\n\n@@@@ GOT TAG " << j << " @@@@\n\n";
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

static bool
row_has_admissible_blocks(const SymmetricSharedBasisMatrix& A, int64_t row,
                          int64_t level) {
  bool has_admis = false;

  for (int64_t i = 0; i < pow(2, level); ++i) {
    if (!A.is_admissible.exists(row, i, level) || exists_and_admissible(A, row, i, level)) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

void
generate_transfer_matrices(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts,
                           int64_t level, double* DENSE_MEM, std::vector<int>& DENSE) {
  if (!MPIRANK) {
    std::cout << "generate transfer matrices level=" << level << std::endl;
  }
  int INFO;
  int N = opts.N;
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  int64_t child_nblocks = pow(2, child_level);
  int rank = opts.max_rank;
  int level_block_size = N / nblocks;
  // 1. Generate blocks from the current admissible blocks for this level.
  int nleaf = opts.nleaf;

  int AY_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPISIZE);
  int AY_local_ncols = fmax(numroc_(&level_block_size, &nleaf, &MYCOL, &ZERO,
                                    &ONE), 1);
  int AY[9];
  if (!MPIRANK) {
    std::cout << "\t ALLOCATE AY_MEM" << std::endl;
  }
  double *AY_MEM = new double[(int64_t)AY_local_nrows * (int64_t)AY_local_ncols]();
  descinit_(AY, &N, &level_block_size, &nleaf, &nleaf,
            &ZERO, &ZERO, &BLACS_CONTEXT, &AY_local_nrows, &INFO);

  if (!MPIRANK) {
    std::cout << "\t DONE ALLOCATE AY_MEM" << std::endl;
  }

  // Allocate temporary AY matrix for accumulation of admissible blocks at this level.
  double ALPHA = 1.0;
  double BETA = 1.0;
  for (int64_t block = 0; block < nblocks; ++block) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (exists_and_inadmissible(A, block, j, level)) { continue; }

      int IA = level_block_size * block + 1;
      int JA = level_block_size * j + 1;

      pdgeadd_(&NOTRANS, &level_block_size, &level_block_size,
               &ALPHA,
               DENSE_MEM, &IA, &JA, DENSE.data(),
               &BETA,
               AY_MEM, &IA, &ONE, AY);
    }
  }

  if (!MPIRANK) {
    std::cout << "\t DONE ADD" << std::endl;
  }

  // Allocate a temporary global matrix to store the product of the real basis with the
  // summation of the admissible blocks.

  int block_nrows = rank * 2;
  int TEMP_nrows = nblocks * block_nrows;
  int TEMP_local_nrows = fmax(numroc_(&TEMP_nrows, &rank, &MYROW, &ZERO, &MPISIZE),1);
  int TEMP_local_ncols = fmax(numroc_(&level_block_size, &nleaf,
                                      &MYCOL, &ZERO, &ONE), 1);
  int TEMP[9];
  // Use rank as NB cuz it throws a 806 error for an unknown reason.
  descinit_(TEMP, &TEMP_nrows, &level_block_size, &rank, &rank, &ZERO, &ZERO,
            &BLACS_CONTEXT, &TEMP_local_nrows, &INFO);
  double *TEMP_MEM = new double[(int64_t)TEMP_local_nrows * (int64_t)TEMP_local_ncols]();
  int child_block_size = N / child_nblocks;

  // 2. Apply the real basis U to the summation of the admissible blocks.
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t c1 = block * 2;
    int64_t c2 = block * 2 + 1;
    int rank = opts.max_rank;

    double ALPHA = 1.0;
    double BETA = 1.0;

    // Upper basis block.
    int IU = c1 * child_block_size + 1;
    int JU = 1;
    int IA = c1 * child_block_size + 1;
    int JA = 1;
    int ITEMP = c1 * rank + 1;
    int JTEMP = 1;
    pdgemm_(&TRANS, &NOTRANS,
            &rank, &child_block_size, &child_block_size,
            &ALPHA,
            U_MEM, &IU, &JU, U,
            AY_MEM, &IA, &JA, AY,
            &BETA,
            TEMP_MEM, &ITEMP, &JTEMP, TEMP);

    // Lower basis block.
    IU = c2 * child_block_size + 1;
    JU = 1;
    IA = c2 * child_block_size + 1;
    JA = 1;
    ITEMP = c2 * rank + 1;
    JTEMP = 1;
    pdgemm_(&TRANS, &NOTRANS,
            &rank, &child_block_size, &child_block_size,
            &ALPHA,
            U_MEM, &IU, &JU, U,
            AY_MEM, &IA, &JA, AY,
            &BETA,
            TEMP_MEM, &ITEMP, &JTEMP, TEMP);
  }
  // 3. Calcuate the SVD of the applied block to generate the transfer matrix.
  // Allocate a global matrix to store the transfer matrices. The transfer matrices for the
  // entire level are stacked by row in this global matrix.
  int UTRANSFER_local_nrows = fmax(numroc_(&TEMP_nrows, &rank, &MYROW, &ZERO, &MPISIZE), 1);
  int UTRANSFER_local_ncols = fmax(numroc_(&rank, &rank, &MYCOL, &ZERO, &ONE), 1);
  int UTRANSFER[9];

  descinit_(UTRANSFER, &TEMP_nrows, &rank, &rank, &rank, &ZERO, &ZERO,
            &BLACS_CONTEXT, &UTRANSFER_local_nrows, &INFO);
  double *UTRANSFER_MEM =
    new double[(int64_t)TEMP_local_nrows * (int64_t)TEMP_local_ncols]();

  int LWORK; double *WORK;

  const char JOB_U = 'V';
  const char JOB_VT = 'N';

  for (int64_t block = 0; block < nblocks; ++block) {
    double *S_MEM = new double[(int64_t)rank]();
    // SVD workspace query.
    {
      LWORK = -1;
      int ITEMP = block * block_nrows + 1;
      int JTEMP = 1;
      int IU = block * block_nrows + 1;
      int JU = 1;

      WORK = new double[1]();

      pdgesvd_(&JOB_U, &JOB_VT,
               &block_nrows, &rank,
               TEMP_MEM, &ITEMP, &JTEMP, TEMP,
               S_MEM,
               UTRANSFER_MEM, &IU, &JU, UTRANSFER,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);
      LWORK = WORK[0] + nleaf * (TEMP_local_nrows + TEMP_local_ncols + 1) + TEMP_local_ncols;
      delete[] WORK;
    }

    // SVD computation
    {
      int ITEMP = block * block_nrows + 1;
      int JTEMP = 1;
      int IU = block * block_nrows + 1;
      int JU = 1;
      WORK = new double[(int64_t)LWORK]();

      pdgesvd_(&JOB_U, &JOB_VT,
               &block_nrows, &rank,
               TEMP_MEM, &ITEMP, &JTEMP, TEMP,
               S_MEM,
               UTRANSFER_MEM, &IU, &JU, UTRANSFER,
               NULL, NULL, NULL, NULL,
               WORK, &LWORK,
               &INFO);
      delete[] WORK;
    }

    // Init CBLACS info for the local U block.
    int U_LOCAL_CONTEXT;
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
    int U_LOCAL_nrows = opts.max_rank * 2, U_LOCAL_ncols = opts.max_rank;
    Matrix U_LOCAL_MEM(U_LOCAL_nrows, U_LOCAL_ncols);
    descset_(U_LOCAL,
             &U_LOCAL_nrows, &U_LOCAL_ncols, &U_LOCAL_nrows, &U_LOCAL_ncols,
             &U_LOCAL_PROW, &U_LOCAL_PCOL, &U_LOCAL_CONTEXT, &U_LOCAL_nrows, &INFO);

    int IU = block * opts.max_rank * 2 + 1;
    int JU = 1;
    pdgemr2d_(&U_LOCAL_nrows, &U_LOCAL_ncols,
              UTRANSFER_MEM, &IU, &JU, UTRANSFER,
              &U_LOCAL_MEM, &ONE, &ONE, U_LOCAL,
              &BLACS_CONTEXT);

    if (mpi_rank(block) == MPIRANK) {
      if (row_has_admissible_blocks(A, block, level) && A.max_level != 1) {
        A.U.insert(block, level, std::move(U_LOCAL_MEM));
        // Init US from the row vector.
        Matrix US(U_LOCAL_ncols, U_LOCAL_ncols);
        for (int64_t i = 0; i < U_LOCAL_ncols; ++i) { US(i,i) = S_MEM[i]; }
        A.US.insert(block, level, std::move(US));
      }
      else {
        A.U.insert(block, level, generate_identity_matrix(block_nrows, rank));
        A.US.insert(block, level, generate_identity_matrix(rank, rank));
      }
    }

    A.ranks.insert(block, level, std::move(rank));

    delete[] S_MEM;
    if (IMAP[0] == MPIRANK) {
      Cblacs_gridexit(U_LOCAL_CONTEXT);
    }

    if (!MPIRANK) {
      std::cout << "\t BLOCK " << block << " TRANSFER MATRIX GENERATE." << std::endl;
    }
  }

  // 4. Generate the real basis at this level from the transfer matrices and the real basis one
  // level below.
  int U_REAL_local_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPISIZE);
  int U_REAL_local_ncols = fmax(numroc_(&rank, &rank, &MYCOL, &ZERO, &ONE), 1);
  int U_REAL[9];
  U_REAL_MEM = new double[(int64_t)U_REAL_local_nrows * (int64_t)U_REAL_local_ncols]();
  descinit_(U_REAL, &N, &rank, &nleaf, &rank, &ZERO, &ZERO, &BLACS_CONTEXT,
            &U_REAL_local_nrows, &INFO);

  for (int64_t block = 0; block < 1; ++block) {
    // Apply the child basis to the upper part of the transfer matrix to generate the
    // upper part of the real basis.
    int64_t c1 = block * 2;
    int64_t c2 = block * 2 + 1;

    // Compute upper part of the real basis for this level.
    {
      int IU = c1 * child_block_size + 1;
      int JU = 1;
      int IUTRANSFER = c1 * rank + 1;
      int JUTRANSFER = 1;
      int IU_REAL = c1 * child_block_size + 1;
      int JU_REAL = 1;

      pdgemm_(&NOTRANS, &NOTRANS,
              &child_block_size, &rank, &rank,
              &ALPHA,
              U_MEM, &IU, &JU, U,
              UTRANSFER_MEM, &IUTRANSFER, &JUTRANSFER, UTRANSFER,
              &BETA,
              U_REAL_MEM, &IU_REAL, &JU_REAL, U_REAL);
    }

    // Compute lower part of the real basis for this level.
    {
      int IU = c2 * child_block_size + 1;
      int JU = 1;
      int IUTRANSFER = c2 * rank + 1;
      int JUTRANSFER = 1;
      int IU_REAL = c2 * child_block_size + 1;
      int JU_REAL = 1;

      pdgemm_(&NOTRANS, &NOTRANS,
              &child_block_size, &rank, &rank,
              &ALPHA,
              U_MEM, &IU, &JU, U,
              UTRANSFER_MEM, &IUTRANSFER, &JUTRANSFER, UTRANSFER,
              &BETA,
              U_REAL_MEM, &IU_REAL, &JU_REAL, U_REAL);
    }
  }
  // Free the real basis of the child level and set the U_REAL to real basis.
  delete[] U_MEM;
  U_MEM = U_REAL_MEM;
  memcpy(U, U_REAL, sizeof(int) * 9);

  // Use the real basis for generation of S blocks.

  // Allocate a (nblocks * max_rank) ** 2 global matrix for temporary storage of the S blocks.
  int S_BLOCKS_nrows = nblocks * rank;
  int S_BLOCKS_local_nrows = fmax(numroc_(&S_BLOCKS_nrows, &rank, &MYROW, &ZERO, &MPISIZE), 1);
  int S_BLOCKS_local_ncols = fmax(numroc_(&S_BLOCKS_nrows, &rank, &MYCOL, &ZERO, &ONE), 1);
  int S_BLOCKS[9];
  double *S_BLOCKS_MEM = new double[(int64_t)S_BLOCKS_local_nrows * (int64_t)S_BLOCKS_local_ncols]();

  descinit_(S_BLOCKS, &S_BLOCKS_nrows, &S_BLOCKS_nrows, &rank, &rank,
            &ZERO, &ZERO, &BLACS_CONTEXT, &S_BLOCKS_local_nrows, &INFO);
  // multiply the 0th real basis

  // Allocate a temporary block for storing the intermediate result of the product of the
  // real basis and admissible dense matrix.
  int TEMP_PRODUCT_local_nrows = fmax(numroc_(&rank, &rank, &MYROW, &ZERO, &ONE), 1);
  int TEMP_PRODUCT_local_ncols = fmax(numroc_(&N, &nleaf, &MYCOL, &ZERO, &MPIGRID[1]), 1);
  int TEMP_PRODUCT[9];
  double *TEMP_PRODUCT_MEM =
    new double[(int64_t)TEMP_PRODUCT_local_nrows * (int64_t)TEMP_PRODUCT_local_ncols]();
  descinit_(TEMP_PRODUCT, &rank, &N, &rank, &nleaf,
            &ZERO, &ZERO, &BLACS_CONTEXT, &TEMP_PRODUCT_local_nrows, &INFO);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, level)) {
        // Multiply the real basis with the admissible block and store it in a temporary matrix.
        int IU = i * level_block_size + 1;
        int JU = 1;

        int IDENSE = i * level_block_size + 1;
        int JDENSE = j * level_block_size + 1;

        int ITEMP_PRODUCT = 1;
        int JTEMP_PRODUCT = j * level_block_size + 1;
        pdgemm_(&TRANS, &NOTRANS,
                &rank, &level_block_size, &level_block_size,
                &ALPHA,
                U_MEM, &IU, &JU, U,
                DENSE_MEM, &IDENSE, &JDENSE, DENSE.data(),
                &BETA,
                TEMP_PRODUCT_MEM, &ITEMP_PRODUCT, &JTEMP_PRODUCT, TEMP_PRODUCT);

        IU = j * level_block_size + 1;
        JU = 1;

        int IS_BLOCKS = i * rank + 1;
        int JS_BLOCKS = j * rank + 1;

        pdgemm_(&NOTRANS, &NOTRANS,
                &rank, &rank, &level_block_size,
                &ALPHA,
                TEMP_PRODUCT_MEM, &ITEMP_PRODUCT, &JTEMP_PRODUCT, TEMP_PRODUCT,
                U_MEM, &IU, &JU, U,
                &BETA,
                S_BLOCKS_MEM, &IS_BLOCKS, &JS_BLOCKS, S_BLOCKS);
      }
    }
  }

  if (!MPIRANK) {
    std::cout << "\tBEGIN S BLOCK " << level << std::endl;
  }

  // Copy the S blocks to the H2 matrix data structure.
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, level)) {
        // Init CBLACS info for the local S block.
        int S_LOCAL_CONTEXT;
        int IMAP[1];                // workspace to map the original grid.
        // local process grid parameters.
        int S_LOCAL_PNROWS, S_LOCAL_PNCOLS, S_LOCAL_PROW, S_LOCAL_PCOL;
        // specify the rank from the global grid for the local grid.
        IMAP[0] = mpi_rank(i);

        Cblacs_get(-1, 0, &S_LOCAL_CONTEXT);                    // init the new CBLACS context.
        Cblacs_gridmap(&S_LOCAL_CONTEXT, IMAP, ONE, ONE, ONE);  // init a 1x1 process grid.
        Cblacs_gridinfo(S_LOCAL_CONTEXT,                       // init grid params from the context.
                        &S_LOCAL_PNROWS, &S_LOCAL_PNCOLS, &S_LOCAL_PROW, &S_LOCAL_PCOL);

        int S_LOCAL[9];
        int S_LOCAL_nrows = opts.max_rank, S_LOCAL_ncols = opts.max_rank;
        Matrix S_LOCAL_MEM(S_LOCAL_nrows, S_LOCAL_ncols);

        // Store the info for the S block to be copied into in S_LOCAL.
        if (IMAP[0] != MPIRANK) {
          S_LOCAL_CONTEXT=-1;   // processes that are not part of this context set the local context to -1.
        }
        descset_(S_LOCAL,
                  &S_LOCAL_nrows, &S_LOCAL_ncols, &S_LOCAL_nrows, &S_LOCAL_ncols,
                  &S_LOCAL_PROW, &S_LOCAL_PCOL, &S_LOCAL_CONTEXT, &S_LOCAL_nrows, &INFO);

        int IS_BLOCKS = i * rank + 1;
        int JS_BLOCKS = j * rank + 1;
        pdgemr2d_(&S_LOCAL_nrows, &S_LOCAL_ncols,
                  S_BLOCKS_MEM, &IS_BLOCKS, &JS_BLOCKS, S_BLOCKS,
                  &S_LOCAL_MEM, &ONE, &ONE, S_LOCAL,
                  &BLACS_CONTEXT);

        if (mpi_rank(i) == MPIRANK) {
          A.S.insert(i, j, level, std::move(S_LOCAL_MEM));
          Cblacs_gridexit(S_LOCAL_CONTEXT);
        }
      }
    }
  }


  delete[] S_BLOCKS_MEM;
  delete[] TEMP_PRODUCT_MEM;
  delete[] UTRANSFER_MEM;
  delete[] TEMP_MEM;
  delete[] AY_MEM;
  if (!MPIRANK) {
    std::cout << "END TRANSFER MATRIX " << level << std::endl;
  }
}

void
construct_h2_matrix(SymmetricSharedBasisMatrix& A, const Domain& domain,
                        const Args& opts, double* DENSE_MEM, std::vector<int>& DENSE) {
  // init global U matrix
  int nleaf = opts.nleaf; int N = opts.N; int INFO;
  int U_nrows = numroc_(&N, &nleaf, &MYROW, &ZERO, &MPISIZE);
  int U_ncols = fmax(numroc_(&nleaf, &nleaf, &MYCOL, &ZERO, &ONE), 1);
  U_MEM = new double[(int64_t)U_nrows * (int64_t)U_ncols]();
  descinit_(U, &N, &nleaf, &nleaf, &nleaf, &ZERO, &ZERO, &BLACS_CONTEXT,
            &U_nrows, &INFO);

  generate_leaf_nodes(A, domain, opts, DENSE_MEM, DENSE);

  for (int64_t level = A.max_level-1; level >= A.min_level; --level) {
    generate_transfer_matrices(A, domain, opts, level, DENSE_MEM, DENSE);
  }

  delete[] U_MEM;

  // add a dummy level to facilitate easier interfacing with parsec.
  int64_t level = A.min_level-1;
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t c1 = block * 2;
    int64_t c2 = block * 2 + 1;
    int64_t rank = opts.max_rank;
    int64_t block_size = A.ranks(c1, child_level) + A.ranks(c2, child_level);

    if (mpi_rank(block) == MPIRANK) {
      A.U.insert(block, level, generate_identity_matrix(block_size, opts.max_rank));
      A.US.insert(block, level, generate_identity_matrix(opts.max_rank, opts.max_rank));
    }

    A.ranks.insert(block, level, std::move(rank));
  }
}
