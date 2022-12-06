#include "mpi.h"
#include <random>
#include <algorithm>
#include <set>
#include <cmath>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "globals.hpp"
#include "h2_dtd_construction.hpp"

// #include "mkl.h"
// #include "mkl_pblas.h"
#ifndef MKL_INT
#define MKL_INT int
#endif

using namespace Hatrix;


double *RAND_MEM = NULL, *AY_MEM = NULL; // local memory pointers.
int RAND[9], AY[9];          // scalapack descriptors.

int AY_local_rows, AY_local_cols;

int P;                          // columns of the random matrix.
int NBCOL;                      // col block size of the randomized matrix.

// temp data for storing actual basis temporarily during matrix construction.
std::vector<double*> Uchild_data;

static bool
row_has_admissible_blocks(const SymmetricSharedBasisMatrix& A, int64_t row,
                          int64_t level) {
  bool has_admis = false;

  for (int64_t i = 0; i < pow(2, level); ++i) {
    if (!A.is_admissible.exists(row, i, level) ||
        (A.is_admissible.exists(row, i, level) && A.is_admissible(row, i, level))) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

static void
dual_tree_traversal(SymmetricSharedBasisMatrix& A, const Cell& Ci,
                    const Cell& Cj, const Domain& domain,
                    const Args& opts) {
  int64_t i_level = Ci.level;
  int64_t j_level = Cj.level;

  bool well_separated = false;
  if (i_level == j_level) {
    double distance = 0;
    for (int64_t k = 0; k < opts.ndim; ++k) {
      distance += std::pow(Ci.center[k] - Cj.center[k], 2);
    }
    distance = sqrt(distance);

    if (distance * opts.admis > (Ci.radius + Cj.radius)) {
      // well-separated blocks.
      well_separated = true;
    }

    bool val = well_separated;
    A.is_admissible.insert(Ci.level_index, Cj.level_index, i_level, std::move(val));
  }

  if (i_level <= j_level && Ci.cells.size() > 0 && !well_separated) {
    // j is at a higher level and i is not leaf.
    dual_tree_traversal(A, Ci.cells[0], Cj, domain, opts);
    dual_tree_traversal(A, Ci.cells[1], Cj, domain, opts);
  }
  else if (j_level <= i_level && Cj.cells.size() > 0 && !well_separated) {
    // i is at a higheer level and j is not leaf.
    dual_tree_traversal(A, Ci, Cj.cells[0], domain, opts);
    dual_tree_traversal(A, Ci, Cj.cells[1], domain, opts);
  }
}

void
init_geometry_admis(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  A.max_level = domain.tree.height() - 1;
  dual_tree_traversal(A, domain.tree, domain.tree, domain, opts);
  A.min_level = 0;
  for (int64_t l = A.max_level; l > 0; --l) {
    int64_t nblocks = std::pow(2, l);
    bool all_dense = true;
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, l) && A.is_admissible(i, j, l)) {
          all_dense = false;
        }
      }
    }

    if (all_dense) {
      A.min_level = l;
      break;
    }
  }

  A.min_level++;
}

Matrix Ut_r;
// perform distributed pivoted QR on A and return the rank subject to accuracy
// on all processes.
int
pivoted_QR(double* A, int M, int N,
           int local_rows_A, int local_cols_A,
           int IA, int JA, int MB, int NB,
           int* DESCA, double accuracy) {
  std::vector<int> IPIV(local_cols_A);
  std::vector<double> TAU(local_cols_A);
  std::vector<double> WORK(1);

  pdgeqpf_(&M, &N,
           A, &IA, &JA, DESCA,
           IPIV.data(), TAU.data(), WORK.data(),
           &MINUS_ONE, &info); // workspace query
  int LWORK = WORK[0];
  WORK.resize(LWORK);

  pdgeqpf_(&M, &N,
           A, &IA, &JA, DESCA,
           IPIV.data(), TAU.data(), WORK.data(),
           &LWORK, &info); // pivoted QR

  // Determine diagaonal values of the QR factorized matrix.
  std::vector<double> RANKVECTOR(local_cols_A);
  int min_MN = std::min(M, N);

  int diagonals = 0;
  for (int i = 0; i < min_MN; ++i) {
    int g_row = IA + i;
    int g_col = JA + i;
    int row_proc = indxg2p(g_row, MB, 0, MPIGRID[0]);
    int col_proc = indxg2p(g_col, NB, 0, MPIGRID[1]);

    if (row_proc == MYROW && col_proc == MYCOL) {
      int lrow = indxg2l(g_row, MB,  MPIGRID[0]) - 1;
      int lcol = indxg2l(g_col, NB, MPIGRID[1]) - 1;

      RANKVECTOR[diagonals++] = abs(A[lrow + lcol * local_rows_A]);
    }
  }

  std::vector<int> DIAGONAL_COUNTS(MPISIZE);
  MPI_Allgather((void*)&diagonals, 1, MPI_INT,
                (void*)DIAGONAL_COUNTS.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  int total_count = 0;
  std::vector<int> DIAGONAL_DISPLACEMENTS(MPISIZE+1, 0);
  for (int i = 0; i < MPISIZE; ++i) {
    total_count += DIAGONAL_COUNTS[i];
    DIAGONAL_DISPLACEMENTS[i+1] = total_count;
  }

  std::vector<double> GATHER_RANKVECTOR(total_count);
  MPI_Allgatherv((void*)RANKVECTOR.data(), DIAGONAL_COUNTS[MPIRANK], MPI_DOUBLE,
                 (void*)GATHER_RANKVECTOR.data(), DIAGONAL_COUNTS.data(),
                 DIAGONAL_DISPLACEMENTS.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  std::sort(GATHER_RANKVECTOR.begin(), GATHER_RANKVECTOR.end(),
            std::greater<double>());

  int rank=0;
  for (double eig : GATHER_RANKVECTOR) {
    if (eig <= accuracy) { break; }
    rank++;
  }

  // obtain orthogonal factors
  pdorgqr_(&M, &min_MN, &min_MN,
           A, &IA, &JA, DESCA,
           TAU.data(), WORK.data(),
           &MINUS_ONE, &info); // workspace query

  LWORK = WORK[0];
  WORK.resize(LWORK);
  pdorgqr_(&M, &min_MN, &min_MN,
           A, &IA, &JA, DESCA,
           TAU.data(), WORK.data(),
           &LWORK, &info); // compute Q factors

  return rank;
}

void
generate_leaf_nodes(SymmetricSharedBasisMatrix& A,
                    const Hatrix::Domain& domain,
                    const Hatrix::Args& opts) {
  const int nblocks = std::pow(2, A.max_level);

  // Generate the dense blocks at the leaf node.
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        if (mpi_rank(i, j) == MPIRANK) {
          // regenerate the dense block to avoid communication.
          Matrix Aij = generate_p2p_interactions(domain,
                                                 i, j, A.max_level,
                                                 opts.kernel);
          A.D.insert(i, j, A.max_level, std::move(Aij));
        }
      }
    }
  }

  // randomize inadmissible blocks of the large matrix
  int info;
  int N = opts.N;

  for (int block = 0; block < nblocks; ++block) {
    MKL_INT block_size = domain.cell_size(block, A.max_level);

    for (int j = 0; j < nblocks; ++j) {
      if (exists_and_inadmissible(A, block, j, A.max_level)) { continue; }

      double ALPHA = 1.0;
      double BETA = 1.0;
      MKL_INT IA = block_size * block + 1;
      MKL_INT JA = block_size * j + 1;

      MKL_INT IB = block_size * j + 1;
      MKL_INT JB = 1;

      MKL_INT IC = block_size * block + 1;
      MKL_INT JC = 1;

      pdgemm_(&NOTRANS, &NOTRANS, &block_size, &P, &block_size,
              &ALPHA,
              DENSE_MEM, &IA, &JA, DENSE.data(),
              RAND_MEM, &IB, &JB, RAND,
              &BETA,
              AY_MEM, &IC, &JC, AY);
    }
  }

  // generate column bases from the randomized blocks.
  for (int block = 0; block < nblocks; ++block) {
    int block_size = domain.cell_size(block, A.max_level);

    int IA = block_size * block + 1;
    int JA = 1;

    // Length of ipiv should be LOCc(JA+N-1) = AY_local_col.
    std::vector<int> IPIV(AY_local_cols);
    // Length of TAU is LOCc(JA+MIN(M,N)-1) = AY_local_cols.
    std::vector<double> TAU(AY_local_cols);
    std::vector<double> WORK(1);
    int info;

    pdgeqpf_(&block_size, &P,
             AY_MEM, &IA, &JA, AY,
             IPIV.data(), TAU.data(), WORK.data(),
             &MINUS_ONE, &info); // workspace query

    int LWORK = WORK[0];

    WORK.resize(LWORK);
    pdgeqpf_(&block_size, &P,
             AY_MEM, &IA, &JA, AY,
             IPIV.data(), TAU.data(), WORK.data(),
             &LWORK, &info);    // distributed pivoted QR

    std::vector<double> RANKVECTOR(AY_local_cols, 0);
    int diagonals = 0;
    for (int i = 0; i < P; ++i) {
      int g_row = block * block_size + i + 1;
      int g_col = i + 1;
      int row_proc = indxg2p(g_row, DENSE_NBROW, 0, MPIGRID[0]);
      int col_proc = indxg2p(g_col, NBCOL, 0, MPIGRID[1]);

      if (row_proc == MYROW && col_proc == MYCOL) {
        int lrow = indxg2l(g_row, DENSE_NBROW,  MPIGRID[0]) - 1;
        int lcol = indxg2l(g_col, NBCOL, MPIGRID[1]) - 1;

        RANKVECTOR[diagonals++] = abs(AY_MEM[lrow + lcol * AY_local_rows]);
      }
    }

    std::vector<int> DIAGONAL_COUNTS(MPISIZE, 0);
    MPI_Allgather((void*)&diagonals, 1, MPI_INT,
                  (void*)DIAGONAL_COUNTS.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    int total_count = 0;
    std::vector<int> DIAGONAL_DISPLACEMENTS(MPISIZE+1, 0);
    for (int i = 0; i < MPISIZE; ++i) {
      total_count += DIAGONAL_COUNTS[i];
      DIAGONAL_DISPLACEMENTS[i+1] = total_count;
    }

    std::vector<double> GATHER_RANKVECTOR(total_count);
    MPI_Allgatherv((void*)RANKVECTOR.data(), DIAGONAL_COUNTS[MPIRANK], MPI_DOUBLE,
                   (void*)GATHER_RANKVECTOR.data(), DIAGONAL_COUNTS.data(),
                   DIAGONAL_DISPLACEMENTS.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    std::sort(GATHER_RANKVECTOR.begin(), GATHER_RANKVECTOR.end(),
              std::greater<double>());

    int rank=0;
    for (double eig : GATHER_RANKVECTOR) {
      if (eig <= opts.accuracy) { break; }
      rank++;
    }

    // obtain orthogonal factors
    pdorgqr_(&block_size, &P, &P,
             AY_MEM, &IA, &JA, AY,
             TAU.data(), WORK.data(),
             &MINUS_ONE, &info); // workspace query

    LWORK = WORK[0];
    WORK.resize(LWORK);
    pdorgqr_(&block_size, &P, &P,
             AY_MEM, &IA, &JA, AY,
             TAU.data(), WORK.data(),
             &LWORK, &info); // compute Q factors

    // copy the computed orthogonal factors into U blocks.
    int IMAP[1];                // workspace for mapping the grid to processes.
    IMAP[0] = mpi_rank(block);
    int NODE_CONTEXT;

    Cblacs_get(BLACS_CONTEXT, 10, &NODE_CONTEXT);
    Cblacs_gridmap(&NODE_CONTEXT, IMAP, ONE, ONE, ONE);

    int U_block[9];
    int U_nrows = 0, U_ncols = 0;
    if (mpi_rank(block) == MPIRANK) {
      U_nrows = block_size;
      U_ncols = rank;
    }
    Matrix U(U_nrows, U_ncols);

    int LOCAL_NROWS, LOCAL_NCOLS, LOCAL_ROW, LOCAL_COL;
    Cblacs_gridinfo(NODE_CONTEXT, &LOCAL_NROWS, &LOCAL_NCOLS, &LOCAL_ROW, &LOCAL_COL);
    descset_(U_block, &block_size, &rank, &block_size, &rank,
              &LOCAL_ROW, &LOCAL_COL, &NODE_CONTEXT, &block_size, &info);

    // copy out the U block into its own dedicated storage.
    pdgemr2d_(&block_size, &rank,
              AY_MEM, &IA, &JA, AY,
              &U, &ONE, &ONE, U_block,
              &BLACS_CONTEXT);

    if (mpi_rank(block) == MPIRANK) {
      A.U.insert(block, A.max_level,
                 std::move(U));
    }

    // ranks are stored in all processes.
    A.ranks.insert(block, A.max_level, std::move(rank));
  }

  std::vector<int> comm_world_ranks(MPISIZE);
  comm_world_ranks[0] = 0;
  std::iota(comm_world_ranks.begin()+1, comm_world_ranks.end(), 1);

  MPI_Group WORLD_GROUP, ROW_GROUP;

  // generate the S blocks using P2P communication
  for (int i = 0; i < nblocks; ++i) {
    int i_block_size = domain.cell_size(i, A.max_level);
    int i_proc = mpi_rank(i);

    MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);

    std::set<int> bcast_ranks;
    bcast_ranks.insert(i_proc);

    for (int j = 0; j < i; ++j) {
      int j_block_size = domain.cell_size(j, A.max_level);
      int j_proc = mpi_rank(j);
      int dest_proc = mpi_rank(i, j);

      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        bcast_ranks.insert(dest_proc);
      }
    }

    int Ui_nrows = i_block_size;
    int Ui_ncols = A.ranks(i, A.max_level);
    Matrix Ui(Ui_nrows, Ui_ncols);

    {
      std::vector<int> bcast_ranks_vector(bcast_ranks.begin(),
                                          bcast_ranks.end());
      MPI_Group_incl(WORLD_GROUP, bcast_ranks_vector.size(),
                     bcast_ranks_vector.data(), &ROW_GROUP);


      MPI_Comm MPI_S_ROW_COMM;
      MPI_Comm_create_group(MPI_COMM_WORLD, ROW_GROUP, 0,
                            &MPI_S_ROW_COMM);

      std::vector<int> row_comm_ranks(comm_world_ranks.size());
      MPI_Group_translate_ranks(WORLD_GROUP, comm_world_ranks.size(),
                                comm_world_ranks.data(),
                                ROW_GROUP, row_comm_ranks.data());

      // perform Bcast of the U block on row i along the processes
      // that posses an 'S' block.
      if (mpi_rank(i) == MPIRANK) {
        // copy into Ui if this is the rank on which it exists.
        Ui = A.U(i, A.max_level) ;
      }

      if (MPI_COMM_NULL != MPI_S_ROW_COMM) {
        MPI_Bcast(&Ui, Ui_nrows * Ui_ncols, MPI_DOUBLE,
                  row_comm_ranks[i_proc], MPI_S_ROW_COMM);
      }
    }


    for (int j = 0; j < i; ++j) {
      int j_block_size = domain.cell_size(j, A.max_level);
      int j_proc = mpi_rank(j);
      int dest_proc = mpi_rank(i, j);

      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        int Uj_nrows = domain.cell_size(j, A.max_level);
        int Uj_ncols = A.ranks(j, A.max_level);
        MPI_Request request;

        if (j_proc == MPIRANK) {
          MPI_Isend(&A.U(j, A.max_level), Uj_nrows * Uj_ncols, MPI_DOUBLE, dest_proc,
                    j, MPI_COMM_WORLD, &request);
        }

        if (dest_proc == MPIRANK) {
          Matrix Uj(Uj_nrows, Uj_ncols);
          MPI_Status status;

          MPI_Irecv(&Uj, Uj_nrows * Uj_ncols, MPI_DOUBLE, j_proc, j,
                    MPI_COMM_WORLD, &request);
          MPI_Wait(&request, &status);

          Matrix Aij = generate_p2p_interactions(domain,
                                                 i, j, A.max_level, opts.kernel);

          Matrix S_block = matmul(matmul(Ui, Aij, true, false), Uj);
          A.S.insert(i, j, A.max_level, std::move(S_block));
        }
      }
    }
  }
}

std::tuple<std::vector<int>, std::vector<double>>
generate_transfer_matrices(SymmetricSharedBasisMatrix& A,
                           int level,
                           std::vector<int>& Uchild,
                           std::vector<double>& Uchild_mem,
                           const Hatrix::Domain& domain,
                           const Hatrix::Args& opts) {
  int nblocks = std::pow(2, level);
  int child_level = level + 1;

  // generate randomzed admissible blocks for this level.
  for (int block = 0; block < nblocks; ++block) {
    int block_size = domain.cell_size(block, level);
    for (int j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(block, j, level) &&
          !A.is_admissible(block, j, level)) {  continue; }

      double ALPHA = 1.0;
      double BETA = 1.0;

      int IA = block_size * block + 1;
      int JA = block_size * j + 1;
      int IB = block_size * j + 1;
      int JB = 1;
      int IC = block_size * block + 1;
      int JC = 1;

      pdgemm_(&NOTRANS, &NOTRANS, &block_size, &P, &block_size,
              &ALPHA,
              DENSE_MEM, &IA, &JA, DENSE.data(),
              RAND_MEM, &IB, &JB, RAND,
              &BETA,
              AY_MEM, &IC, &JC, AY);
    }
  }

  // setup the full Utransfer matrix on distributed memory. Utransfer is a large matrix
  // of size rank_total * P. This matrix can store the transfer matrices for the entire
  // level of this H2-matrix.
  std::vector<int> Utransfer(9);
  int rank_total = 0;
  int child_nblocks = std::pow(2, child_level);
  for (int i = 0; i < child_nblocks; ++i) {
    rank_total += A.ranks(i, child_level);
  }
  int MB = rank_total / child_nblocks;
  int Utransfer_local_rows = numroc_(&rank_total, &MB, &MYROW, &ZERO, &MPIGRID[0]);
  int Utransfer_local_cols = numroc_(&P, &NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);
  std::vector<double> Utransfer_mem(Utransfer_local_rows * Utransfer_local_cols);

  descinit_(Utransfer.data(), &rank_total, &P, &MB, &NBCOL, &ZERO, &ZERO,
            &BLACS_CONTEXT, &Utransfer_local_rows, &info);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  // Generation of the transfer matrices.
  for (int block = 0; block < nblocks; ++block) {
    const int c1 = block * 2;
    const int c2 = block * 2 + 1;

    int block_size_c1 = domain.cell_size(c1, child_level);
    int block_size_c2 = domain.cell_size(c2, child_level);

    int rank_c1 = A.ranks(c1, child_level);
    int rank_c2 = A.ranks(c2, child_level);

    if (row_has_admissible_blocks(A, block, level) && A.max_level != 1) {
      double ALPHA = 1.0;
      double BETA = 0.0;

      int IA = c1 * block_size_c1 + 1;
      int JA = 1;

      int IB = c1 * block_size_c1 + 1;
      int JB = 1;

      int IC = 1;
      for (int i = 0; i < c1; ++i) {
        IC += A.ranks(i, child_level);
      }
      int JC = 1;

      // multiply the child level full basis with the randomized matrix
      // to project the child basis onto the randomized data for computing
      // the nested basis.

      // multiply the first child Ubig with the upper slice of AY.
      pdgemm_(&TRANS, &NOTRANS,
              &rank_c1, &P, &block_size_c1,
              &ALPHA,
              Uchild_mem.data(), &IA, &JA, Uchild.data(),
              AY_MEM, &IB, &JB, AY,
              &BETA,
              Utransfer_mem.data(), &IC, &JC, Utransfer.data());

      IA = c2 * block_size_c2 + 1;
      IB = c2 * block_size_c2 + 1;
      IC += rank_c1;

      // Multiply the second child block with the lower slice of this part
      // of the randomized matrix.
      pdgemm_(&TRANS, &NOTRANS, &rank_c2, &P, &block_size_c2,
              &ALPHA,
              Uchild_mem.data(), &IA, &JA, Uchild.data(),
              AY_MEM, &IB, &JB, AY,
              &BETA,
              Utransfer_mem.data(), &IC, &JC, Utransfer.data());

      // pivoted QR on the transfer matrix.
      int Utransfer_nrows = rank_c1 + rank_c2;
      IA = 1;
      for (int i = 0; i < block; ++i) {
        int c1_i = i * 2;
        int c2_i = i * 2 + 1;
        IA += A.ranks(c1_i, child_level) + A.ranks(c2_i, child_level);
      }

      int qr_rank = pivoted_QR(Utransfer_mem.data(),
                               Utransfer_nrows, P,
                               Utransfer_local_rows, Utransfer_local_cols,
                               IA, ONE,
                               MB, NBCOL,
                               Utransfer.data(), opts.accuracy);

      // copy the transfer matrix into the RowLevel map for use with parsec.
      int IMAP[1];
      IMAP[0] = mpi_rank(block);

      int NODE_CONTEXT;
      Cblacs_get(BLACS_CONTEXT, 10, &NODE_CONTEXT);
      Cblacs_gridmap(&NODE_CONTEXT, IMAP, ONE, ONE, ONE);

      std::vector<int> U_block(DESC_LEN);
      int U_nrows = 0, U_ncols = 0;
      if (mpi_rank(block) == MPIRANK) {
        U_nrows = Utransfer_nrows;
        U_ncols = qr_rank;
      }
      Matrix U(U_nrows, U_ncols);

      int LOCAL_NROWS, LOCAL_NCOLS, LOCAL_ROW, LOCAL_COL;
      Cblacs_gridinfo(NODE_CONTEXT, &LOCAL_NROWS, &LOCAL_NCOLS, &LOCAL_ROW, &LOCAL_COL);
      descset_(U_block.data(), &Utransfer_nrows, &qr_rank, &Utransfer_nrows, &qr_rank,
               &LOCAL_ROW, &LOCAL_COL, &NODE_CONTEXT, &Utransfer_nrows, &info);

      IA = 1; for (int i = 0; i < c1; ++i) { IA += A.ranks(i, child_level); }
      JA = 1;

      // copy out the U block into its own dedicated storage.
      pdgemr2d_(&Utransfer_nrows, &qr_rank,
                Utransfer_mem.data(), &IA, &JA, Utransfer.data(),
                &U, &ONE, &ONE, U_block.data(),
                &BLACS_CONTEXT);

      if (mpi_rank(block) == MPIRANK) {
        A.U.insert(block, level, std::move(U));
      }

      A.ranks.insert(block, level, std::move(qr_rank));
    } // if row_has_admissible_blocks
    else {
      int rank = std::max(A.ranks(c1, child_level), A.ranks(c2, child_level));
      if (mpi_rank(block) == MPIRANK) {
        A.U.insert(block, level, generate_identity_matrix(A.ranks(c1, child_level) + A.ranks(c2, child_level),
                                                          rank));
      }
      A.ranks.insert(block, level, std::move(rank));
    }
  }

  // compute dimensions for Ubig at this level.
  int max_rank = 0;
  for (int i = 0; i < nblocks; ++i) { max_rank = std::max<int>(max_rank, A.ranks(i, level)); }

  std::vector<int> Ubig(9);
  int Ubig_NBCOL = ceil(max_rank / double(MPIGRID[1]));
  int Ubig_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int Ubig_local_cols = numroc_(&max_rank, &Ubig_NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);
  std::vector<double> Ubig_mem(Ubig_local_rows * Ubig_local_cols);

  descinit_(Ubig.data(), &N, &max_rank, &DENSE_NBROW, &Ubig_NBCOL, &ZERO, &ZERO,
            &BLACS_CONTEXT, &Ubig_local_rows, &info);


  // multiply the real bases at the child level with the transfer matrix in Utransfer to generate
  // the full bases at this level.
  for (int block = 0; block < nblocks; ++block) {
    int block_size = domain.cell_size(block, level);

    int c1 = block * 2;
    int c2 = block * 2 + 1;

    MKL_INT block_size_c1 = domain.cell_size(c1, child_level);
    int block_size_c2 = domain.cell_size(c2, child_level);

    MKL_INT rank_c1 = A.ranks(c1, child_level);
    int rank_c2 = A.ranks(c2, child_level);

    MKL_INT Utransfer_ncols = A.ranks(block, level);

    double ALPHA = 1.0;
    double BETA = 0.0;

    MKL_INT IA = block_size_c1 * c1 + 1;
    MKL_INT JA = 1;

    MKL_INT IB = 1; for (int i = 0; i < c1; ++i) { IB += A.ranks(i, child_level); }
    MKL_INT JB = 1;

    MKL_INT IC = block * block_size + 1;
    MKL_INT JC = 1;

    // multiply Ubig c1  with the upper slice of the transfer matrix.
    pdgemm_(&NOTRANS, &NOTRANS,
            &block_size_c1, &Utransfer_ncols, &rank_c1,
            &ALPHA,
            Uchild_mem.data(), &IA, &JA, Uchild.data(),
            Utransfer_mem.data(), &IB, &JB, Utransfer.data(),
            &BETA,
            Ubig_mem.data(), &IC, &JC, Ubig.data());

    IA += block_size_c1;
    IB += rank_c1;
    IC += block_size_c1;

    // multiply Ubig c2 with the lower slice of the transfer matrix.
    pdgemm_(&NOTRANS, &NOTRANS,
            &block_size_c2, &Utransfer_ncols, &rank_c2,
            &ALPHA,
            Uchild_mem.data(), &IA, &JA, Uchild.data(),
            Utransfer_mem.data(), &IB, &JB, Utransfer.data(),
            &BETA,
            Ubig_mem.data(), &IC, &JC, Ubig.data());
  }

  // Generate S blocks at this level. This uses the temporary Ubig matrices so we
  // use a max_rank * N sized temporary distributed buffer for storing the intermediate
  // results of the product en route the S block.
  std::vector<int> Stemp(DESC_LEN);
  int Stemp_NBROW = ceil(double(max_rank) / MPIGRID[0]);
  int Stemp_local_rows = numroc_(&max_rank, &Stemp_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int Stemp_local_cols = numroc_(&N, &DENSE_NBROW, &MYCOL, &ZERO, &MPIGRID[1]);
  std::vector<double> Stemp_mem(Stemp_local_rows * Stemp_local_cols);

  if (Stemp_local_rows == 0) { Stemp_local_rows = 1; }

  descinit_(Stemp.data(), &max_rank, &N, &Stemp_NBROW, &DENSE_NBROW,
            &ZERO, &ZERO, &BLACS_CONTEXT, &Stemp_local_rows, &info);

  // Rather clumsy way of calculating the S blocks by making a rank_nrows * rank_ncols
  // sized distributed matrix that holds all the S blocks. Use scalapack for the compute
  // and then redistribute into the respective A.S maps later.
  std::vector<int> Sblock(DESC_LEN);
  int rank_nrows = 0;
  for (int i = 0; i < nblocks; ++i) { rank_nrows += A.ranks(i, level); }
  int Sblock_NBROW = ceil(rank_nrows / double(MPIGRID[0]));
  int Sblock_NBCOL = ceil(rank_nrows / double(MPIGRID[1]));
  int Sblock_local_rows = numroc_(&rank_nrows, &Sblock_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int Sblock_local_cols = numroc_(&rank_nrows, &Sblock_NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);
  std::vector<double> Sblock_mem(Sblock_local_rows * Sblock_local_cols);

  descinit_(Sblock.data(), &rank_nrows, &rank_nrows, &Sblock_NBROW,
            &Sblock_NBCOL, &ZERO, &ZERO, &BLACS_CONTEXT, &Sblock_local_rows, &info);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) &&
          A.is_admissible(i, j, level)) {
        int row_block_size = domain.cell_size(i, level);
        int col_block_size = domain.cell_size(j, level);
        int row_rank = A.ranks(i, level);
        int col_rank = A.ranks(j, level);

        double ALPHA = 1.0;
        double BETA = 0.0;

        int IA = row_block_size * i + 1;
        int JA = 1;

        int IB = row_block_size * i + 1;
        int JB = col_block_size * j + 1;

        int IC = JA;
        int JC = IA;

        pdgemm_(&TRANS, &NOTRANS, // partial product.
                &row_rank, &row_block_size, &col_block_size,
                &ALPHA,
                Ubig_mem.data(), &IA, &JA, Ubig.data(),
                DENSE_MEM, &IB, &JB, DENSE.data(),
                &BETA,
                Stemp_mem.data(), &IC, &JC, Stemp.data());

        IA = IC;
        JA = JC;

        IB = row_block_size * j + 1;
        JB = 1;

        IC = 1; for (int ii=0; ii<i; ++ii) {IC += A.ranks(ii, level);}
        JC = 1; for (int jj=0; jj<j; ++jj) {JC += A.ranks(jj, level);}

        pdgemm_(&NOTRANS, &NOTRANS, // compute S block.
                &row_rank, &col_rank, &col_block_size,
                &ALPHA,
                Stemp_mem.data(), &IA, &JA, Stemp.data(),
                Ubig_mem.data(), &IB, &JB, Ubig.data(),
                &BETA,
                Sblock_mem.data(), &IC, &JC, Sblock.data());

        // copy the S block into the map.
        int IMAP[1];
        IMAP[0] = mpi_rank(i, j);

        int NODE_CONTEXT;
        Cblacs_get(BLACS_CONTEXT, 10, &NODE_CONTEXT);
        Cblacs_gridmap(&NODE_CONTEXT, IMAP, ONE, ONE, ONE);

        std::vector<int> DESCS(DESC_LEN);
        int S_nrows = 0, S_ncols = 0;
        if (mpi_rank(i, j) == MPIRANK) {
          S_nrows = row_rank;
          S_ncols = col_rank;
        }
        Matrix S(S_nrows, S_ncols);

        int LOCAL_NROWS, LOCAL_NCOLS, LOCAL_ROW, LOCAL_COL;
        Cblacs_gridinfo(NODE_CONTEXT, &LOCAL_NROWS, &LOCAL_NCOLS, &LOCAL_ROW, &LOCAL_COL);

        descset_(DESCS.data(), &S_nrows, &S_ncols, &S_nrows, &S_ncols,
                 &LOCAL_ROW, &LOCAL_COL, &NODE_CONTEXT, &S_nrows, &info);

        IA = IC;
        JA = JC;

        pdgemr2d_(&row_rank, &col_rank,
                  Sblock_mem.data(), &IA, &JA, Sblock.data(),
                  &S, &ONE, &ONE, DESCS.data(),
                  &BLACS_CONTEXT);

        if (mpi_rank(i, j) == MPIRANK) {
          A.S.insert(i, j, level, std::move(S));
        }
      }
    }
  }

  // copy the Ubig computed at this level into the Uchild and Uchild_mem for use
  // at the next (upper) level.
  return std::make_tuple(std::move(Ubig), std::move(Ubig_mem));
}

void
construct_h2_matrix_dtd(SymmetricSharedBasisMatrix& A,
                        const Hatrix::Domain& domain,
                        const Hatrix::Args& opts) {
  P = opts.max_rank;

  NBCOL = P / MPIGRID[1];

  int RAND_NB_COLS = P / MPIGRID[1];

  int RAND_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int RAND_local_cols = numroc_(&P, &RAND_NB_COLS, &MYCOL, &ZERO, &MPIGRID[1]);

  AY_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  AY_local_cols = numroc_(&P, &NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);

  descinit_(RAND, &N, &P, &DENSE_NBROW, &RAND_NB_COLS, &ZERO, &ZERO,
            &BLACS_CONTEXT, &RAND_local_rows, &info);
  descinit_(AY, &N, &P, &DENSE_NBROW, &NBCOL, &ZERO, &ZERO,
            &BLACS_CONTEXT, &AY_local_rows, &info);

  RAND_MEM = (double*)calloc(RAND_local_rows * RAND_local_cols,
                             sizeof(double));
  AY_MEM = (double*)calloc(AY_local_rows * AY_local_cols,
                           sizeof(double));

  // seed with the rank to prevent the same sequence on every proc.
  std::mt19937 gen(MPIRANK);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp parallel for
  for (int i = 0; i < RAND_local_rows; ++i) {
#pragma omp parallel for
    for (int j = 0; j < RAND_local_cols; ++j) {
      RAND_MEM[i + j * RAND_local_rows] = dist(gen);
    }
  }

  generate_leaf_nodes(A, domain, opts);

  // temporary storage for actual basis matrices during construction.
  int nblocks = std::pow(2, A.max_level);
  std::vector<int> Uchild;
  Uchild.resize(9);
  int max_rank = 0;
  for (int i = 0; i < nblocks; ++i) {
    max_rank = std::max<int>(max_rank, A.ranks(i, A.max_level));
  }
  int Uchild_NBCOL = ceil(max_rank / double(MPIGRID[1]));
  int Uchild_local_rows = numroc_(&N, &DENSE_NBROW, &MYROW, &ZERO, &MPIGRID[0]);
  int Uchild_local_cols = numroc_(&max_rank, &Uchild_NBCOL, &MYCOL, &ZERO, &MPIGRID[1]);

  std::vector<double> Uchild_mem(Uchild_local_rows * Uchild_local_cols);

  descinit_(Uchild.data(), &N, &max_rank, &DENSE_NBROW, &Uchild_NBCOL, &ZERO, &ZERO,
            &BLACS_CONTEXT, &Uchild_local_rows, &info);

  // copy AY into Uchild for use in transfer matrix generation.
  for (int block = 0; block < nblocks; ++block) {
    int block_size = domain.cell_size(block, A.max_level);
    int rank = A.ranks(block, A.max_level);

    int IA = block * block_size + 1;
    int JA = 1;
    int IB = block * block_size + 1;
    int JB = 1;

    pdgemr2d_(&block_size, &rank,
              AY_MEM, &IA, &JA, AY,
              Uchild_mem.data(), &IB, &JB, Uchild.data(),
              &BLACS_CONTEXT);
  }

  for (int level = A.max_level-1; level >= A.min_level-1; --level) {
    // init scratch product memory space to 0.
    for (int i = 0; i < AY_local_rows * AY_local_cols; ++i) { AY_MEM[i] = 0; }
    std::tie(Uchild, Uchild_mem) = generate_transfer_matrices(A, level,
                                                              Uchild, Uchild_mem, domain, opts);
  }

  free(RAND_MEM);
  free(AY_MEM);
}
