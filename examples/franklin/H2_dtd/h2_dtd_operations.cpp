#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "globals.hpp"
#include "h2_dtd_operations.hpp"

using namespace Hatrix;

static void
multiply_S(SymmetricSharedBasisMatrix& A,
           std::vector<Matrix>& x_hat,
           std::vector<Matrix>& b_hat,
           int x_hat_offset, int b_hat_offset, int level) {
  int nblocks = pow(2, level);

  for (int i = 0; i < nblocks; ++i) {
    int proc_i = mpi_rank(i);
    int S_nrows = A.ranks(i, level);

    for (int j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) &&
          A.is_admissible(i, j, level)) {
        int proc_j = mpi_rank(j);
        int proc_S = mpi_rank(i, j);
        int S_ncols = A.ranks(j, level);
        MPI_Request i_request, j_request;

        int S_tag = i + j * nblocks;
        if (proc_S == MPIRANK) {
          Matrix& Sblock = A.S(i, j, level);

          MPI_Isend(&Sblock, S_nrows * S_ncols, MPI_DOUBLE, proc_i,
                    S_tag, MPI_COMM_WORLD, &i_request);
          MPI_Isend(&Sblock, S_nrows * S_ncols, MPI_DOUBLE, proc_j,
                    S_tag, MPI_COMM_WORLD, &j_request);
        }

        int x_hat_i_tag = i + S_tag + 1;
        if (proc_i == MPIRANK) {
          // send x_hat[i] where b_hat[j] exists.
          int x_hat_index = x_hat_offset + i / MPISIZE;
          MPI_Isend(&x_hat[x_hat_index], x_hat[x_hat_index].numel(), MPI_DOUBLE,
                    proc_j, x_hat_i_tag, MPI_COMM_WORLD, &j_request);

        }

        int x_hat_j_tag = j + S_tag + 1;
        if (proc_j == MPIRANK) {
          // send x_hat(j) where b_hat(i) exists.
          int x_hat_index = x_hat_offset + j / MPISIZE;
          MPI_Isend(&x_hat[x_hat_index], x_hat[x_hat_index].numel(), MPI_DOUBLE,
                    proc_i, x_hat_j_tag, MPI_COMM_WORLD, &i_request);

        }
      }
    }

    // receive and perform operations
    for (int j = 0; j < i; ++j) {
      int S_ncols = A.ranks(j, level);

      if (A.is_admissible.exists(i, j, level) &&
          A.is_admissible(i, j, level)) {
        int S_tag = i + j * nblocks;
        int x_hat_j_tag = j + S_tag + 1;
        int x_hat_i_tag = i + S_tag + 1;
        int proc_j = mpi_rank(j);
        int proc_S = mpi_rank(i, j);

        if (proc_i == MPIRANK) {
          // receive for b_hat[i]
          Matrix Sij(S_nrows, S_ncols);
          Matrix x_hat_j(S_ncols, 1);
          MPI_Status status;

          MPI_Recv(&Sij, S_nrows * S_ncols, MPI_DOUBLE, proc_S, S_tag,
                   MPI_COMM_WORLD, &status);
          MPI_Recv(&x_hat_j, S_ncols, MPI_DOUBLE, proc_j, x_hat_j_tag,
                   MPI_COMM_WORLD, &status);

          int index = b_hat_offset + i / MPISIZE;
          assert(index < b_hat.size());
          matmul(Sij, x_hat_j, b_hat[index]);
        }

        if (proc_j == MPIRANK) {
          // receive for b_hat[j]
          Matrix Sij(S_nrows, S_ncols);
          Matrix x_hat_i(S_nrows, 1);
          MPI_Status status;
          MPI_Recv(&Sij, S_nrows * S_ncols, MPI_DOUBLE, proc_S, S_tag,
                   MPI_COMM_WORLD, &status);
          MPI_Recv(&x_hat_i, S_nrows, MPI_DOUBLE, proc_i, x_hat_i_tag,
                   MPI_COMM_WORLD, &status);

          int index = b_hat_offset + j / MPISIZE;
          assert(index < b_hat.size());
          matmul(Sij, x_hat_i, b_hat[index], true, false);
        }
      }
    }
  }
}

// Note: This function assumes that the vector is distributed vertically
// in a cyclic process layout.
void
matmul(SymmetricSharedBasisMatrix& A,
       const Domain& domain,
       std::vector<Matrix>& x,
       std::vector<Matrix>& b) {
  int leaf_nblocks = pow(2, A.max_level);
  int nblocks_per_proc = ceil(leaf_nblocks / double(MPISIZE));
  std::vector<Matrix> x_hat;

  // Apply V leaf basis nodes.
  for (int i = 0; i < nblocks_per_proc; i += 1) {
    int index = i * MPISIZE + MPIRANK;
    x_hat.push_back(matmul(A.U(index, A.max_level), x[i], true, false, 1.0));
  }

  // Apply V transfer nodes.
  int x_hat_offset = 0;         // x_hat_offset within the same process.
  for (int level = A.max_level - 1; level >= A.min_level; --level) {
    int child_level = level + 1;
    int nblocks = pow(2, level);
    nblocks_per_proc = ceil(nblocks / double(MPISIZE));

    for (int block = 0; block < nblocks; ++block) {
      int c1 = block * 2;
      int c2 = block * 2 + 1;
      int proc_block = mpi_rank(block);
      int proc_c1 = mpi_rank(c1);
      int proc_c2 = mpi_rank(c2);

      MPI_Request c1_request, c2_request;

      int x_hat_c1_nrows = A.ranks(c1, child_level);
      if (proc_c1 == MPIRANK) {
        int x_hat_index = x_hat_offset + c1 / MPISIZE;
        MPI_Isend(&x_hat[x_hat_index], x_hat_c1_nrows, MPI_DOUBLE,
                  proc_block, c1, MPI_COMM_WORLD, &c1_request);
      }

      int x_hat_c2_nrows = A.ranks(c2, child_level);
      if (proc_c2 == MPIRANK) {
        int x_hat_index = x_hat_offset + c2 / MPISIZE;
        MPI_Isend(&x_hat[x_hat_index], x_hat_c2_nrows, MPI_DOUBLE,
                  proc_block, c2, MPI_COMM_WORLD, &c2_request);
      }

      if (proc_block == MPIRANK) {
        Matrix x_temp = Matrix(A.U(block, level).rows, 1);

        MPI_Status status;
        MPI_Irecv(&x_temp(0, 0), x_hat_c1_nrows, MPI_DOUBLE, proc_c1,
                  c1, MPI_COMM_WORLD, &c1_request);
        MPI_Irecv(&x_temp(x_hat_c1_nrows, 0), x_hat_c2_nrows, MPI_DOUBLE, proc_c2,
                  c2, MPI_COMM_WORLD, &c2_request);

        MPI_Wait(&c1_request, &status);
        MPI_Wait(&c2_request, &status);

        x_hat.push_back(matmul(A.U(block, level), x_temp, true, false, 1.0));
      }
    }
    x_hat_offset += ceil(pow(2, child_level) / double(MPISIZE));
  }

  // allocate b_hat blocks for the lowest level on each process.
  int nblocks = pow(2, A.min_level);
  // nblocks_per_proc = ceil(nblocks / double(MPISIZE));
  std::vector<Matrix> b_hat;
  for (int i = MPIRANK; i < nblocks; i += MPISIZE) {
    b_hat.push_back(Matrix(A.ranks(i, A.min_level), 1));
  }
  int b_hat_offset = 0;

  // update the b_hat with S blocks.
  multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset, A.min_level);

  for (int level = A.min_level; level < A.max_level; ++level) {
    int nblocks = pow(2, level);
    int child_level = level + 1;
    int child_nblocks = pow(2, child_level);
    x_hat_offset -= ceil(double(child_nblocks) / MPISIZE);

    // compute and send the U * b_hat parts.
    std::vector<Matrix> Ub_array;
    for (int block = 0; block < nblocks; ++block) {
      int c1 = block * 2;
      int c2 = block * 2 + 1;
      int index = block / MPISIZE;
      int p_block = mpi_rank(block);
      int p_c1 = mpi_rank(c1);
      int p_c2 = mpi_rank(c2);

      int c1_block_size = A.ranks(c1, child_level);
      int c2_block_size = A.ranks(c2, child_level);

      if (p_block == MPIRANK) {
        MPI_Request r1, r2;

        Matrix Ub = matmul(A.U(block, level), b_hat[b_hat_offset + index]);
        Ub_array.push_back(Ub);
        int s = Ub_array.size();

        MPI_Isend(&Ub_array[s-1](0,0), c1_block_size, MPI_DOUBLE, p_c1, c1, MPI_COMM_WORLD, &r1);
        MPI_Isend(&Ub_array[s-1](c1_block_size, 0), c2_block_size, MPI_DOUBLE, p_c2, c2,
                  MPI_COMM_WORLD, &r2);
      }
    }

    // receive and save the Ub * b_hat parts.
    for (int block = 0; block < nblocks; ++block) {
      int c1 = block * 2;
      int c2 = block * 2 + 1;
      int index = block / MPISIZE;
      int p_block = mpi_rank(block);
      int p_c1 = mpi_rank(c1);
      int p_c2 = mpi_rank(c2);
      int c1_block_size = A.ranks(c1, child_level);
      int c2_block_size = A.ranks(c2, child_level);

      if (p_c1 == MPIRANK) {
        Matrix b_hat_c1(c1_block_size, 1);
        MPI_Status status;

        MPI_Recv(&b_hat_c1, c1_block_size, MPI_DOUBLE, p_block, c1, MPI_COMM_WORLD, &status);
        b_hat.push_back(b_hat_c1);
      }

      if (p_c2 == MPIRANK) {
        Matrix b_hat_c2(c2_block_size, 1);
        MPI_Status status;

        MPI_Recv(&b_hat_c2, c2_block_size, MPI_DOUBLE, p_block, c2, MPI_COMM_WORLD, &status);
        b_hat.push_back(b_hat_c2);
      }
    }

    int nblocks_per_proc = pow(2, level) / MPISIZE;
    if (nblocks_per_proc == 0 && MPIRANK < pow(2, level)) {
      nblocks_per_proc = 1;
    }
    multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset + nblocks_per_proc, child_level);

    b_hat_offset += nblocks_per_proc;
  }

  // process the b blocks with the leaf level.
  nblocks_per_proc = ceil(leaf_nblocks / double(MPISIZE));
  for (int i = 0; i < nblocks_per_proc; ++i) {
    int index = i * MPISIZE + MPIRANK;
    matmul(A.U(index, A.max_level), b_hat[b_hat_offset + i], b[i]);
  }

  // multiply the x with the dense blocks and add to the b
  for (int i = 0; i < nblocks_per_proc; ++i) {
    int index = i * MPISIZE + MPIRANK;
    matmul(A.D(index, index, A.max_level), x[i], b[i]);
  }

  for (int i = 0; i < leaf_nblocks; ++i) {
    int proc_i = mpi_rank(i);
    int D_nrows = domain.cell_size(i, A.max_level);

    for (int j = 0; j < i; ++j) {
      int D_ncols = domain.cell_size(j, A.max_level);
      int proc_j = mpi_rank(j);
      int proc_D = mpi_rank(i, j);
      int D_tag = i + j * nblocks;

      // send D blocks where they are to be computed.
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        MPI_Request i_request, j_request;

        if (proc_D == MPIRANK) {
          Matrix& Dblock = A.D(i, j, A.max_level);
          MPI_Isend(&Dblock, D_nrows * D_ncols, MPI_DOUBLE, proc_i,
                    D_tag, MPI_COMM_WORLD, &i_request);
          MPI_Isend(&Dblock, D_nrows * D_ncols, MPI_DOUBLE, proc_j,
                    D_tag, MPI_COMM_WORLD, &j_request);
        }

        int x_i_tag = i + D_tag + 1;
        if (proc_i == MPIRANK) {
          // send x(i) to proc_j
          int x_index = i / MPISIZE;
          MPI_Isend(&x[x_index], x[x_index].numel(), MPI_DOUBLE, proc_j,
                    x_i_tag, MPI_COMM_WORLD, &j_request);
        }

        int x_j_tag = j + D_tag + 1;
        if (proc_j == MPIRANK) {
          // send x(j) to proc_i
          int x_index = j / MPISIZE;
          MPI_Isend(&x[x_index], x[x_index].numel(), MPI_DOUBLE, proc_i,
                    x_j_tag, MPI_COMM_WORLD, &i_request);
        }
      }
    }

    // receive blocks and compute
    for (int j = 0; j < i; ++j) {
      int D_ncols = domain.cell_size(j, A.max_level);
      int proc_j = mpi_rank(j);
      int proc_D = mpi_rank(i, j);
      int D_tag = i + j * nblocks;

      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        int x_j_tag = j + D_tag + 1;
        if (proc_i == MPIRANK) {
          // receive for b(i)
          Matrix Dij(D_nrows, D_ncols);
          Matrix xj(D_ncols, 1);
          MPI_Status status;

          MPI_Recv(&Dij, D_nrows * D_ncols, MPI_DOUBLE, proc_D, D_tag,
                   MPI_COMM_WORLD, &status);
          MPI_Recv(&xj, D_ncols, MPI_DOUBLE, proc_j, x_j_tag,
                   MPI_COMM_WORLD, &status);

          int index = i / MPISIZE;
          matmul(Dij, xj, b[index]);
        }

        int x_i_tag = i + D_tag + 1;
        if (proc_j == MPIRANK) {
          Matrix Dij(D_nrows, D_ncols);
          Matrix xi(D_nrows, 1);
          MPI_Status status;

          MPI_Recv(&Dij, D_nrows * D_ncols, MPI_DOUBLE, proc_D, D_tag,
                   MPI_COMM_WORLD, &status);
          MPI_Recv(&xi, D_nrows, MPI_DOUBLE, proc_i, x_i_tag,
                   MPI_COMM_WORLD, &status);

          int index = j / MPISIZE;
          matmul(Dij, xi, b[index], true, false);
        }
      }
    }
  }
}

void factorize(SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts) {

}

void solve(SymmetricSharedBasisMatrix& A, std::vector<Matrix>& x, std::vector<Matrix>& h2_solution) {

}
