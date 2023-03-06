#include <random>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "globals.hpp"
#include "h2_operations.hpp"

using namespace Hatrix;

void
compute_fill_ins(SymmetricSharedBasisMatrix& A,
                 const Hatrix::Domain& domain,
                 int64_t block,
                 int64_t level);

int64_t
get_dim(const SymmetricSharedBasisMatrix& A, const Domain& domain,
        const int64_t block, const int64_t level) {
  return level == A.max_level ? domain.cell_size(block, level) :
    (A.ranks(block * 2, level+1) + A.ranks(block * 2 + 1, level + 1));
}

void
solve_forward_level(SymmetricSharedBasisMatrix& A,
                    const Hatrix::Domain& domain,
                    std::vector<Matrix>& x_level,
                    int64_t level) {
  int64_t nblocks = pow(2, level);

  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t block_index = block / MPISIZE;
    if (mpi_rank(block) == MPIRANK) {
      Matrix U_F = make_complement(A.U(block, level));
      Matrix prod = matmul(U_F, x_level[block_index]);
      x_level[block_index] = prod;
    }

    // send the diagonal block to the rank that has the corresponding vector.
    if (mpi_rank(block) == MPIRANK) {
      MPI_Request block_request;
      MPI_Isend(&A.D(block, block, level), A.D(block, block, level).numel(), MPI_DOUBLE,
                mpi_rank(block), block, MPI_COMM_WORLD, &block_request);
    }

    if (mpi_rank(block) == MPIRANK) {
      MPI_Status block_status;
      Matrix D_copy(get_dim(A, domain, block, level), get_dim(A, domain, block, level));

      MPI_Recv(&D_copy, D_copy.numel(), MPI_DOUBLE,
               mpi_rank(block, block), block, MPI_COMM_WORLD, &block_status);

      int64_t rank = A.ranks(block, level);
      auto block_splits = split_dense(D_copy,
                                      D_copy.rows - rank,
                                      D_copy.cols - rank);

      auto x_block_splits =
        x_level[block_index].split(std::vector<int64_t>(1, D_copy.rows - rank),
                                   {});

      solve_triangular(block_splits[0], x_block_splits[0],
                       Hatrix::Left, Hatrix::Lower, false,
                       false, 1.0);
      matmul(block_splits[2], x_block_splits[0], x_block_splits[1],
             false, false, -1.0, 1.0);
    }

    // apply the oc blocks that are actually in the upper triangular matrix.
    for (int64_t irow = 0; irow < block; ++irow) {
      if (exists_and_inadmissible(A, block, irow, level)) {
        int64_t irow_index = irow / MPISIZE;
        int64_t block_index = block / MPISIZE;
        if (mpi_rank(block) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&A.D(block, irow, level), A.D(block, irow, level).numel(), MPI_DOUBLE,
                    mpi_rank(irow), irow, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(block) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&x_level[block_index], x_level[block_index].numel(), MPI_DOUBLE,
                    mpi_rank(irow), block, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(irow) == MPIRANK) {
          int64_t D_block_irow_nrows = get_dim(A, domain, block, level);
          int64_t D_block_irow_ncols = get_dim(A, domain, irow, level);
          MPI_Status status;

          Matrix D_block_irow(D_block_irow_nrows, D_block_irow_ncols);

          MPI_Recv(&D_block_irow, D_block_irow.numel(), MPI_DOUBLE,
                   mpi_rank(block), irow, MPI_COMM_WORLD, &status);

          Matrix x_block(D_block_irow_nrows, 1);

          MPI_Recv(&x_block, x_block.numel(), MPI_DOUBLE,
                   mpi_rank(block), block, MPI_COMM_WORLD, &status);

          int64_t row_split = D_block_irow_nrows - A.ranks(block, level);
          int64_t col_split = D_block_irow_ncols - A.ranks(irow, level);

          auto D_block_irow_splits = split_dense(D_block_irow,
                                                 row_split,
                                                 col_split);

          Matrix& x_irow = x_level[irow_index];
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, col_split), {});

          matmul(D_block_irow_splits[1], x_block_splits[0], x_irow_splits[1],
                 true, false, -1.0, 1.0);
        }
      }
    }

    // forward substitute with (cc;oc) blocks below the diagonal
    for (int64_t irow = block+1; irow < nblocks; ++irow) {
      if (exists_and_inadmissible(A, irow, block, level)) {
        int64_t irow_index = irow / MPISIZE;
        int64_t block_index = block / MPISIZE;
        if (mpi_rank(irow) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&A.D(irow, block, level), A.D(irow, block, level).numel(), MPI_DOUBLE,
                    mpi_rank(irow), irow, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(block) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&x_level[block_index], x_level[block_index].numel(), MPI_DOUBLE,
                    mpi_rank(irow), block, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(irow) == MPIRANK) {
          MPI_Status status;
          int64_t D_irow_block_nrows = get_dim(A, domain, irow, level);
          int64_t D_irow_block_ncols = get_dim(A, domain, block, level);
          int64_t col_split = D_irow_block_ncols - A.ranks(block, level);

          Matrix D_irow_block(D_irow_block_nrows, D_irow_block_ncols);
          MPI_Recv(&D_irow_block, D_irow_block.numel(), MPI_DOUBLE,
                   mpi_rank(irow), irow, MPI_COMM_WORLD, &status);

          Matrix x_block(D_irow_block_ncols, 1);
          MPI_Recv(&x_block, x_block.numel(), MPI_DOUBLE,
                   mpi_rank(block), block, MPI_COMM_WORLD, &status);

          auto lower_splits = D_irow_block.split({},
                                                 std::vector<int64_t>(1, col_split));
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});
          matmul(lower_splits[0], x_block_splits[0], x_level[irow_index],
                 false, false, -1.0, 1.0);
        }
      }
    }
  }
}

std::vector<Matrix>
permute_forward_and_copy(SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         std::vector<Matrix>& x_level,
                         int64_t level) {
  // Generate a new vector with different lengths to store the permutation order.
  int64_t parent_level = level - 1;
  int64_t parent_nblocks = pow(2, parent_level);
  std::vector<Matrix> x_ranks;

  // pre allocate the blocks on this process.
  for (int64_t i = 0; i < parent_nblocks; ++i) {
    if (mpi_rank(i) == MPIRANK) {
      int64_t c1 = i * 2, c2 = i * 2 + 1;
      int64_t length = A.ranks(c1, level) + A.ranks(c2, level);
      Matrix temp(length, 1);
      x_ranks.push_back(temp);
    }
  }

  int64_t nblocks = pow(2, level);
  for (int64_t block = 0; block < nblocks; block += 2) {
    int64_t block_index = block / MPISIZE;
    int64_t rank = A.ranks(block, level);
    int64_t parent_block = block / 2;
    int64_t parent_block_index = parent_block / MPISIZE;

    if (mpi_rank(block) == MPIRANK) {
      int64_t c_size = get_dim(A, domain, block, level) - rank;
      Matrix& x_level_rank_part = x_level[block_index];
      MPI_Request request;

      MPI_Isend(&x_level_rank_part(c_size, 0), rank, MPI_DOUBLE,
                mpi_rank(parent_block), block, MPI_COMM_WORLD, &request);
    }

    int64_t block_2 = block + 1;
    int64_t block_2_index = block_2 / MPISIZE;
    int64_t rank_2 = A.ranks(block_2, level);

    if (mpi_rank(block_2) == MPIRANK) {
      int64_t c_size = get_dim(A, domain, block_2, level) - rank_2;
      Matrix& x_level_rank_2_part = x_level[block_2_index];
      MPI_Request request;

      MPI_Isend(&x_level_rank_2_part(c_size, 0), rank_2, MPI_DOUBLE,
                mpi_rank(parent_block), block_2, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(parent_block) == MPIRANK) {
      MPI_Status status;
      Matrix& x_rank = x_ranks[parent_block_index];

      MPI_Recv(&x_rank(0, 0), rank, MPI_DOUBLE,
               mpi_rank(block), block, MPI_COMM_WORLD, &status);

      MPI_Recv(&x_rank(rank, 0), rank_2, MPI_DOUBLE,
               mpi_rank(block_2), block_2, MPI_COMM_WORLD, &status);
    }
  }

  return x_ranks;
}

void
permute_backward_and_copy(SymmetricSharedBasisMatrix& A,
                          const Hatrix::Domain& domain,
                          std::vector<Matrix>& x_level,
                          std::vector<Matrix>& x_level_child,
                          int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);

  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t child1 = block * 2;
    int64_t child2 = block * 2 + 1;

    int64_t block_index = block / MPISIZE;
    int64_t child1_index = child1 / MPISIZE;
    int64_t child2_index = child2 / MPISIZE;

    int64_t rank_child1 = A.ranks(child1, child_level);
    int64_t rank_child2 = A.ranks(child2, child_level);

    if (mpi_rank(block) == MPIRANK) {
      Matrix& x_level_block = x_level[block_index];
      MPI_Request request1, request2;

      MPI_Isend(&x_level_block(0, 0), rank_child1, MPI_DOUBLE,
                mpi_rank(child1), child1, MPI_COMM_WORLD, &request1);

      MPI_Isend(&x_level_block(rank_child1, 0), rank_child2, MPI_DOUBLE,
                mpi_rank(child2), child2, MPI_COMM_WORLD, &request2);
    }

    if (mpi_rank(child1) == MPIRANK) {
      int64_t c_size = get_dim(A, domain, child1, child_level) - rank_child1;
      Matrix& x_level_child1 = x_level_child[child1_index];
      MPI_Status status;

      MPI_Recv(&x_level_child1(c_size, 0), rank_child1, MPI_DOUBLE,
               mpi_rank(block), child1, MPI_COMM_WORLD, &status);

    }

    if (mpi_rank(child2) == MPIRANK) {
      int64_t c_size = get_dim(A, domain, child2, child_level) - rank_child2;
      Matrix& x_level_child2 = x_level_child[child2_index];
      MPI_Status status;

      MPI_Recv(&x_level_child2(c_size, 0), rank_child2, MPI_DOUBLE,
               mpi_rank(block), child2, MPI_COMM_WORLD, &status);
    }
  }
}

void
solve_backward_level(SymmetricSharedBasisMatrix& A,
                     const Hatrix::Domain& domain,
                     std::vector<Matrix>& x_level,
                     int64_t level) {
  int64_t nblocks = pow(2, level);

  for (int64_t block = nblocks-1; block >=0; --block) {
    int64_t block_index = block / MPISIZE;

    // apply the tranpose of the oc block that is actually in the lower triangle.
    for (int64_t icol = 0; icol < block; ++icol) {
      if (exists_and_inadmissible(A, block, icol, level)) {
        int64_t icol_index = icol / MPISIZE;
        if (mpi_rank(block) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&A.D(block, icol, level), A.D(block, icol, level).numel(), MPI_DOUBLE,
                    mpi_rank(icol), icol, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(block) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&x_level[block_index], x_level[block_index].numel(), MPI_DOUBLE,
                    mpi_rank(icol), block, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(icol) == MPIRANK) {
          MPI_Status status;
          int64_t D_block_icol_nrows = get_dim(A, domain, block, level);
          int64_t D_block_icol_ncols = get_dim(A, domain, icol, level);
          int64_t row_split = D_block_icol_nrows - A.ranks(block, level);
          int64_t col_split = D_block_icol_ncols - A.ranks(icol, level);

          Matrix D_block_icol(D_block_icol_nrows, D_block_icol_ncols);
          MPI_Recv(&D_block_icol, D_block_icol.numel(), MPI_DOUBLE,
                   mpi_rank(block), icol, MPI_COMM_WORLD, &status);

          Matrix x_block(D_block_icol_nrows, 1);
          MPI_Recv(&x_block, x_block.numel(), MPI_DOUBLE,
                   mpi_rank(block), block, MPI_COMM_WORLD, &status);

          auto D_block_icol_splits = split_dense(D_block_icol, row_split, col_split);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split),
                                              {});
          auto x_icol_splits = x_level[icol_index].split(std::vector<int64_t>(1, col_split),
                                                   {});
          matmul(D_block_icol_splits[2], x_block_splits[1], x_icol_splits[0],
                 true, false, -1.0, 1.0);
        }
      }
    }

    // apply cc and oc blocks (transposed) to the respective slice of the vector.
    for (int64_t icol = nblocks-1; icol > block; --icol) {
      if (exists_and_inadmissible(A, icol, block, level)) {
        int64_t icol_index = icol / MPISIZE;
        int64_t block_index = block / MPISIZE;
        if (mpi_rank(icol) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&A.D(icol, block, level), A.D(icol, block, level).numel(), MPI_DOUBLE,
                    mpi_rank(block), block, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(icol) == MPIRANK) {
          MPI_Request request;
          MPI_Isend(&x_level[icol_index], x_level[icol_index].numel(), MPI_DOUBLE,
                    mpi_rank(block), icol, MPI_COMM_WORLD, &request);
        }

        if (mpi_rank(block) == MPIRANK) {
          MPI_Status status;
          int64_t D_icol_block_nrows = get_dim(A, domain, icol, level);
          int64_t D_icol_block_ncols = get_dim(A, domain, block, level);
          int64_t col_split = D_icol_block_ncols - A.ranks(block, level);

          Matrix D_icol_block(D_icol_block_nrows, D_icol_block_ncols);
          MPI_Recv(&D_icol_block, D_icol_block.numel(), MPI_DOUBLE,
                   mpi_rank(icol), block, MPI_COMM_WORLD, &status);

          Matrix x_icol(D_icol_block_nrows, 1);
          MPI_Recv(&x_icol, x_icol.numel(), MPI_DOUBLE,
                   mpi_rank(icol), icol, MPI_COMM_WORLD, &status);

          auto D_icol_block_splits = D_icol_block.split({},
                                                        std::vector<int64_t>(1, col_split));
          auto x_block_splits = x_level[block_index].split(std::vector<int64_t>(1,
                                                                          col_split),
                                                     {});
          matmul(D_icol_block_splits[0], x_icol, x_block_splits[0],
                 true, false, -1.0, 1.0);
        }
      }
    }

    // backward using the diagonal block.
    int64_t rank = A.ranks(block, level);
    int64_t row_split = get_dim(A, domain, block, level) - rank;
    int64_t col_split = get_dim(A, domain, block, level) - rank;

    if (mpi_rank(block) == MPIRANK) {
      MPI_Request request;

      MPI_Isend(&A.D(block, block, level), A.D(block, block, level).numel(), MPI_DOUBLE,
                mpi_rank(block), block, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(block) == MPIRANK) {
      MPI_Status status;

      Matrix D_copy(get_dim(A, domain, block, level), get_dim(A, domain, block, level));
      MPI_Recv(&D_copy, D_copy.numel(), MPI_DOUBLE,
               mpi_rank(block), block, MPI_COMM_WORLD, &status);
      auto block_splits = split_dense(D_copy,
                                      D_copy.rows - rank,
                                      D_copy.cols - rank);

      auto x_block_splits =
        x_level[block_index].split(std::vector<int64_t>(1, D_copy.rows - rank),
                                   {});

      matmul(block_splits[2], x_block_splits[1], x_block_splits[0],
             true, false, -1.0, 1.0);
      solve_triangular(block_splits[0], x_block_splits[0],
                       Hatrix::Left, Hatrix::Lower, false,
                       true, 1.0);
    }

    if (mpi_rank(block) == MPIRANK) {
      Matrix V_F = make_complement(A.U(block, level));
      Matrix prod = matmul(V_F, x_level[block_index], true);
      x_level[block_index] = prod;
    }
  }
}

void
solve(SymmetricSharedBasisMatrix& A,
      std::vector<Matrix>& b,
      std::vector<Matrix>& h2_solution,
      const Hatrix::Domain& domain) {
  std::vector<Matrix> x(b);
  int64_t level;

  int64_t x_levels_index = 0;
  std::vector<std::vector<Matrix>> x_levels;
  x_levels.push_back(x);

  for (level = A.max_level; level >= A.min_level; --level) {
    std::vector<Matrix>& x_level = x_levels[x_levels_index];
    // partial forward solve.
    solve_forward_level(A, domain, x_level, level);

    // permute and copy.
    auto x_level_permuted = permute_forward_and_copy(A, domain, x_level, level);
    x_levels.push_back(x_level_permuted);
    x_levels_index++;
  }

  int64_t last_nodes = pow(2, level);

  // forward of the last blocks
  std::vector<Matrix>& x_last = x_levels[x_levels_index];
  for (int64_t i = 0; i < last_nodes; ++i) {
    for (int64_t j = 0; j < i; ++j) { // off-diagonals
      int dense_block_tag = i * last_nodes + j;
      int64_t i_index = i / MPISIZE;
      int64_t j_index = j / MPISIZE;
      if (mpi_rank(i) == MPIRANK && exists_and_inadmissible(A, i, j, level)) {
        MPI_Request request;
        MPI_Isend(&A.D(i, j, level), A.D(i, j, level).numel(), MPI_DOUBLE,
                  mpi_rank(i), dense_block_tag, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(j) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&x_last[j_index], x_last[j_index].numel(), MPI_DOUBLE,
                  mpi_rank(i), j, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(i) == MPIRANK && exists_and_inadmissible(A, i, j, level)) {
        MPI_Status status;
        int64_t D_ij_nrows = get_dim(A, domain, i, level);
        int64_t D_ij_ncols = get_dim(A, domain, j, level);

        Matrix D_ij(D_ij_nrows, D_ij_ncols);
        MPI_Recv(&D_ij, D_ij.numel(), MPI_DOUBLE,
                 mpi_rank(i), dense_block_tag, MPI_COMM_WORLD, &status);

        Matrix x_last_j(D_ij_ncols, 1);
        MPI_Recv(&x_last_j, x_last_j.numel(), MPI_DOUBLE,
                 mpi_rank(j), j, MPI_COMM_WORLD, &status);

        matmul(D_ij, x_last_j, x_last[i_index], false, false, -1, 1);
      }
    }

    // diagonal block
    if (mpi_rank(i) == MPIRANK) {
      Matrix& D_ii = A.D(i, i, level);
      MPI_Request request;
      MPI_Isend(&D_ii, D_ii.numel(), MPI_DOUBLE,
                mpi_rank(i), i, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(i) == MPIRANK) {
      MPI_Status status;
      Matrix D_ii(get_dim(A, domain, i, level), get_dim(A, domain, i, level));

      MPI_Recv(&D_ii, D_ii.numel(), MPI_DOUBLE,
               mpi_rank(i), i, MPI_COMM_WORLD, &status);

      int64_t i_index = i / MPISIZE;
      solve_triangular(D_ii, x_last[i_index], Hatrix::Left,
                       Hatrix::Lower, false, false, 1.0);
    }
  }


  // backward of the last blocks
  for (int64_t j = last_nodes - 1; j >= 0; --j) {
    for (int64_t i = last_nodes - 1; i > j; --i) {
      int64_t i_index = i / MPISIZE;
      int64_t j_index = j / MPISIZE;
      int dense_block_tag = i * last_nodes + j;
      if (mpi_rank(i) == MPIRANK && exists_and_inadmissible(A, i, j, level)) {
        MPI_Request request;
        MPI_Isend(&A.D(i, j, level), A.D(i, j, level).numel(), MPI_DOUBLE,
                  mpi_rank(j), dense_block_tag, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(i) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&x_last[i_index], x_last[i_index].numel(), MPI_DOUBLE,
                  mpi_rank(j), i, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(j) == MPIRANK && exists_and_inadmissible(A, i, j, level)) {
        MPI_Status status;
        int64_t D_ij_nrows = get_dim(A, domain, i, level);
        int64_t D_ij_ncols = get_dim(A, domain, j, level);

        Matrix D_ij(D_ij_nrows, D_ij_ncols);
        MPI_Recv(&D_ij, D_ij.numel(), MPI_DOUBLE,
                 mpi_rank(i), dense_block_tag, MPI_COMM_WORLD, &status);

        Matrix x_last_i(D_ij_nrows, 1);
        MPI_Recv(&x_last_i, x_last_i.numel(), MPI_DOUBLE,
                 mpi_rank(i), i, MPI_COMM_WORLD, &status);

        matmul(D_ij, x_last_i, x_last[j_index], true, false, -1.0, 1.0);
      }
    }

    if (mpi_rank(j) == MPIRANK) {
      Matrix& D_jj = A.D(j, j, level);
      MPI_Request request;
      MPI_Isend(&D_jj, D_jj.numel(), MPI_DOUBLE,
                mpi_rank(j), j, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(j) == MPIRANK) {
      MPI_Status status;
      Matrix D_jj(get_dim(A, domain, j, level), get_dim(A, domain, j, level));

      MPI_Recv(&D_jj, D_jj.numel(), MPI_DOUBLE,
               mpi_rank(j), j, MPI_COMM_WORLD, &status);

      int64_t j_index = j / MPISIZE;
      solve_triangular(D_jj, x_last[j_index], Hatrix::Left,
                       Hatrix::Lower, false, true, 1.0);
    }
  }

  ++level;

  for (; level <= A.max_level; ++level) {
    std::vector<Matrix>& x_level = x_levels[x_levels_index];
    std::vector<Matrix>& x_level_child = x_levels[x_levels_index-1];
    // permute and copy.
    permute_backward_and_copy(A, domain, x_level, x_level_child, level-1);
    // partial backward solve.
    solve_backward_level(A, domain, x_level_child, level);
    x_levels_index--;
  }

  for (int64_t i = 0; i < x_levels[0].size(); ++i) {
    h2_solution[i] = x_levels[0][i];
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

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
        int proc_S = mpi_rank(i);
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
        int proc_S = mpi_rank(i);

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

  if (MPIRANK >= leaf_nblocks) {
    return;
  }

  // Apply V leaf basis nodes.
  for (int i = 0; i < nblocks_per_proc; i += 1) {
    int index = i * MPISIZE + MPIRANK;
    x_hat.push_back(matmul(A.U(index, A.max_level), x[i],
                           true, false, 1.0));
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
        MPI_Irecv(&x_temp(x_hat_c1_nrows, 0), x_hat_c2_nrows,
                  MPI_DOUBLE, proc_c2, c2, MPI_COMM_WORLD, &c2_request);

        MPI_Wait(&c1_request, &status);
        MPI_Wait(&c2_request, &status);

        x_hat.push_back(matmul(A.U(block, level), x_temp,
                               true, false, 1.0));
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

    std::vector<Matrix> Ub_array;
    // If you call push_back dynamically without pre-allocating the array
    // then it leads to memory issues since the push_back will delete and
    // reallocate if the size of the elements is not enough.
    for (int block = 0; block < nblocks; ++block) {
      int p_block = mpi_rank(block);
      if (p_block == MPIRANK) {
        Ub_array.push_back(Matrix(A.U(block, level).rows, 1));
      }
    }

    int Ub_index = 0;

    // compute and send the U * b_hat parts.
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
        Matrix Ub =
          matmul(A.U(block, level), b_hat[b_hat_offset + index]);
        Ub_array[Ub_index] = Ub;

        Matrix& Ub_ref = Ub_array[Ub_index];

        MPI_Isend(&Ub_ref(0,0), c1_block_size, MPI_DOUBLE,
                  p_c1, c1, MPI_COMM_WORLD, &r1);
        MPI_Isend(&Ub_ref(c1_block_size, 0), c2_block_size, MPI_DOUBLE,
                  p_c2, c2, MPI_COMM_WORLD, &r2);
        Ub_index++;
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

        MPI_Recv(&b_hat_c1, c1_block_size, MPI_DOUBLE,
                 p_block, c1, MPI_COMM_WORLD, &status);
        b_hat.push_back(b_hat_c1);
      }

      if (p_c2 == MPIRANK) {
        Matrix b_hat_c2(c2_block_size, 1);
        MPI_Status status;

        MPI_Recv(&b_hat_c2, c2_block_size, MPI_DOUBLE,
                 p_block, c2, MPI_COMM_WORLD, &status);
        b_hat.push_back(b_hat_c2);
      }
    }

    int nblocks_per_proc = pow(2, level) / MPISIZE;
    if (nblocks_per_proc == 0 && MPIRANK < pow(2, level)) {
      nblocks_per_proc = 1;
    }
    multiply_S(A, x_hat, b_hat, x_hat_offset,
               b_hat_offset + nblocks_per_proc, child_level);

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
      int proc_D = mpi_rank(i);
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
      int proc_D = mpi_rank(i);
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
