#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "globals.hpp"
#include "h2_tasks.hpp"
#include "h2_dtd_operations.hpp"

using namespace Hatrix;

static h2_dc_t parsec_U, parsec_S, parsec_D;
static Hatrix::RowColLevelMap<int> arena_D, arena_S;
static Hatrix::RowColMap<int> arena_U;

int64_t
get_dim(const SymmetricSharedBasisMatrix& A, const Domain& domain,
        const int64_t block, const int64_t level) {
  return level == A.max_level ? domain.cell_size(block, level) :
    (A.ranks(block * 2, level+1) + A.ranks(block * 2 + 1, level + 1));
}

parsec_data_t*
data_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  h2_dc_t* dc = (h2_dc_t*)desc;
  parsec_data_t* data = NULL;

  if (dc->data_map.count(key) != 0) {
    data = dc->data_map[key];
  }

  if (NULL == data) {
    Matrix* mat = dc->matrix_map[key];
    if (mat) {
      data = parsec_data_create(&dc->data_map[key], desc, key, &(*mat),
                                mat->numel() *
                                parsec_datadist_getsizeoftype(PARSEC_MATRIX_DOUBLE),
                                PARSEC_DATA_FLAG_PARSEC_MANAGED);
    }
    else {
      data = parsec_data_create(&dc->data_map[key], desc, key, NULL, 0,
                                PARSEC_DATA_FLAG_PARSEC_MANAGED);
    }
    dc->data_map[key] = data;
  }

  return data;
}

uint32_t rank_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  h2_dc_t *dc = (h2_dc_t*)desc;
  return dc->mpi_ranks[key];
}

parsec_data_key_t data_key_1d(parsec_data_collection_t* dc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, dc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (b == -1 || level == -1) {
    std::cout << "[HATRIX_ERROR] data_key_1d() -> received wrong parameters b = "
              << b << " and level = " << level << ". Aborting.\n";
    abort();
  }

  parsec_data_key_t key = 0;
  for (int i = 0; i < level; ++i) {
    key += pow(2, i);
  }

  return key + b;
}

parsec_data_key_t data_key_2d(parsec_data_collection_t* dc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, dc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (m == -1 || n == -1 || level == -1) {
    std::cout << "[HATRIX ERROR] data_key_2d() -> received a wrong key in m, n or level.\n";
    abort();
  }

  parsec_data_key_t key = 1;
  for (int i = 0; i < level; ++i) {
    key += pow(pow(2, i), 2);
  }

  int stride = pow(2, level);
  auto data_key = key + m * stride + n;

  return data_key;
}

uint32_t rank_of_1d(parsec_data_collection_t* desc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, desc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (b == -1 || level == -1) {
    std::cout << "[HATRIX_ERROR] data_key_1d() -> received wrong parameters b = "
              << b << " and level = " << level << ". Aborting.\n";
    abort();
  }

  return mpi_rank(b);
}

// block cyclic distribution.
uint32_t rank_of_2d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (m == -1 || n == -1 || level == -1) {
    std::cout << "[HATRIX ERROR] data_key_2d() -> received a wrong key in m, n or level.\n";
    abort();
  }

  return mpi_rank(m, n);
}

void
h2_dc_init(h2_dc_t& parsec_data,
           parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...),
           uint32_t (*rank_of_func)(parsec_data_collection_t*, ...)) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->rank_of = rank_of_func;
  o->data_key = data_key_func;
  o->data_of_key = data_of_key;
  o->rank_of_key = rank_of_key;

  parsec_dtd_data_collection_init(o);
}

void
h2_dc_destroy(h2_dc_t& parsec_data) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_dtd_data_collection_fini(o);
  for (auto iter = parsec_data.data_map.begin(); iter != parsec_data.data_map.end(); ++iter) {
    parsec_data_destroy((iter->second));
  }
  parsec_data_collection_destroy(o);
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

  if (MPIRANK >= leaf_nblocks) {
    return;
  }

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

        Matrix& Ub_ref = Ub_array[s-1];

        MPI_Isend(&Ub_ref(0,0), c1_block_size, MPI_DOUBLE, p_c1, c1, MPI_COMM_WORLD, &r1);
        MPI_Isend(&Ub_ref(c1_block_size, 0), c2_block_size, MPI_DOUBLE, p_c2, c2,
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

// Copy blocks from the child level into level.
void
merge_unfactorized_blocks(SymmetricSharedBasisMatrix& A, const Domain& domain, int64_t level) {
  const int64_t parent_level = level - 1;
  const int64_t parent_nblocks = pow(2, parent_level);

  for (int64_t i = 0; i < parent_nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (exists_and_inadmissible(A, i, j, parent_level)) {
        std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});
        int64_t D_unelim_rows = A.ranks(i_children[0], level) + A.ranks(i_children[1], level);
        int64_t D_unelim_cols = A.ranks(j_children[0], level) + A.ranks(j_children[1], level);
        parsec_data_key_t D_unelim_key = parsec_D.super.data_key(&parsec_D.super, i, j, parent_level);

        int64_t D_unelim_row_rank = A.ranks(i_children[0], level);
        int64_t D_unelim_col_rank = A.ranks(j_children[0], level);

        for (int ic1 = 0; ic1 < 2; ++ic1) {
          for (int jc2 = 0; jc2 < ((i == j) ? (ic1+1) : 2); ++jc2) {
            int64_t c1 = i_children[ic1], c2 = j_children[jc2];
            bool copy_dense;
            int D_unelim_split_index = ic1 * 2 + jc2;

            int64_t D_c1c2_rows = get_dim(A, domain, c1, level);
            int64_t D_c1c2_cols = get_dim(A, domain, c2, level);
            int64_t D_c1c2_row_rank = A.ranks(c1, level);
            int64_t D_c1c2_col_rank = A.ranks(c2, level);

            if (A.is_admissible.exists(c1, c2, level)) {
              if (!A.is_admissible(c1, c2, level)) {
                // copy oo portion of the D blocks.
                copy_dense = true;
                parsec_data_key_t D_c1c2_key = parsec_D.super.data_key(&parsec_D.super, c1, c2, level);

                parsec_dtd_insert_task(dtd_tp, task_copy_blocks, 0, PARSEC_DEV_CPU,
                  "copy_blocks_task",
                  sizeof(bool), &copy_dense, PARSEC_VALUE,
                  PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_unelim_key), PARSEC_INOUT | arena_D(i, j, parent_level) | PARSEC_AFFINITY,
                  sizeof(int64_t), &D_unelim_rows, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_cols, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_row_rank, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_col_rank, PARSEC_VALUE,
                  PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_c1c2_key), PARSEC_INPUT | arena_D(c1, c2, level),
                  sizeof(int64_t), &D_c1c2_rows, PARSEC_VALUE,
                  sizeof(int64_t), &D_c1c2_cols, PARSEC_VALUE,
                  sizeof(int64_t), &D_c1c2_row_rank, PARSEC_VALUE,
                  sizeof(int64_t), &D_c1c2_col_rank, PARSEC_VALUE,
                  sizeof(int), &D_unelim_split_index, PARSEC_VALUE,
                  PARSEC_DTD_ARG_END);
              }
              else {
                // copy full S blocks into the parent D block
                copy_dense = false;
                parsec_data_key_t  S_c1c2_key = parsec_S.super.data_key(&parsec_S.super, c1, c2, level);
                int64_t S_c1c2_rows = A.ranks(c1, level);
                int64_t S_c1c2_cols = A.ranks(c2, level);
                int64_t MINUS_ONE = -1;

                parsec_dtd_insert_task(dtd_tp, task_copy_blocks, 0, PARSEC_DEV_CPU,
                  "copy_blocks_task",
                  sizeof(bool), &copy_dense, PARSEC_VALUE,
                  PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_unelim_key), PARSEC_INOUT | arena_D(i, j, parent_level) | PARSEC_AFFINITY,
                  sizeof(int64_t), &D_unelim_rows, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_cols, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_row_rank, PARSEC_VALUE,
                  sizeof(int64_t), &D_unelim_col_rank, PARSEC_VALUE,
                  PASSED_BY_REF, parsec_dtd_tile_of(&parsec_S.super, S_c1c2_key), PARSEC_INPUT | arena_S(c1, c2, level),
                  sizeof(int64_t), &S_c1c2_rows, PARSEC_VALUE,
                  sizeof(int64_t), &S_c1c2_cols, PARSEC_VALUE,
                  sizeof(int64_t), &MINUS_ONE, PARSEC_VALUE,
                  sizeof(int64_t), &MINUS_ONE, PARSEC_VALUE,
                  sizeof(int), &D_unelim_split_index, PARSEC_VALUE,
                  PARSEC_DTD_ARG_END);
              }
            }
          } // for ic1
        }   // for jc2
      }   // if exists and inadmissible
    }   // for j
  }   // for i

  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_S.super);
}

void
multiply_complements(SymmetricSharedBasisMatrix& A,
                     Domain& domain,
                     const int64_t block, const int64_t level) {
  int64_t nblocks = pow(2, level);
  bool left = true;

  parsec_data_key_t U_key = parsec_U.super.data_key(&parsec_U.super, block, level);
  parsec_data_key_t D_key = parsec_D.super.data_key(&parsec_D.super, block, block, level);
  auto U_tile = parsec_dtd_tile_of(&parsec_U.super, U_key);

  int64_t D_nrows = get_dim(A, domain, block, level), U_nrows = get_dim(A, domain, block, level);
  int64_t D_ncols = get_dim(A, domain, block, level), U_ncols = A.ranks(block, level);

  parsec_dtd_insert_task(dtd_tp, task_multiply_full_complement, 0, PARSEC_DEV_CPU,
    "multiply_full_complement_task",
    sizeof(bool), &left, PARSEC_VALUE,
    PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_key), PARSEC_INOUT | arena_D(block, block, level) | PARSEC_AFFINITY,
    sizeof(int64_t), &D_nrows, PARSEC_VALUE,
    sizeof(int64_t), &D_ncols, PARSEC_VALUE,
    PASSED_BY_REF, U_tile, PARSEC_INPUT | arena_U(block, level),
    sizeof(int64_t), &U_nrows, PARSEC_VALUE,
    sizeof(int64_t), &U_ncols, PARSEC_VALUE,
    PARSEC_DTD_ARG_END);


  // for (int64_t j = 0; j < block; ++j) {
  //   parsec_dtd_insert_task(dtd_tp, task_multiply_partial_complement, 0, PARSEC_DEV_CPU,
  //                          "multiply_complement_left_task",
  //                          PARSEC_DTD_ARG_END);
  // }

  // left = false;
  // parsec_dtd_insert_task(dtd_tp, task_multiply_full_complement, 0, PARSEC_DEV_CPU,
  //   "multiply_full_complement_task",
  //   sizeof(bool), &left, PARSEC_VALUE,
  //   PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_key), PARSEC_INOUT | arena_D(block, block, level) | PARSEC_AFFINITY,
  //   sizeof(int64_t), &D_nrows, PARSEC_VALUE,
  //   sizeof(int64_t), &D_ncols, PARSEC_VALUE,
  //   PASSED_BY_REF, U_tile, PARSEC_INPUT | arena_U(block, level),
  //   sizeof(int64_t), &U_nrows, PARSEC_VALUE,
  //   sizeof(int64_t), &U_ncols, PARSEC_VALUE,
  //   PARSEC_DTD_ARG_END);


  // for (int64_t i = block+1; i < nblocks; ++i) {
  //   parsec_dtd_insert_task(dtd_tp, task_multiply_partial_complement, 0, PARSEC_DEV_CPU,
  //                          "multiply_complement_left_task",
  //                          PARSEC_DTD_ARG_END);
  // }

  parsec_dtd_data_flush_all(dtd_tp, &parsec_U.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
}

void factorize_diagonal(SymmetricSharedBasisMatrix& A,
                        const Domain& domain,
                        const int64_t block,
                        const int64_t level) {
  int64_t D_nrows = get_dim(A, domain, block, level);
  int64_t rank_nrows = A.ranks(block, level);
  auto D_key = parsec_D.super.data_key(&parsec_D.super, block, block, level);


  parsec_dtd_insert_task(dtd_tp, task_factorize_diagonal, 0, PARSEC_DEV_CPU,
    "factorize_diagonal_task",
    sizeof(int64_t), &D_nrows, PARSEC_VALUE,
    sizeof(int64_t), &rank_nrows, PARSEC_VALUE,
    PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_key), PARSEC_INOUT | arena_D(block, block, level) | PARSEC_AFFINITY,
    PARSEC_DTD_ARG_END);


  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
}

void partial_triangle_reduce(SymmetricSharedBasisMatrix& A,
                                 parsec_data_key_t diagonal_key,
                                 const Domain& domain,
                                 const int64_t i,
                                 const int64_t block,
                                 const int64_t level,
                                 int64_t SPLIT_INDEX,
                                 Hatrix::Side side,
                                 Hatrix::Mode uplo,
                                 bool UNIT_DIAG,
                                 bool TRANS_A) {
    if (exists_and_inadmissible(A, i, block, level)) {
      parsec_data_key_t other_key = parsec_D.super.data_key(&parsec_D.super, i, block, level);

      int64_t D_rows = get_dim(A, domain, block, level);
      int64_t D_cols = D_rows;
      int64_t D_row_rank = A.ranks(block, level);
      int64_t D_col_rank = A.ranks(block, level);

      int64_t O_rows = get_dim(A, domain, i, level);
      int64_t O_cols = D_cols;
      int64_t O_row_rank = A.ranks(i, level);
      int64_t O_col_rank = A.ranks(block, level);

      parsec_dtd_insert_task(dtd_tp, task_partial_trsm, 0, PARSEC_DEV_CPU,
        "partial_trsm_task",
        sizeof(int64_t), &D_rows, PARSEC_VALUE,
        sizeof(int64_t), &D_cols, PARSEC_VALUE,
        sizeof(int64_t), &D_row_rank, PARSEC_VALUE,
        sizeof(int64_t), &D_col_rank, PARSEC_VALUE,
        PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, diagonal_key), PARSEC_INPUT | arena_D(block, block, level),
        sizeof(int64_t), &O_rows, PARSEC_VALUE,
        sizeof(int64_t), &O_cols, PARSEC_VALUE,
        sizeof(int64_t), &O_row_rank, PARSEC_VALUE,
        sizeof(int64_t), &O_col_rank, PARSEC_VALUE,
        PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, other_key), PARSEC_INOUT | arena_D(i, block, level) | PARSEC_AFFINITY,
        sizeof(Hatrix::Side), &side, PARSEC_VALUE,
        sizeof(Hatrix::Mode), &uplo, PARSEC_VALUE,
        sizeof(bool), &UNIT_DIAG, PARSEC_VALUE,
        sizeof(bool), &TRANS_A, PARSEC_VALUE,
        sizeof(int64_t), &SPLIT_INDEX, PARSEC_VALUE,
        PARSEC_DTD_ARG_END);
    }
}


void triangle_reduction(SymmetricSharedBasisMatrix& A,
                        const Domain& domain,
                        const int64_t block,
                        const int64_t level) {
  int64_t nblocks = pow(2, level);
  parsec_data_key_t diagonal_key = parsec_D.super.data_key(&parsec_D.super, block, block, level);

  // trsm with oc along the 'block' column
  for (int64_t i = block; i < nblocks; ++i) {
    // TODO: Fuse this task with the one below since it is on the same block.
    partial_triangle_reduce(A, diagonal_key, domain, i, block, level, 2,
                            Hatrix::Right, Hatrix::Lower, false, true);
  }

  // TRSM with cc blocks along the 'block' column after the diagonal block.
  for (int64_t i = block+1; i < nblocks; ++i) {
    partial_triangle_reduce(A, diagonal_key, domain, i, block, level, 0,
                            Hatrix::Right, Hatrix::Lower, false, true);

    parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);

    // TODO: work with george to remove this.
    int rc = parsec_dtd_taskpool_wait(dtd_tp);
    PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
  }

  // TRSM with co blocks behing the diagonal on the 'block' row.
  for (int64_t j = 0; j < block; ++j) {
    partial_triangle_reduce(A, diagonal_key, domain, block, j, level, 1,
                            Hatrix::Left, Hatrix::Lower, false, false);
  }


  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);

  // TODO: work with george to remove this.
  int rc = parsec_dtd_taskpool_wait(dtd_tp);
  PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");
}

template<typename T> void
reduction_loop1(SymmetricSharedBasisMatrix& A,
                const Domain& domain,
                int64_t block, int64_t level, T&& body) {
  const int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_i_block_key = parsec_D.super.data_key(&parsec_D.super, i, block, level);
      int64_t D_i_block_rows = get_dim(A, domain, i, level);
      int64_t D_i_block_cols = get_dim(A, domain, block, level);
      int64_t D_i_block_row_rank = A.ranks(i, level);
      int64_t D_i_block_col_rank = A.ranks(block, level);

      for (int64_t j = block+1; j <= i; ++j) {
        if (exists_and_inadmissible(A, j, block, level)) {
          auto D_j_block_key = parsec_D.super.data_key(&parsec_D.super, j, block, level);
          int64_t D_j_block_rows = get_dim(A, domain, j, level);
          int64_t D_j_block_cols = get_dim(A, domain, block, level);
          int64_t D_j_block_row_rank = A.ranks(j, level);
          int64_t D_j_block_col_rank = A.ranks(block, level);

          body(i, j,
               D_i_block_key, D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank,
               D_j_block_key, D_j_block_rows, D_j_block_cols, D_j_block_row_rank, D_j_block_col_rank);
        }
      }
    }
  }
}

template<typename T> void
reduction_loop2(SymmetricSharedBasisMatrix& A,
                const Domain& domain,
                int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);

  for (int64_t i = block; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_i_block_key = parsec_D.super.data_key(&parsec_D.super, i, block, level);
      int64_t D_i_block_rows = get_dim(A, domain, i, level);
      int64_t D_i_block_cols = get_dim(A, domain, block, level);
      int64_t D_i_block_row_rank = A.ranks(i, level);
      int64_t D_i_block_col_rank = A.ranks(block, level);

      for (int64_t j = block; j <= i; ++j) {
        if (exists_and_inadmissible(A, j, block, level)) {
          auto D_j_block_key = parsec_D.super.data_key(&parsec_D.super, j, block, level);
          int64_t D_j_block_rows = get_dim(A, domain, j, level);
          int64_t D_j_block_cols = get_dim(A, domain, block, level);
          int64_t D_j_block_row_rank = A.ranks(j, level);
          int64_t D_j_block_col_rank = A.ranks(block, level);

          body(i, j,
               D_i_block_key, D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank,
               D_j_block_key, D_j_block_rows, D_j_block_cols, D_j_block_row_rank, D_j_block_col_rank);
        }
      }
    }
  }
}

template<typename T> void
reduction_loop4(SymmetricSharedBasisMatrix& A,
                const Domain& domain,
                int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_i_block_key = parsec_D.super.data_key(&parsec_D.super, i, block, level);
      int64_t D_i_block_rows = get_dim(A, domain, i, level);
      int64_t D_i_block_cols = get_dim(A, domain, block, level);
      int64_t D_i_block_row_rank = A.ranks(i, level);
      int64_t D_i_block_col_rank = A.ranks(block, level);

      for (int64_t j = 0; j <= block; ++j) {
        if (exists_and_inadmissible(A, block, j, level)) {
          auto    D_block_j_key = parsec_D.super.data_key(&parsec_D.super, block, j, level);
          int64_t D_block_j_rows = get_dim(A, domain, block, level);
          int64_t D_block_j_cols = get_dim(A, domain, j, level);
          int64_t D_block_j_row_rank = A.ranks(block, level);
          int64_t D_block_j_col_rank = A.ranks(j, level);

          body(i, j,
               D_i_block_key, D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank,
               D_block_j_key, D_block_j_rows, D_block_j_cols, D_block_j_row_rank, D_block_j_col_rank);
        }
      }
    }
  }
}

template<typename T> void
reduction_loop5(SymmetricSharedBasisMatrix& A,
                const Domain& domain,
                int64_t block, int64_t level, T&& body) {
  const int64_t nblocks = pow(2, level);
  for (int64_t i = block; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_i_block_key = parsec_D.super.data_key(&parsec_D.super, i, block, level);
      int64_t D_i_block_rows = get_dim(A, domain, i, level);
      int64_t D_i_block_cols = get_dim(A, domain, block, level);
      int64_t D_i_block_row_rank = A.ranks(i, level);
      int64_t D_i_block_col_rank = A.ranks(block, level);

      for (int64_t j = 0; j < block; ++j) {
        if (exists_and_inadmissible(A, block, j, level)) {
          auto    D_block_j_key = parsec_D.super.data_key(&parsec_D.super, block, j, level);
          int64_t D_block_j_rows = get_dim(A, domain, block, level);
          int64_t D_block_j_cols = get_dim(A, domain, j, level);
          int64_t D_block_j_row_rank = A.ranks(block, level);
          int64_t D_block_j_col_rank = A.ranks(j, level);

          body(i, j,
               D_i_block_key, D_i_block_rows, D_i_block_cols, D_i_block_row_rank, D_i_block_col_rank,
               D_block_j_key, D_block_j_rows, D_block_j_cols, D_block_j_row_rank, D_block_j_col_rank);
        }
      }
    }
  }
}

template <typename T> void
reduction_loop6(SymmetricSharedBasisMatrix& A, const Domain& domain, int64_t block, int64_t level,
                T&& body) {
  for (int64_t i = 0; i < block; ++i) {
    if (exists_and_inadmissible(A, block, i, level)) {
      auto    D_block_i_key = parsec_D.super.data_key(&parsec_D.super, block, i, level);
      int64_t D_block_i_rows = get_dim(A, domain, block, level);
      int64_t D_block_i_cols = get_dim(A, domain, i, level);
      int64_t D_block_i_row_rank = A.ranks(block, level);
      int64_t D_block_i_col_rank = A.ranks(i, level);

      body(i, D_block_i_key, D_block_i_rows,
           D_block_i_cols, D_block_i_row_rank, D_block_i_col_rank);
    }
  }
}


void partial_syrk(SymmetricSharedBasisMatrix& A,
                  const Domain& domain,
                  int64_t block,
                  int64_t row,
                  int64_t col,
                  int64_t level,
                  parsec_data_key_t D_i_block_key,
                  int64_t D_i_block_rows, int64_t D_i_block_cols,
                  int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                  int64_t D_i_block_split_index,
                  int64_t D_ij_split_index,
                  Hatrix::Mode uplo, bool unit_diag, bool flip_row_index=false) {
  if (exists_and_inadmissible(A, row, col, level)) {
    auto D_ij_key = parsec_D.super.data_key(&parsec_D.super, row, col, level);
    int64_t D_ij_rows = get_dim(A, domain, row, level);
    int64_t D_ij_cols = get_dim(A, domain, col, level);
    int64_t D_ij_row_rank = A.ranks(row, level);
    int64_t D_ij_col_rank = A.ranks(col, level);

    parsec_dtd_insert_task(dtd_tp, task_partial_syrk, 0, PARSEC_DEV_CPU,
      "partial_syrk_task",
      // D_i_block
      sizeof(int64_t), &D_i_block_rows, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_cols, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_row_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_col_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_split_index, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_i_block_key),
                           PARSEC_INPUT | arena_D(flip_row_index ? block : row, flip_row_index ? row : block, level),
      // D_ij
      sizeof(int64_t), &D_ij_rows, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_cols, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_row_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_col_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_split_index, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_ij_key), PARSEC_INOUT | arena_D(row, col, level) | PARSEC_AFFINITY,
      sizeof(Hatrix::Mode), &uplo, PARSEC_VALUE,
      sizeof(bool), &unit_diag, PARSEC_VALUE,
      PARSEC_DTD_ARG_END);
  }
}

void
partial_matmul(SymmetricSharedBasisMatrix& A,
               const Domain& domain,
               int64_t block,
               int64_t row,
               int64_t col,
               int64_t level,

               parsec_data_key_t D_i_block_key, int64_t D_i_block_rows, int64_t D_i_block_cols,
               int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
               int64_t D_i_block_split_index,

               parsec_data_key_t D_j_block_key, int64_t D_j_block_rows, int64_t D_j_block_cols,
               int64_t D_j_block_row_rank, int64_t D_j_block_col_rank,
               int64_t D_j_block_split_index, int64_t D_ij_split_index,
               bool transA, bool transB, bool flip_col_index=false) {
  if (exists_and_inadmissible(A, row, col, level)) {
    auto D_ij_key = parsec_D.super.data_key(&parsec_D.super, row, col, level);
    int64_t D_ij_rows = get_dim(A, domain, row, level);
    int64_t D_ij_cols = get_dim(A, domain, col, level);
    int64_t D_ij_row_rank = A.ranks(row, level);
    int64_t D_ij_col_rank = A.ranks(col, level);

    parsec_dtd_insert_task(dtd_tp, task_partial_matmul, 0, PARSEC_DEV_CPU,
      "partial_matmul_task",
      // D_i_block
      sizeof(int64_t), &D_i_block_rows, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_cols, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_row_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_col_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_i_block_split_index, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_i_block_key), PARSEC_INPUT | arena_D(row, block, level),
      // D_j_block
      sizeof(int64_t), &D_j_block_rows, PARSEC_VALUE,
      sizeof(int64_t), &D_j_block_cols, PARSEC_VALUE,
      sizeof(int64_t), &D_j_block_row_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_j_block_col_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_j_block_split_index, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_j_block_key),
                           PARSEC_INPUT | arena_D(flip_col_index ? block : col, flip_col_index ? col : block, level),
      // D_ij
      sizeof(int64_t), &D_ij_rows, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_cols, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_row_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_col_rank, PARSEC_VALUE,
      sizeof(int64_t), &D_ij_split_index, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_ij_key),
                           PARSEC_INOUT | arena_D(row, col, level) | PARSEC_AFFINITY,
      sizeof(bool), &transA, PARSEC_VALUE,
      sizeof(bool), &transB, PARSEC_VALUE,
      PARSEC_DTD_ARG_END);
  }
}

void
compute_schurs_complement(SymmetricSharedBasisMatrix& A,
                          const Domain& domain,
                          const int64_t block,
                          const int64_t level) {
  reduction_loop1(A, domain, block, level,
                  [&](int64_t i, int64_t j,
                      parsec_data_key_t D_i_block_key, int64_t D_i_block_rows, int64_t D_i_block_cols,
                      int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                      parsec_data_key_t D_j_block_key, int64_t D_j_block_rows, int64_t D_j_block_cols,
                      int64_t D_j_block_row_rank, int64_t D_j_block_col_rank) {
                    if (i == j) {
                      partial_syrk(A, domain, block, i, j, level,
                                   D_i_block_key,
                                   D_i_block_rows, D_i_block_cols,
                                   D_i_block_row_rank, D_i_block_col_rank,
                                   0, 0,
                                   Hatrix::Lower, false);
                    }
                    else {
                      partial_matmul(A, domain, block, i, j, level,
                                     D_i_block_key, D_i_block_rows, D_i_block_cols,
                                     D_i_block_row_rank, D_i_block_col_rank,
                                     0,
                                     D_j_block_key, D_j_block_rows, D_j_block_cols,
                                     D_j_block_row_rank, D_j_block_col_rank,
                                     0, 0, false, true);

                      if (i < j) {
                        partial_matmul(A, domain, block, i, j, level,
                                       D_i_block_key, D_i_block_rows, D_i_block_cols,
                                       D_i_block_row_rank, D_i_block_col_rank,
                                       2,
                                       D_j_block_key, D_j_block_rows, D_j_block_cols,
                                       D_j_block_row_rank, D_j_block_col_rank,
                                       0, 2, false, true);
                      }
                      else {
                        partial_matmul(A, domain, block, i, j, level,
                                       D_i_block_key, D_i_block_rows, D_i_block_cols,
                                       D_i_block_row_rank, D_i_block_col_rank,
                                       0,
                                       D_j_block_key, D_j_block_rows, D_j_block_cols,
                                       D_j_block_row_rank, D_j_block_col_rank,
                                       2, 1, false, true);
                      }
                    }
                  });

  reduction_loop2(A, domain, block, level,
                  [&](int64_t i, int64_t j,
                      parsec_data_key_t D_i_block_key, int64_t D_i_block_rows, int64_t D_i_block_cols,
                      int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                      parsec_data_key_t D_j_block_key, int64_t D_j_block_rows, int64_t D_j_block_cols,
                      int64_t D_j_block_row_rank, int64_t D_j_block_col_rank) {
                    if (i == j) {
                      partial_syrk(A, domain, block, i, j, level,
                                   D_i_block_key,
                                   D_i_block_rows, D_i_block_cols,
                                   D_i_block_row_rank, D_i_block_col_rank,
                                   2,
                                   3,
                                   Hatrix::Lower, false);
                    }
                    else {
                      partial_matmul(A, domain, block, i, j, level,
                                     D_i_block_key, D_i_block_rows, D_i_block_cols,
                                     D_i_block_row_rank, D_i_block_col_rank,
                                     2,
                                     D_j_block_key, D_j_block_rows, D_j_block_cols,
                                     D_j_block_row_rank, D_j_block_col_rank,
                                     2, 3, false, true);
                    }
                  });

  reduction_loop4(A, domain, block, level,
                  [&](int64_t i, int64_t j,
                      parsec_data_key_t D_i_block_key, int64_t D_i_block_rows,
                      int64_t D_i_block_cols,
                      int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                      parsec_data_key_t D_block_j_key, int64_t D_block_j_rows,
                      int64_t D_block_j_cols,
                      int64_t D_block_j_row_rank, int64_t D_block_j_col_rank) {
                    partial_matmul(A, domain, block, i, j, level,
                                   D_i_block_key, D_i_block_rows, D_i_block_cols,
                                   D_i_block_row_rank, D_i_block_col_rank,
                                   0,
                                   D_block_j_key, D_block_j_rows, D_block_j_cols,
                                   D_block_j_row_rank, D_block_j_col_rank,
                                   1,
                                   1, false, false, true);
                  });


  reduction_loop5(A, domain, block, level,
                  [&](int64_t i, int64_t j,
                      parsec_data_key_t D_i_block_key, int64_t D_i_block_rows,
                      int64_t D_i_block_cols,
                      int64_t D_i_block_row_rank, int64_t D_i_block_col_rank,
                      parsec_data_key_t D_j_block_key, int64_t D_j_block_rows,
                      int64_t D_j_block_cols,
                      int64_t D_j_block_row_rank, int64_t D_j_block_col_rank) {
                    partial_matmul(A, domain, block, i, j, level,
                                   D_i_block_key, D_i_block_rows, D_i_block_cols,
                                   D_i_block_row_rank, D_i_block_col_rank,
                                   2,
                                   D_j_block_key, D_j_block_rows, D_j_block_cols,
                                   D_j_block_row_rank, D_j_block_col_rank,
                                   1,
                                   3, false, false, true);
                  });

  reduction_loop6(A, domain, block, level,
                  [&](int64_t i,
                      parsec_data_key_t D_block_i_key, int64_t D_block_i_rows,
                      int64_t D_block_i_cols,
                      int64_t D_block_i_row_rank, int64_t D_block_i_col_rank) {
                    partial_syrk(A, domain, block, i, i, level,
                                 D_block_i_key, D_block_i_rows, D_block_i_cols,
                                 D_block_i_row_rank, D_block_i_col_rank,
                                 1,
                                 3,
                                 Hatrix::Lower, false, true);
                  });


  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
}

void
factorize_level(SymmetricSharedBasisMatrix& A,
                Hatrix::Domain& domain,
                int64_t level, RowColLevelMap<Matrix>& F,
                RowMap<Matrix>& r, RowMap<Matrix>& t,
                const Hatrix::Args& opts) {
  const int64_t nblocks = pow(2, level);
  for (int64_t block = 0; block < nblocks; ++block) {
    multiply_complements(A, domain, block, level);
    // factorize_diagonal(A, domain, block, level);
    // triangle_reduction(A, domain, block, level);
    // compute_schurs_complement(A, domain, block, level);
  }
}

void
preallocate_blocks(SymmetricSharedBasisMatrix& A) {
  for (int level = A.max_level - 1; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int child_level = level + 1;

    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (exists_and_inadmissible(A, i, j, level)) {
          if (mpi_rank(i, j) == MPIRANK) {
            std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});

            const int64_t c_rows = A.ranks(i_children[0], child_level) +
              A.ranks(i_children[1], child_level);
            const int64_t c_cols = A.ranks(j_children[0], child_level) +
              A.ranks(j_children[1], child_level);
            Matrix D_unelim(c_rows, c_cols);

            A.D.insert(i, j, level, std::move(D_unelim));
          }
        }
      }
    }
  }

  int level = A.min_level - 1;
  int child_level = level + 1;
  int nblocks = pow(2, level);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (mpi_rank(i, j) == MPIRANK) {
        std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});

        const int64_t c_rows = A.ranks(i_children[0], child_level) +
          A.ranks(i_children[1], child_level);
        const int64_t c_cols = A.ranks(j_children[0], child_level) +
          A.ranks(j_children[1], child_level);
        Matrix D_unelim(c_rows, c_cols);

        A.D.insert(i, j, level, std::move(D_unelim));
      }
    }
  }
}

void
update_parsec_pointers(SymmetricSharedBasisMatrix& A, const Domain& domain, int64_t level) {
  const int64_t nblocks = pow(2, level);
  const int64_t parent_level = level-1;
  const int64_t parent_nblocks = pow(2, parent_level);

  // setup pointers to data for use with parsec.
  for (int64_t i = 0; i < nblocks; ++i) { // U
    parsec_data_key_t U_data_key = parsec_U.super.data_key(&parsec_U.super, i, level);
    if (mpi_rank(i) == MPIRANK) {
      Matrix& U_i = A.U(i, level);
      parsec_U.matrix_map[U_data_key] = std::addressof(U_i); // TODO: how to do this in a C++-way?
    }
    parsec_U.mpi_ranks[U_data_key] = mpi_rank(i);

    int U_ARENA;
    int block_size = get_dim(A, domain, i, level), rank = A.ranks(i, level);

    parsec_arena_datatype_t* bases_t = parsec_dtd_create_arena_datatype(parsec, &U_ARENA);
    parsec_add2arena_rect(bases_t, parsec_datatype_double_t, block_size, rank, block_size);
    arena_U.insert(i, level, std::move(U_ARENA));
    bases_t = NULL;
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      int S_ARENA;
      int row_size = A.ranks(i, level), col_size = A.ranks(j, level);
      parsec_arena_datatype_t* bases_t = parsec_dtd_create_arena_datatype(parsec, &S_ARENA);
      parsec_add2arena_rect(bases_t, parsec_datatype_double_t, row_size, col_size, row_size);
      arena_S.insert(i, j, level, std::move(S_ARENA));
      parsec_data_key_t S_data_key = parsec_S.super.data_key(&parsec_S.super, i, j, level);
      bases_t = NULL;

      if (exists_and_admissible(A, i, j, level) && mpi_rank(i, j) == MPIRANK) {     // S blocks.
        Matrix& S_ij = A.S(i, j, level);
        parsec_S.matrix_map[S_data_key] = std::addressof(S_ij);
      }
      parsec_S.mpi_ranks[S_data_key] = mpi_rank(i, j);

      int D_ARENA;
      row_size = get_dim(A, domain, i, level), col_size = get_dim(A, domain, j, level);
      bases_t = parsec_dtd_create_arena_datatype(parsec, &D_ARENA);
      parsec_add2arena_rect(bases_t, parsec_datatype_double_t, row_size, col_size, row_size);
      arena_D.insert(i, j, level, std::move(D_ARENA));
      parsec_data_key_t D_data_key = parsec_D.super.data_key(&parsec_D.super, i, j, level);
      parsec_D.mpi_ranks[D_data_key] = mpi_rank(i, j);
      bases_t = NULL;

      if (exists_and_inadmissible(A, i, j, level) && (mpi_rank(i, j) == MPIRANK)) { // D blocks.
        Matrix& D_ij = A.D(i, j, level);
        parsec_D.matrix_map[D_data_key] = std::addressof(D_ij);
      }
    }
  }
}

void h2_dc_init_maps() {
  h2_dc_init(parsec_U, data_key_1d, rank_of_1d);
  h2_dc_init(parsec_S, data_key_2d, rank_of_2d);
  h2_dc_init(parsec_D, data_key_2d, rank_of_2d);
}

void h2_dc_destroy_maps() {
  h2_dc_destroy(parsec_U);
  h2_dc_destroy(parsec_S);
  h2_dc_destroy(parsec_D);
}

void h2_destroy_arenas(int64_t max_level, int64_t min_level) {
  for (int64_t level = max_level; level >= min_level-1; --level) {
    const int64_t nblocks = pow(2, level);
    for (int64_t i = 0; i < nblocks; ++i) { //
      parsec_dtd_destroy_arena_datatype(parsec, arena_U(i, level));
      for (int64_t j = 0; j <= i; ++j) {
        parsec_dtd_destroy_arena_datatype(parsec, arena_D(i, j, level));
        parsec_dtd_destroy_arena_datatype(parsec, arena_S(i, j, level));
      }
    }
  }
}

// This function is meant to be used with parsec. It will register the
// data that has already been allocated by the matrix with parsec and
// make it work with the runtime.
void factorize(SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain, const Hatrix::Args& opts) {
  int64_t level;
  RowColLevelMap<Matrix> F;
  RowMap<Matrix> r, t;

  h2_dc_init_maps();

  preallocate_blocks(A);
  update_parsec_pointers(A, domain, A.max_level);

  for (level = A.max_level; level >= A.max_level; --level) {
    factorize_level(A, domain, level, F, r, t, opts);
    // update_parsec_pointers(A, domain, level-1);
    // merge_unfactorized_blocks(A, domain, level);
  }

  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_S.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_U.super);


  int rc = parsec_dtd_taskpool_wait(dtd_tp);
  PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

  // h2_dc_destroy_maps();
  // h2_destroy_arenas(A.max_level, A.min_level);
}

void solve(SymmetricSharedBasisMatrix& A, std::vector<Matrix>& x, std::vector<Matrix>& h2_solution) {
  std::vector<Matrix> b(x);

}
