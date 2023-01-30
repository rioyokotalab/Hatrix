#include <random>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "globals.hpp"
#include "h2_tasks.hpp"
#include "h2_dtd_operations.hpp"

using namespace Hatrix;

static h2_dc_t parsec_U, parsec_S, parsec_D;
static int U_ARENA, D_ARENA, S_ARENA, FINAL_DENSE_ARENA, U_NON_LEAF_ARENA;

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

  // std::cout << "   !! DATA KEY : -> "  << data << std::endl;

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
  if (parsec_data.data_map.size()) {
    parsec_dtd_data_collection_fini(o);
    for (auto iter = parsec_data.data_map.begin(); iter != parsec_data.data_map.end(); ++iter) {
      parsec_data_destroy((iter->second));
    }

    parsec_data_collection_destroy(o);
  }
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

std::vector<int> MERGE_ARENAS;
parsec_arena_datatype_t* merge_arena_t;

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

            if (exists_and_inadmissible(A, c1, c2, level)) {
              // copy oo portion of the D blocks.
              copy_dense = true;
              parsec_data_key_t D_unelim_key =
                parsec_D.super.data_key(&parsec_D.super, i, parent_level);
              parsec_data_key_t D_c1c2_key =
                parsec_D.super.data_key(&parsec_D.super, c1, level);

              int write_arena = A.max_level == parent_level ? D_ARENA : FINAL_DENSE_ARENA;
              int read_arena = A.max_level == level ? D_ARENA : FINAL_DENSE_ARENA;

              parsec_dtd_insert_task(dtd_tp, task_copy_blocks, 90, PARSEC_DEV_CPU,
                "copy_blocks_task",
                sizeof(bool), &copy_dense, PARSEC_VALUE,
                PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_unelim_key),
                                     PARSEC_INOUT | write_arena | PARSEC_AFFINITY,
                sizeof(int64_t), &D_unelim_rows, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_cols, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_row_rank, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_col_rank, PARSEC_VALUE,
                PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_c1c2_key),
                                     PARSEC_INPUT | read_arena,
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
              parsec_data_key_t D_unelim_key =
                parsec_D.super.data_key(&parsec_D.super, i, parent_level);
              parsec_data_key_t  S_c1c2_key =
                parsec_S.super.data_key(&parsec_S.super, c1, c2, level);
              int64_t S_c1c2_rows = A.ranks(c1, level);
              int64_t S_c1c2_cols = A.ranks(c2, level);
              int64_t MINUS_ONE = -1;
              int write_arena = A.max_level == parent_level ? D_ARENA : FINAL_DENSE_ARENA;

              parsec_dtd_insert_task(dtd_tp, task_copy_blocks, 80, PARSEC_DEV_CPU,
                "copy_blocks_task",
                sizeof(bool), &copy_dense, PARSEC_VALUE,
                PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_unelim_key),
                                     PARSEC_INOUT | write_arena | PARSEC_AFFINITY,
                sizeof(int64_t), &D_unelim_rows, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_cols, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_row_rank, PARSEC_VALUE,
                sizeof(int64_t), &D_unelim_col_rank, PARSEC_VALUE,
                PASSED_BY_REF, parsec_dtd_tile_of(&parsec_S.super, S_c1c2_key),
                                     PARSEC_INPUT | S_ARENA,
                sizeof(int64_t), &S_c1c2_rows, PARSEC_VALUE,
                sizeof(int64_t), &S_c1c2_cols, PARSEC_VALUE,
                sizeof(int64_t), &MINUS_ONE, PARSEC_VALUE,
                sizeof(int64_t), &MINUS_ONE, PARSEC_VALUE,
                sizeof(int), &D_unelim_split_index, PARSEC_VALUE,
                PARSEC_DTD_ARG_END);
            }
          } // for ic1
        }   // for jc2
      }   // if exists and inadmissible
    }   // for j
  }   // for i
}

void
multiply_complements(SymmetricSharedBasisMatrix& A,
                     Domain& domain,
                     const int64_t block, const int64_t level) {
  int64_t nblocks = pow(2, level);

  parsec_data_key_t U_key = parsec_U.super.data_key(&parsec_U.super, block, level);
  parsec_data_key_t D_key = parsec_D.super.data_key(&parsec_D.super, block, level);

  int64_t D_nrows = get_dim(A, domain, block, level), U_nrows = get_dim(A, domain, block, level);
  int64_t D_ncols = get_dim(A, domain, block, level), U_ncols = A.ranks(block, level);

  int u_arena_type = A.max_level == level ? U_ARENA : U_NON_LEAF_ARENA;
  int d_arena_type = A.max_level == level ? D_ARENA : FINAL_DENSE_ARENA;

  parsec_dtd_insert_task(dtd_tp, task_multiply_full_complement, 100, PARSEC_DEV_CPU,
    "multiply_full_complement_task",
    PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_key),
                         PARSEC_INOUT | d_arena_type | PARSEC_AFFINITY,
    sizeof(int64_t), &D_nrows, PARSEC_VALUE,
    sizeof(int64_t), &D_ncols, PARSEC_VALUE,
    PASSED_BY_REF, parsec_dtd_tile_of(&parsec_U.super, U_key),
                         PARSEC_INPUT | u_arena_type,
    sizeof(int64_t), &U_nrows, PARSEC_VALUE,
    sizeof(int64_t), &U_ncols, PARSEC_VALUE,
    PARSEC_DTD_ARG_END);
}

void factorize_diagonal(SymmetricSharedBasisMatrix& A,
                        const Domain& domain,
                        const int64_t block,
                        const int64_t level) {
  int64_t D_nrows = get_dim(A, domain, block, level);
  int64_t rank_nrows = A.ranks(block, level);
  auto D_key = parsec_D.super.data_key(&parsec_D.super, block, level);

  int write_arena = A.max_level == level ? D_ARENA : FINAL_DENSE_ARENA;

  parsec_dtd_insert_task(dtd_tp, task_factorize_diagonal, 90, PARSEC_DEV_CPU,
    "factorize_diagonal_task",
    sizeof(int64_t), &D_nrows, PARSEC_VALUE,
    sizeof(int64_t), &rank_nrows, PARSEC_VALUE,
    PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_key),
                         PARSEC_INOUT | write_arena | PARSEC_AFFINITY,
    PARSEC_DTD_ARG_END);
}

void
factorize_level(SymmetricSharedBasisMatrix& A,
                Hatrix::Domain& domain,
                int64_t level,
                const Hatrix::Args& opts) {
  const int64_t nblocks = pow(2, level);
  for (int64_t block = 0; block < nblocks; ++block) {
    multiply_complements(A, domain, block, level);
    factorize_diagonal(A, domain, block, level);
  }
}

void
preallocate_blocks(SymmetricSharedBasisMatrix& A) {
  for (int level = A.max_level - 1; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int child_level = level + 1;

    // D blocks for holding merged blocks on level.
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (exists_and_inadmissible(A, i, j, level)) {
          if (mpi_rank(i) == MPIRANK) {
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
      if (mpi_rank(i) == MPIRANK) {
        std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});

        const int64_t c_rows = A.ranks(i_children[0], child_level) +
          A.ranks(i_children[1], child_level);
        const int64_t c_cols = A.ranks(j_children[0], child_level) +
          A.ranks(j_children[1], child_level);
        Matrix D_unelim(c_rows, c_cols);
        A.D.insert(i, i, level, std::move(D_unelim));
      }
    }
  }
}

void
update_parsec_pointers(SymmetricSharedBasisMatrix& A, const Domain& domain, int64_t level) {
  const int64_t nblocks = pow(2, level);

  // setup pointers to data for use with parsec.
  for (int64_t i = 0; i < nblocks; ++i) { // U
    parsec_data_key_t U_data_key = parsec_U.super.data_key(&parsec_U.super, i, level);

    if (mpi_rank(i) == MPIRANK) {
      Matrix& U_i = A.U(i, level);
      parsec_U.matrix_map[U_data_key] = std::addressof(U_i);

    }
    parsec_U.mpi_ranks[U_data_key] = mpi_rank(i);
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      int row_size = A.ranks(i, level), col_size = A.ranks(j, level);
      parsec_data_key_t S_data_key = parsec_S.super.data_key(&parsec_S.super, i, j, level);

      if (exists_and_admissible(A, i, j, level) && mpi_rank(i, j) == MPIRANK) {     // S blocks.
        Matrix& S_ij = A.S(i, j, level);
        parsec_S.matrix_map[S_data_key] = std::addressof(S_ij);
      }
      parsec_S.mpi_ranks[S_data_key] = mpi_rank(i, j);

      row_size = get_dim(A, domain, i, level), col_size = get_dim(A, domain, j, level);
      parsec_data_key_t D_data_key = parsec_D.super.data_key(&parsec_D.super, i, level);
      parsec_D.mpi_ranks[D_data_key] = mpi_rank(i);

      if (exists_and_inadmissible(A, i, j, level) && (mpi_rank(i) == MPIRANK)) { // D blocks.
        Matrix& D_ij = A.D(i, i, level);
        parsec_D.matrix_map[D_data_key] = std::addressof(D_ij);
      }
    }
  }
}


void h2_dc_init_maps() {
  h2_dc_init(parsec_U, data_key_1d, rank_of_1d);
  h2_dc_init(parsec_S, data_key_2d, rank_of_2d);
  h2_dc_init(parsec_D, data_key_1d, rank_of_1d);
}

void
h2_dc_destroy_maps() {
  h2_dc_destroy(parsec_U);
  h2_dc_destroy(parsec_S);
  h2_dc_destroy(parsec_D);
}

void
final_dense_factorize(SymmetricSharedBasisMatrix& A,
                      const Hatrix::Domain& domain,
                      const Hatrix::Args& opts,
                      const int64_t level) {
  const int64_t nblocks = pow(2, level);

  for (int64_t d = 0; d < nblocks; ++d) {
    int64_t D_dd_nrows = get_dim(A, domain, d, level);
    int64_t D_dd_ncols = get_dim(A, domain, d, level);
    parsec_data_key_t D_dd_key = parsec_D.super.data_key(&parsec_D.super, d, level);

    parsec_dtd_insert_task(dtd_tp, task_cholesky_full, 0, PARSEC_DEV_CPU,
      "full_cholesky_task",
      sizeof(int64_t), &D_dd_nrows, PARSEC_VALUE,
      sizeof(int64_t), &D_dd_ncols, PARSEC_VALUE,
      PASSED_BY_REF, parsec_dtd_tile_of(&parsec_D.super, D_dd_key),
                           PARSEC_INOUT | FINAL_DENSE_ARENA | PARSEC_AFFINITY,
      PARSEC_DTD_ARG_END);

  }
}

void
factorize_setup(SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                const Hatrix::Args& opts) {
  parsec_arena_datatype_t* u_arena_t = parsec_dtd_create_arena_datatype(parsec, &U_ARENA);
  parsec_add2arena_rect(u_arena_t, parsec_datatype_double_t, opts.nleaf, opts.max_rank, opts.nleaf);

  parsec_arena_datatype_t* d_arena_t = parsec_dtd_create_arena_datatype(parsec, &D_ARENA);
  parsec_add2arena_rect(d_arena_t, parsec_datatype_double_t, opts.nleaf, opts.nleaf, opts.nleaf);

  parsec_arena_datatype_t* s_arena_t = parsec_dtd_create_arena_datatype(parsec, &S_ARENA);
  parsec_add2arena_rect(s_arena_t, parsec_datatype_double_t, opts.max_rank, opts.max_rank,
                        opts.max_rank);

  parsec_arena_datatype_t* final_dense_arena_t =
    parsec_dtd_create_arena_datatype(parsec, &FINAL_DENSE_ARENA);
  parsec_add2arena_rect(final_dense_arena_t, parsec_datatype_double_t,
                        opts.max_rank * 2, opts.max_rank * 2,
                        opts.max_rank * 2);

  parsec_arena_datatype_t* u_non_leaf_arena_t =
    parsec_dtd_create_arena_datatype(parsec, &U_NON_LEAF_ARENA);
  parsec_add2arena_rect(u_non_leaf_arena_t, parsec_datatype_double_t,
                        opts.max_rank * 2, opts.max_rank,
                        opts.max_rank * 2);

  h2_dc_init_maps();

  preallocate_blocks(A);
  update_parsec_pointers(A, domain, A.max_level);

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    update_parsec_pointers(A, domain, level-1);
  }
}

void factorize_teardown() {
  h2_dc_destroy_maps();
  parsec_dtd_destroy_arena_datatype(parsec, U_ARENA);
  parsec_dtd_destroy_arena_datatype(parsec, D_ARENA);
  parsec_dtd_destroy_arena_datatype(parsec, S_ARENA);
  parsec_dtd_destroy_arena_datatype(parsec, FINAL_DENSE_ARENA);
  parsec_dtd_destroy_arena_datatype(parsec, U_NON_LEAF_ARENA);

}


// This function is meant to be used with parsec. It will register the
// data that has already been allocated by the matrix with parsec and
// make it work with the runtime.
long long int
factorize(SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain, const Hatrix::Args& opts) {
  int64_t level;

  for (level = A.max_level; level >= A.min_level; --level) {
    factorize_level(A, domain, level, opts);

    merge_unfactorized_blocks(A, domain, level);
  }

  final_dense_factorize(A, domain, opts, level);

  parsec_dtd_data_flush_all(dtd_tp, &parsec_D.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_S.super);
  parsec_dtd_data_flush_all(dtd_tp, &parsec_U.super);

  int rc = parsec_taskpool_wait(dtd_tp);
  PARSEC_CHECK_ERROR(rc, "parsec_dtd_taskpool_wait");

  return 0;
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
    if (mpi_rank(block, block) == MPIRANK) {
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
        if (mpi_rank(block, irow) == MPIRANK) {
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
                   mpi_rank(block, irow), irow, MPI_COMM_WORLD, &status);

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
        if (mpi_rank(irow, block) == MPIRANK) {
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
                   mpi_rank(irow, block), irow, MPI_COMM_WORLD, &status);

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
        if (mpi_rank(block, icol) == MPIRANK) {
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
                   mpi_rank(block, icol), icol, MPI_COMM_WORLD, &status);

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
        if (mpi_rank(icol, block) == MPIRANK) {
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
                   mpi_rank(icol, block), block, MPI_COMM_WORLD, &status);

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

    if (mpi_rank(block, block) == MPIRANK) {
      MPI_Request request;

      MPI_Isend(&A.D(block, block, level), A.D(block, block, level).numel(), MPI_DOUBLE,
                mpi_rank(block), block, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(block) == MPIRANK) {
      MPI_Status status;

      Matrix D_copy(get_dim(A, domain, block, level), get_dim(A, domain, block, level));
      MPI_Recv(&D_copy, D_copy.numel(), MPI_DOUBLE,
               mpi_rank(block, block), block, MPI_COMM_WORLD, &status);
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
      if (mpi_rank(i, j) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&A.D(i, j, level), A.D(i, j, level).numel(), MPI_DOUBLE,
                  mpi_rank(i), dense_block_tag, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(j) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&x_last[j_index], x_last[j_index].numel(), MPI_DOUBLE,
                  mpi_rank(i), j, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(i) == MPIRANK) {
        MPI_Status status;
        int64_t D_ij_nrows = get_dim(A, domain, i, level);
        int64_t D_ij_ncols = get_dim(A, domain, j, level);

        Matrix D_ij(D_ij_nrows, D_ij_ncols);
        MPI_Recv(&D_ij, D_ij.numel(), MPI_DOUBLE,
                 mpi_rank(i, j), dense_block_tag, MPI_COMM_WORLD, &status);

        Matrix x_last_j(D_ij_ncols, 1);
        MPI_Recv(&x_last_j, x_last_j.numel(), MPI_DOUBLE,
                 mpi_rank(j), j, MPI_COMM_WORLD, &status);

        matmul(D_ij, x_last_j, x_last[i_index], false, false, -1, 1);
      }
    }

    // diagonal block
    if (mpi_rank(i, i) == MPIRANK) {
      Matrix& D_ii = A.D(i, i, level);
      MPI_Request request;
      MPI_Isend(&D_ii, D_ii.numel(), MPI_DOUBLE,
                mpi_rank(i), i, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(i) == MPIRANK) {
      MPI_Status status;
      Matrix D_ii(get_dim(A, domain, i, level), get_dim(A, domain, i, level));

      MPI_Recv(&D_ii, D_ii.numel(), MPI_DOUBLE,
               mpi_rank(i, i), i, MPI_COMM_WORLD, &status);

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
      if (mpi_rank(i, j) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&A.D(i, j, level), A.D(i, j, level).numel(), MPI_DOUBLE,
                  mpi_rank(j), dense_block_tag, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(i) == MPIRANK) {
        MPI_Request request;
        MPI_Isend(&x_last[i_index], x_last[i_index].numel(), MPI_DOUBLE,
                  mpi_rank(j), i, MPI_COMM_WORLD, &request);
      }

      if (mpi_rank(j) == MPIRANK) {
        MPI_Status status;
        int64_t D_ij_nrows = get_dim(A, domain, i, level);
        int64_t D_ij_ncols = get_dim(A, domain, j, level);

        Matrix D_ij(D_ij_nrows, D_ij_ncols);
        MPI_Recv(&D_ij, D_ij.numel(), MPI_DOUBLE,
                 mpi_rank(i, j), dense_block_tag, MPI_COMM_WORLD, &status);

        Matrix x_last_i(D_ij_nrows, 1);
        MPI_Recv(&x_last_i, x_last_i.numel(), MPI_DOUBLE,
                 mpi_rank(i), i, MPI_COMM_WORLD, &status);

        matmul(D_ij, x_last_i, x_last[j_index], true, false, -1.0, 1.0);
      }
    }

    if (mpi_rank(j, j) == MPIRANK) {
      Matrix& D_jj = A.D(j, j, level);
      MPI_Request request;
      MPI_Isend(&D_jj, D_jj.numel(), MPI_DOUBLE,
                mpi_rank(j), j, MPI_COMM_WORLD, &request);
    }

    if (mpi_rank(j) == MPIRANK) {
      MPI_Status status;
      Matrix D_jj(get_dim(A, domain, j, level), get_dim(A, domain, j, level));

      MPI_Recv(&D_jj, D_jj.numel(), MPI_DOUBLE,
               mpi_rank(j, j), j, MPI_COMM_WORLD, &status);

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
}
