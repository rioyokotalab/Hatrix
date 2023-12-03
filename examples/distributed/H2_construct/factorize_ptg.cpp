#include <cmath>
#include <mutex>
#include <algorithm>

#include "factorize_ptg.hpp"
#include "ptg_interfaces.h"

using namespace Hatrix;

parsec_context_t* parsec = NULL;
H2_ptg_t DENSE_BLOCKS, NON_LEAF_DENSE_BLOCKS, BASES_BLOCKS, S_BLOCKS;

// Global copy of admissibility for use with parsec.
RowColLevelMap<bool> global_is_admissible;

int64_t
near_neighbours_size(int64_t node, int64_t level) {
  return near_neighbours(node, level).size();
}

int64_t
near_neighbours_index(int64_t node, int64_t level, int64_t array_loc) {
  return near_neighbours(node, level)[array_loc];
}

int64_t
near_neighbours_reverse_index(int64_t node, int64_t level, int64_t index) {
  const std::vector<int64_t>& col_list = near_neighbours(node, level);
  return std::distance(col_list.begin(),
                       std::find(col_list.begin(), col_list.end(), index));
}

int64_t
far_neighbours_size(int64_t node, int64_t level) {
  return far_neighbours(node, level).size();
}

int64_t
far_neighbours_index(int64_t node, int64_t level, int64_t array_loc) {
  return far_neighbours(node, level)[array_loc];
}

int64_t
far_neighbours_reverse_index(int64_t node, int64_t level, int64_t index) {
  const std::vector<int64_t>& col_list = far_neighbours(node, level);
  return std::distance(col_list.begin(),
                       std::find(col_list.begin(), col_list.end(), index));
}

int is_admissible(int64_t row, int64_t col, int64_t level) {
  return global_is_admissible.exists(row, col, level) &&
    global_is_admissible(row, col, level);
}

int is_inadmissible(int64_t row, int64_t col, int64_t level) {
  return global_is_admissible.exists(row, col, level) &&
    !global_is_admissible(row, col, level);
}

parsec_data_key_t
block_key(int64_t row, int64_t col, int64_t level) {
  parsec_data_key_t level_offset = 0;
  for (int64_t l = 0; l < level; ++l) {
    level_offset += pow(pow(2, l), 2);
  }
  int64_t stride = pow(2, level);

  return level_offset + row * stride + col;
}

parsec_data_key_t
bases_key(int64_t row, int64_t level) {
  parsec_data_key_t level_offset = 0;
  for (int64_t l = 0; l < level; ++l) {
    level_offset += pow(2, l);
  }

  return level_offset + row;
}

uint32_t
rank_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  H2_ptg_t *dc = (H2_ptg_t*)desc;
  return (uint32_t)dc->mpi_rank_map[key];
}

std::mutex map_lock;
// NOTE: This function is called only by a process that own the data with key 'key'.
// So it will not return a NULL. Ever.
parsec_data_t*
data_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  H2_ptg_t *dc = (H2_ptg_t*)desc;
  parsec_data_t *data = nullptr;
  {
    std::lock_guard<std::mutex> guard(map_lock);

    if (dc->data_map.find(key) != dc->data_map.end()) {
      data = dc->data_map[key];
    }

    if (data == nullptr) {
      Matrix * matrix = dc->matrix_map[key];
      data = parsec_data_create(&data,
                                desc,
                                key,
                                matrix->data_ptr,
                                matrix->numel() * parsec_datadist_getsizeoftype(PARSEC_MATRIX_DOUBLE),
                                PARSEC_DATA_FLAG_PARSEC_MANAGED);

      // Scope for the lock_guard to be created and destroyed by itself.
      dc->data_map[key] = data;
    }
  }

  return data;
}

parsec_data_t*
data_of_bases(parsec_data_collection_t* desc, ...) {
  H2_ptg_t *dc = (H2_ptg_t*)desc;

  int row, level;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return data_of_key(desc, bases_key(row, level));
}

parsec_data_t*
data_of_block(parsec_data_collection_t* desc, ...) {
  H2_ptg_t *dc = (H2_ptg_t*)desc;

  int row, col, level;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  col = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return data_of_key(desc, block_key(row, col, level));
}

parsec_data_key_t
data_key_block(parsec_data_collection_t* desc, ...) {
  int row, col, level;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  col = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return block_key(row, col, level);
}

parsec_data_key_t
data_key_bases(parsec_data_collection_t* desc, ...) {
  int row, level;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return bases_key(row, level);
}

uint32_t rank_of(parsec_data_collection_t* desc, ...) {
  int row;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  va_end(ap);

  return mpi_rank(row);
}

int32_t vpid_of(parsec_data_collection_t* desc, ...) {
  return 0;
}

int32_t vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key) {
  return 0;
}

void
init_generic_block_dc(H2_ptg_t& parsec_data) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->data_key = data_key_block;

  o->rank_of = rank_of;
  o->rank_of_key = rank_of_key;

  o->data_of = data_of_block;
  o->data_of_key = data_of_key;

  o->vpid_of = vpid_of;
  o->vpid_of_key = vpid_of_key;
}

void
init_block_dc(H2_ptg_t& parsec_data, const Hatrix::Args& opts) {
  init_generic_block_dc(parsec_data);
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_datatype_t LEAF_BLOCK_TYPE;
  parsec_type_create_contiguous(opts.nleaf * opts.nleaf,
                                parsec_datatype_double_t,
                                &LEAF_BLOCK_TYPE);
  o->default_dtt = LEAF_BLOCK_TYPE;
}

void
init_non_leaf_block_dc(H2_ptg_t& parsec_data, const Hatrix::Args& opts) {
  init_generic_block_dc(parsec_data);
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_datatype_t NON_LEAF_BLOCK_TYPE;
  parsec_type_create_contiguous((opts.max_rank * 2) * (opts.max_rank * 2),
                                parsec_datatype_double_t,
                                &NON_LEAF_BLOCK_TYPE);
  o->default_dtt = NON_LEAF_BLOCK_TYPE;
}

void
init_s_block_dc(H2_ptg_t& parsec_data, const Hatrix::Args& opts) {
  init_generic_block_dc(parsec_data);
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_datatype_t S_BLOCK_TYPE;
  parsec_type_create_contiguous(opts.max_rank * opts.max_rank,
                                parsec_datatype_double_t,
                                &S_BLOCK_TYPE);
  o->default_dtt = S_BLOCK_TYPE;
}

void
init_basis_dc(H2_ptg_t& parsec_data, const Args& opts) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->data_key = data_key_bases;

  o->rank_of = rank_of;
  o->rank_of_key = rank_of_key;

  o->data_of = data_of_bases;
  o->data_of_key = data_of_key;

  o->vpid_of = vpid_of;
  o->vpid_of_key = vpid_of_key;

  parsec_datatype_t LEAF_BASES_BLOCK_TYPE;
  parsec_type_create_contiguous(opts.nleaf * opts.max_rank,
                                parsec_datatype_double_t,
                                &LEAF_BASES_BLOCK_TYPE);
  o->default_dtt = LEAF_BASES_BLOCK_TYPE;
}

parsec_h2_factorize_flows_taskpool_t *
h2_factorize_ptg_New(SymmetricSharedBasisMatrix& A,
                     const Domain& domain,
                     const Args& opts) {
  // Populate DENSE_BLOCKS struct with pointers of the leaf level dense blocks.
  init_block_dc(DENSE_BLOCKS, opts);
  int64_t nblocks = pow(2, A.max_level);
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    for (int64_t j = 0; j < near_neighbours_size(i, A.max_level); ++j) {
      int64_t index_j = near_neighbours_index(i, A.max_level, j);
      parsec_data_key_t key = block_key(i, index_j, A.max_level);
      DENSE_BLOCKS.matrix_map[key] =
        std::addressof(A.D(i, index_j, A.max_level));
      DENSE_BLOCKS.mpi_rank_map[key] = mpi_rank(i);
    }
  }

  // Populate non-leaf dense blocks with pointers of the non-leaf dense blocks.
  init_non_leaf_block_dc(NON_LEAF_DENSE_BLOCKS, opts);
  for (int64_t level = A.max_level - 1; level >= A.min_level - 1; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      for (int64_t j = 0; j < near_neighbours_size(i, level); ++j) {
        int64_t index_j = near_neighbours_index(i, level, j);
        parsec_data_key_t key = block_key(i, index_j, level);
        NON_LEAF_DENSE_BLOCKS.matrix_map[key] =
          std::addressof(A.D(i, index_j, level));
        NON_LEAF_DENSE_BLOCKS.mpi_rank_map[key] = mpi_rank(i);
      }
    }
  }

  // Populate leaf-level bases with pointers of the leaf level bases.
  init_basis_dc(BASES_BLOCKS, opts);
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t node = MPIRANK; node < nblocks; node += MPISIZE) {
      parsec_data_key_t key = bases_key(node, level);
      BASES_BLOCKS.matrix_map[key] =
        std::addressof(A.U(node, level));
      BASES_BLOCKS.mpi_rank_map[key] = mpi_rank(node);
    }
  }

  // Populate S_BLOCKS data with pointes to the skeleton blocks.
  init_s_block_dc(S_BLOCKS, opts);
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      for (int64_t j = 0; j < far_neighbours_size(i, level); ++j) {
        int64_t index_j = far_neighbours_index(i, level, j);
        parsec_data_key_t key = block_key(i, index_j, level);
        S_BLOCKS.matrix_map[key] =
          std::addressof(A.S(i, index_j, level));
        S_BLOCKS.mpi_rank_map[key] = mpi_rank(i);
      }
    }
  }

  parsec_data_collection_t* DENSE_BLOCKS_dc = &DENSE_BLOCKS.super;
  parsec_data_collection_t* NON_LEAF_DENSE_BLOCKS_dc = &NON_LEAF_DENSE_BLOCKS.super;
  parsec_data_collection_t* BASES_BLOCKS_dc = &BASES_BLOCKS.super;
  parsec_data_collection_t* S_BLOCKS_dc = &S_BLOCKS.super;

  parsec_h2_factorize_flows_taskpool_t *taskpool =
    parsec_h2_factorize_flows_new(BASES_BLOCKS_dc,
                                  NON_LEAF_DENSE_BLOCKS_dc,
                                  DENSE_BLOCKS_dc,
                                  S_BLOCKS_dc,
                                  opts.nleaf,
                                  opts.max_rank,
                                  A.max_level,
                                  A.min_level);

  // Create arena type for the leaf dense block.
  // parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_ADT_IDX],
  //                  parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
  //                  opts.nleaf, opts.nleaf, opts.nleaf,
  //                  PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // Create arena type for the leaf bases block.
  // parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_LEAF_BASES_ADT_IDX],
  //                  parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
  //                  opts.nleaf, opts.max_rank, opts.nleaf,
  //                  PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // Create arena type for skeleton blocks.
  // parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_S_BLOCK_TILE_ADT_IDX],
  //                  parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
  //                  opts.max_rank, opts.max_rank, opts.max_rank,
  //                  PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // Create arena type for the non-leaf dense block.
  parsec_add2arena(&taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_NON_LEAF_TILE_ADT_IDX],
                   parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
                   opts.max_rank * 2, opts.max_rank * 2, opts.max_rank * 2,
                   PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // Set the default data type to the leaf dense block.
  taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_DEFAULT_ADT_IDX].opaque_dtt =
      DENSE_BLOCKS_dc->default_dtt;
  taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_DEFAULT_ADT_IDX].arena =
    PARSEC_OBJ_NEW(parsec_arena_t);
  parsec_arena_construct(taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_DEFAULT_ADT_IDX].arena,
                         opts.nleaf * opts.nleaf,
                         PARSEC_ARENA_ALIGNMENT_SSE);

  // TILE_LOWER_RIGHT_CORNER type
  // MPI_Datatype MPI_TILE_LOWER_RIGHT_CORNER;
  // int nleaf = opts.nleaf, max_rank = opts.max_rank;
  // const int array_of_sizes_br[2] = {nleaf, nleaf};
  // const int array_of_subsizes_br[2] = {max_rank, max_rank};
  // const int array_of_starts_br[2] =
  //   {nleaf - max_rank, nleaf - max_rank};
  // MPI_Type_create_subarray(2,
  //                          array_of_sizes_br,
  //                          array_of_subsizes_br,
  //                          array_of_starts_br,
  //                          MPI_ORDER_FORTRAN,
  //                          MPI_DOUBLE,
  //                          &MPI_TILE_LOWER_RIGHT_CORNER);
  // MPI_Type_commit(&MPI_TILE_LOWER_RIGHT_CORNER);
  // taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_LOWER_RIGHT_CORNER_ADT_IDX].opaque_dtt =
  //   MPI_TILE_LOWER_RIGHT_CORNER;
  // taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_LOWER_RIGHT_CORNER_ADT_IDX].arena =
  //   PARSEC_OBJ_NEW(parsec_arena_t);
  // parsec_arena_construct(taskpool->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_LOWER_RIGHT_CORNER_ADT_IDX].arena,
  //                        max_rank * max_rank,
  //                        PARSEC_ARENA_ALIGNMENT_SSE);

  return taskpool;
}

void clear_maps(SymmetricSharedBasisMatrix& A, H2_ptg_t& map) {
  map.matrix_map.clear();
  map.data_map.clear();
  map.mpi_rank_map.clear();
  parsec_data_collection_t *dc = &map.super;
  parsec_data_collection_destroy(dc);
}

void
factorize_ptg(SymmetricSharedBasisMatrix& A,
              const Domain& domain,
              const Args& opts) {
  auto start_ptg_setup = std::chrono::system_clock::now();
  parsec_h2_factorize_flows_taskpool_t *tp = h2_factorize_ptg_New(A, domain, opts);
  parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)tp);
  auto stop_ptg_setup = std::chrono::system_clock::now();
  double ptg_setup_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_ptg_setup - start_ptg_setup).count();
  std::cout << "### setup ptg: " << ptg_setup_time << std::endl;

  parsec_context_start(parsec);
  parsec_context_wait(parsec);

  auto start_ptg_teardown = std::chrono::system_clock::now();

  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_ADT_IDX]);
  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_NON_LEAF_TILE_ADT_IDX]);
  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_DEFAULT_ADT_IDX]);

  parsec_taskpool_free((parsec_taskpool_t*)tp);


  // clear_maps(A, DENSE_BLOCKS);
  // clear_maps(A, NON_LEAF_DENSE_BLOCKS);
  // clear_maps(A, BASES_BLOCKS);
  // clear_maps(A, S_BLOCKS);

  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_TILE_LOWER_RIGHT_CORNER_ADT_IDX]);
  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_LEAF_BASES_ADT_IDX]);
  // parsec_del2arena(&tp->arenas_datatypes[PARSEC_h2_factorize_flows_S_BLOCK_TILE_ADT_IDX]);

  // This has to be below taskpool_free cuz that function frees the parsec data handles.


  int64_t nblocks = pow(2, A.max_level);
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    for (int64_t j = 0; j < near_neighbours_size(i, A.max_level); ++j) {
      int64_t index_j = near_neighbours_index(i, A.max_level, j);
      parsec_data_key_t key = block_key(i, index_j, A.max_level);
      // DENSE_BLOCKS.matrix_map[key] = nullptr;
    }
  }

  // Populate non-leaf dense blocks with pointers of the non-leaf dense blocks.
  for (int64_t level = A.max_level - 1; level >= A.min_level - 1; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      for (int64_t j = 0; j < near_neighbours_size(i, level); ++j) {
        int64_t index_j = near_neighbours_index(i, level, j);
        parsec_data_key_t key = block_key(i, index_j, level);
        // NON_LEAF_DENSE_BLOCKS.matrix_map[key] = nullptr;
      }
    }
  }

  // Populate leaf-level bases with pointers of the leaf level bases.
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t node = MPIRANK; node < nblocks; node += MPISIZE) {
      parsec_data_key_t key = bases_key(node, level);
      // BASES_BLOCKS.matrix_map[key] = nullptr;
    }
  }

  // Populate S_BLOCKS data with pointes to the skeleton blocks.
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      for (int64_t j = 0; j < far_neighbours_size(i, level); ++j) {
        int64_t index_j = far_neighbours_index(i, level, j);
        parsec_data_key_t key = block_key(i, index_j, level);
        // S_BLOCKS.matrix_map[key] = nullptr;
      }
    }
  }
  auto stop_ptg_teardown = std::chrono::system_clock::now();
  double ptg_teardown_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_ptg_teardown - start_ptg_teardown).count();
  std::cout << "### PTG teardown time: " << ptg_teardown_time << std::endl;
}
