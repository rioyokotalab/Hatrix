#include "factorize_ptg.hpp"
#include "ptg_interfaces.h"

using namespace Hatrix;

parsec_context_t* parsec = NULL;
H2_ptg_t DENSE_BLOCKS;

int64_t
near_neighbours_size(int64_t node, int64_t level) {
  return near_neighbours(node, level).size();
}

int64_t
near_neighbours_index(int64_t node, int64_t level, int64_t array_loc) {
  return near_neighbours(node, level)[array_loc];
}

int64_t
far_neighbours_size(int64_t node, int64_t level) {
  return far_neighbours(node, level).size();
}

int64_t
far_neighbours_index(int64_t node, int64_t level, int64_t array_loc) {
  return far_neighbours(node, level)[array_loc];
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
basis_key(int64_t row, int64_t level) {
  parsec_data_key_t level_offset = 0;
  for (int64_t l = 0; l < level; ++l) {
    level_offset += pow(2, l);
  }

  return level_offset + row;
}

uint32_t
rank_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  return 0;
}

// NOTE: This function is called only by a process that own the data with key 'key'.
// So it will not return a NULL. Ever.
parsec_data_t*
data_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  H2_ptg_t *dc = (H2_ptg_t*)desc;
  parsec_data_t *data = nullptr;

  if (dc->data_map.count(key) != 0) {
    data = dc->data_map[key];
  }

  if (data == nullptr) {
    Matrix * matrix = dc->matrix_map[key];
    data = parsec_data_create(&dc->data_map[key],
                              desc,
                              key,
                              matrix->data_ptr,
                              matrix->numel() * parsec_datadist_getsizeoftype(PARSEC_MATRIX_DOUBLE),
                              PARSEC_DATA_FLAG_PARSEC_MANAGED);
    dc->data_map[key] = data;
  }

  return data;
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

  return basis_key(row, level);
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
  int row;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  va_end(ap);

  return mpi_rank(row);
}

int32_t vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key) {
  return (int32_t)rank_of_key(desc, key);
}

void
init_block_dc(H2_ptg_t& parsec_data, SymmetricSharedBasisMatrix* A) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->data_key = data_key_block;

  o->rank_of = rank_of;
  o->rank_of_key = rank_of_key;

  o->data_of = data_of_block;
  o->data_of_key = data_of_key;

  o->vpid_of = vpid_of;
  o->vpid_of_key = vpid_of_key;

  o->default_dtt;               // TODO: Figure out how to set this up.

  // parsec_dtd_data_collection_init(o);
}

parsec_h2_factorize_flows_taskpool_t *
h2_factorize_ptg_New(SymmetricSharedBasisMatrix& A,
                     const Domain& domain,
                     const Args& opts) {
  // Populate DENSE_BLOCKS struct with pointers of the dense blocks.
  init_block_dc(DENSE_BLOCKS, &A);
  for (int64_t level = A.max_level; level >= A.min_level-1; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      for (int64_t j = 0; j < near_neighbours_size(i, level); ++j) {
        int64_t index_j = near_neighbours_index(i, level, j);

        DENSE_BLOCKS.matrix_map[block_key(i, index_j, level)] =
          std::addressof(A.D(i, index_j, level));
      }
    }
  }

  parsec_data_collection_t* DENSE_BLOCKS_dc = &DENSE_BLOCKS.super;

  parsec_h2_factorize_flows_taskpool_t *taskpool =
    parsec_h2_factorize_flows_new(DENSE_BLOCKS_dc,
                                  A.max_level,
                                  A.min_level);

  return taskpool;
}

void
factorize_ptg(SymmetricSharedBasisMatrix& A,
              const Domain& domain,
              const Args& opts) {
  parsec_h2_factorize_flows_taskpool_t *tp = h2_factorize_ptg_New(A, domain, opts);

  parsec_context_add_taskpool(parsec, (parsec_taskpool_t*)tp);
  parsec_context_start(parsec);
  parsec_context_wait(parsec);
}
