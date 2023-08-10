#include "factorize_ptg.hpp"
#include "ptg_interfaces.h"

using namespace Hatrix;

parsec_context_t* parsec = NULL;
H2_ptg_t DENSE_BLOCKS;

uint32_t
rank_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  return 0;
}

parsec_data_t*
data_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {

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

  int stride = pow(2, level);

  return pow(2, level) + row * stride + col;
}

parsec_data_key_t
data_key_bases(parsec_data_collection_t* desc, ...) {
  int row, level;
  va_list ap;
  va_start(ap, desc);
  row = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return pow(2, level) + row;
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
init_block_dc(H2_ptg_t& parsec_data) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->data_key = data_key_block;

  o->rank_of = rank_of;
  o->rank_of_key = rank_of_key;

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
  init_block_dc(DENSE_BLOCKS);

  parsec_data_collection_t* DENSE_BLOCKS_dc = &DENSE_BLOCKS.super;

  parsec_h2_factorize_flows_taskpool_t *taskpool =
    parsec_h2_factorize_flows_new(DENSE_BLOCKS_dc);

  return taskpool;
}

void
factorize_ptg(SymmetricSharedBasisMatrix& A,
              const Domain& domain,
              const Args& opts) {
  parsec_h2_factorize_flows_taskpool_t *tp = h2_factorize_ptg_New(A, domain, opts);

}
