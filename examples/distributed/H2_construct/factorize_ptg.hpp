#pragma once

#include <unordered_map>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "h2_factorize_flows.h"

// Global declarations for use with the parsec runtime engine.
extern parsec_context_t *parsec;

using namespace Hatrix;

typedef struct H2_ptg_t {
  parsec_data_collection_t super;

  std::unordered_map<parsec_data_key_t, parsec_data_t*> data_map;
  std::unordered_map<parsec_data_key_t, Matrix*> matrix_map;
} H2_ptg_t;

extern RowColMap<std::vector<int64_t> > near_neighbours, far_neighbours;

parsec_data_key_t data_key_block(parsec_data_collection_t* desc, ...);
parsec_data_key_t data_key_bases(parsec_data_collection_t* desc, ...);
uint32_t rank_of(parsec_data_collection_t* desc, ...);

parsec_h2_factorize_flows_taskpool_t *
h2_factorize_ptg_New(SymmetricSharedBasisMatrix& A,
                     const Domain& domain,
                     const Args& opts);

void
factorize_ptg(SymmetricSharedBasisMatrix& A,
              const Domain& domain,
              const Args& opts);
