#pragma once

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"
#include "globals.hpp"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

typedef struct h2_dc_t {
  parsec_data_collection_t super; // inherit from parsec_data_collection_t
  // map of parsec data descriptors.
  std::unordered_map<parsec_data_key_t, parsec_data_t*> data_map;
  // MPI rank of the corresponding data key.
  std::unordered_map<parsec_data_key_t, uint32_t> mpi_ranks;
  // map of actual Matrix data pointers.
  std::unordered_map<parsec_data_key_t, Hatrix::Matrix*> matrix_map;
} h2_dc_t;

void
h2_dc_init(h2_dc_t& parsec_data,
           parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...),
           uint32_t (*rank_of_func)(parsec_data_collection_t*, ...));
void h2_dc_destroy(h2_dc_t& parsec_type);

parsec_data_key_t data_key_1d(parsec_data_collection_t* dc, ...);
parsec_data_key_t data_key_2d(parsec_data_collection_t* dc, ...);
uint32_t rank_of_1d(parsec_data_collection_t* desc, ...);
uint32_t rank_of_2d(parsec_data_collection_t* desc, ...);

int hatrix_factorize_New(parsec_context_t *parsec);
