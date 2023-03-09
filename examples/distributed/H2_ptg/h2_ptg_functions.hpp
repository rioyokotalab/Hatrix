#pragma once

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "globals.hpp"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "h2_ptg_internal.h"

typedef struct h2_dc_t {
  parsec_data_collection_t super; // inherit from parsec_data_collection_t
  // map of parsec data descriptors.
  std::unordered_map<parsec_data_key_t, parsec_data_t*> data_map;
  // MPI rank of the corresponding data key.
  std::unordered_map<parsec_data_key_t, uint32_t> mpi_ranks;
  // map of actual Matrix data pointers.
  std::unordered_map<parsec_data_key_t, Hatrix::Matrix*> matrix_map;
} h2_dc_t;

extern h2_dc_t parsec_U, parsec_S, parsec_D, parsec_F,
  parsec_temp_fill_in_rows, parsec_temp_fill_in_cols,
  parsec_US, parsec_r, parsec_t;

extern int U_ARENA, D_ARENA, S_ARENA, FINAL_DENSE_ARENA, U_NON_LEAF_ARENA;

extern Hatrix::RowColLevelMap<Hatrix::Matrix> F;
extern Hatrix::RowColMap<Hatrix::Matrix> r, t;
// store temporary fill-ins.
extern Hatrix::RowColMap<Hatrix::Matrix> temp_fill_in_rows, temp_fill_in_cols;

void
h2_dc_init(h2_dc_t& parsec_data,
           parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...),
           uint32_t (*rank_of_func)(parsec_data_collection_t*, ...));
void h2_dc_destroy(h2_dc_t& parsec_type);

parsec_data_key_t data_key_1d(parsec_data_collection_t* dc, ...);
parsec_data_key_t data_key_2d(parsec_data_collection_t* dc, ...);
uint32_t rank_of_1d(parsec_data_collection_t* desc, ...);
uint32_t rank_of_2d(parsec_data_collection_t* desc, ...);

void
factorize_setup(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                const Hatrix::Args& opts);
void factorize_teardown();

// make a task pool for this factorization.
parsec_taskpool_t *
h2_factorize_New(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                 const Hatrix::Args& opts, h2_factorize_params_t* h2_params);
void
h2_factorize_Destruct(parsec_taskpool_t *h2_factorize);
