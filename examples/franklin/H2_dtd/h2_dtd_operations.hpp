#pragma once

#include <unordered_map>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "globals.hpp"

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

typedef struct h2_dc_t {
  parsec_data_collection_t super;
  // map of parsec data descriptors.
  std::unordered_map<parsec_data_key_t, parsec_data_t*> data_map;
  // MPI rank of the corresponding data key.
  std::unordered_map<parsec_data_key_t, uint32_t> mpi_ranks;
  // map of actual Matrix data pointers.
  std::unordered_map<parsec_data_key_t, Hatrix::Matrix*> matrix_map;
} h2_dc_t;

extern h2_dc_t parsec_U, parsec_S, parsec_D;

void h2_dc_init(h2_dc_t& parsec_type,
                parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...));
void h2_dc_destroy(h2_dc_t& parsec_type);

// matvec between H2 matrix and vector X. Store the result in B.
// This function expects the vectors to be in the scalapack layout.
void
matmul(Hatrix::SymmetricSharedBasisMatrix& A,
       const Hatrix::Domain& domain,
       std::vector<Hatrix::Matrix>& x,
       std::vector<Hatrix::Matrix>& b);

void
factorize(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain, const Hatrix::Args& opts);

void
solve(Hatrix::SymmetricSharedBasisMatrix& A,
      std::vector<Hatrix::Matrix>& x, std::vector<Hatrix::Matrix>& h2_solution);
