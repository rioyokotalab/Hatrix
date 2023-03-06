#pragma once

#include <unordered_map>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

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

void
h2_dc_init(h2_dc_t& parsec_data,
           parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...),
           uint32_t (*rank_of_func)(parsec_data_collection_t*, ...));
void h2_dc_destroy(h2_dc_t& parsec_type);

parsec_data_key_t data_key_1d(parsec_data_collection_t* dc, ...);
parsec_data_key_t data_key_2d(parsec_data_collection_t* dc, ...);
uint32_t rank_of_1d(parsec_data_collection_t* desc, ...);
uint32_t rank_of_2d(parsec_data_collection_t* desc, ...);

// matvec between H2 matrix and vector X. Store the result in B.
// This function expects the vectors to be in the scalapack layout.
void
matmul(Hatrix::SymmetricSharedBasisMatrix& A,
       const Hatrix::Domain& domain,
       std::vector<Hatrix::Matrix>& x,
       std::vector<Hatrix::Matrix>& b);

long long int
factorize(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain, const Hatrix::Args& opts);

void
solve(Hatrix::SymmetricSharedBasisMatrix& A,
      std::vector<Hatrix::Matrix>& x,
      std::vector<Hatrix::Matrix>& h2_solution,
      const Hatrix::Domain& domain);

void
multiply_complements(Hatrix::SymmetricSharedBasisMatrix& A,
                     Hatrix::Domain& domain,
                     const int64_t block, const int64_t level);

void
factorize_diagonal(Hatrix::SymmetricSharedBasisMatrix& A,
                   const Hatrix::Domain& domain,
                   const int64_t block,
                   const int64_t level);

void
triangle_reduction(Hatrix::SymmetricSharedBasisMatrix& A,
                   const Hatrix::Domain& domain,
                   const int64_t block,
                   const int64_t level);

void
compute_schurs_complement(Hatrix::SymmetricSharedBasisMatrix& A,
                          const Hatrix::Domain& domain,
                          const int64_t block,
                          const int64_t level);
void
preallocate_blocks(Hatrix::SymmetricSharedBasisMatrix& A);

void
update_parsec_pointers(Hatrix::SymmetricSharedBasisMatrix& A,
                       const Hatrix::Domain& domain, int64_t level);

void
merge_unfactorized_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                          const Hatrix::Domain& domain,
                          int64_t level);

void
preallocate_blocks(Hatrix::SymmetricSharedBasisMatrix& A);

void
update_parsec_pointers(Hatrix::SymmetricSharedBasisMatrix& A,
                       const Hatrix::Domain& domain, int64_t level);

void h2_dc_init_maps();
void h2_dc_destroy_maps();
void h2_destroy_arenas(int64_t max_level, int64_t min_level);

void
factorize_setup(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                const Hatrix::Args& opts);
void factorize_teardown();
