#pragma once

#include <unordered_map>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

// matvec between H2 matrix and vector X. Store the result in B.
// This function expects the vectors to be in the scalapack layout.
void
matmul(Hatrix::SymmetricSharedBasisMatrix& A,
       const Hatrix::Domain& domain,
       std::vector<Hatrix::Matrix>& x,
       std::vector<Hatrix::Matrix>& b);

int64_t
get_dim(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Domain& domain,
        const int64_t block, const int64_t level);
