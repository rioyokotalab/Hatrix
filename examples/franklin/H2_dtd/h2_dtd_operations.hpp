#pragma once

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "globals.hpp"

// matvec between H2 matrix and vector X. Store the result in B.
// This function expects the vectors to be in the scalapack layout.
void
matmul(Hatrix::SymmetricSharedBasisMatrix& A,
       const Hatrix::Domain& domain,
       std::vector<Hatrix::Matrix>& x,
       std::vector<Hatrix::Matrix>& b);

void
factorize(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts);

void
solve(Hatrix::SymmetricSharedBasisMatrix& A,
      std::vector<Hatrix::Matrix>& x, std::vector<Hatrix::Matrix>& h2_solution);
