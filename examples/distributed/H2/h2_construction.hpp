#pragma once

#include "distributed/distributed.hpp"

void
construct_h2_matrix_miro(Hatrix::SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);
