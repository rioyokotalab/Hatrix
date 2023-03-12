#pragma once

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

void
construct_h2_matrix_dtd(Hatrix::SymmetricSharedBasisMatrix& A,
                        const Hatrix::Domain& domain,
                        const Hatrix::Args& opts,
                        double* DENSE_MEM, std::vector<int>& DENSE);
