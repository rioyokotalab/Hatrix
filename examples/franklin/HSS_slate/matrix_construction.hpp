#pragma once

#include "franklin/franklin.hpp"

#include "MPISymmSharedBasisMatrix.hpp"
#include "MPIWrapper.hpp"

void init_diagonal_admis(MPISymmSharedBasisMatrix& A, const Hatrix::Args& opts);
void construct_h2_mpi_miro(MPISymmSharedBasisMatrix& A, const Hatrix::Domain& domain,
                           const Hatrix::Args& opts);
