#pragma once

#include "Domain.hpp"
#include "SymmetricSharedBasisMatrix.hpp"
#include "Args.hpp"

void init_diagonal_admis(Hatrix::SymmetricSharedBasisMatrix& A);
void init_geometry_admis(Hatrix::SymmetricSharedBasisMatrix& A);

void construct_h2_matrix_miro(Hatrix::SymmetricSharedBasisMatrix& A,
                              const Hatrix::Domain& domain,
                              const Hatrix::Args& args);
