#pragma once

#include "franklin/franklin.hpp"

void init_geometry_admis(Hatrix::SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);
void
construct_h2_matrix_miro(Hatrix::SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);
