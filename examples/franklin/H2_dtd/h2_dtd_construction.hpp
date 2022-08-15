#pragma once

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

void
init_geometry_admis(Hatrix::SymmetricSharedBasisMatrix& A,
                    const Hatrix::Domain& domain,
                    const Hatrix::Args& opts);
void
construct_h2_matrix_dtd(Hatrix::SymmetricSharedBasisMatrix& A,
                        const Hatrix::Domain& domain,
                        const Hatrix::Args& opts);
