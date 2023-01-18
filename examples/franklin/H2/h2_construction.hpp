#pragma once

#include "franklin/franklin.hpp"

// storage for near and far blocks at each level.
extern Hatrix::RowColMap<std::vector<int64_t>> near_neighbours, far_neighbours;  // This is actually RowLevelMap

void init_geometry_admis(Hatrix::SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);
void
construct_h2_matrix_miro(Hatrix::SymmetricSharedBasisMatrix& A,
                         const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);
