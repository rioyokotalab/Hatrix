#include <algorithm>
#include <exception>
#include <cmath>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_construction.hpp"
#include "SymmetricSharedBasisMatrix.hpp"

static void
dual_tree_traversal(SymmetricSharedBasisMatrix& A, const Cell& Ci, const Cell& Cj,
                    const Domain& domain, const Args& opts) {
  int64_t i_level = Ci.level;
  int64_t j_level = Cj.level;

  if (i_level == j_level) {
    double distance = 0;
    for (int64_t k = 0; k < opts.ndim; ++k) {
      distance += pow(Ci.center[k] - Cj.center[k], 2);
    }
    distance = sqrt(distance);
    bool well_separated = false;

    if (distance * opts.admis > Ci.radius + Cj.radius) { // well-separated blocks.
      well_separated = true;
    }

    A.is_admissible.insert(Ci.level_index, Cj.level_index, i_level, std::move(well_separated));
  }
  if (i_level <= j_level && Ci.cells.size() > 0) { // j is at a higher level and i is not leaf.
    dual_tree_traversal(A, Ci.cells[0], Cj, domain, opts);
    dual_tree_traversal(A, Ci.cells[1], Cj, domain, opts);
  }
  else if (j_level <= i_level && Cj.cells.size() > 0) { // i is at a higheer level and j is not leaf.
    dual_tree_traversal(A, Ci, Cj.cells[0], domain, opts);
    dual_tree_traversal(A, Ci, Cj.cells[1], domain, opts);
  }
}

void init_geometry_admis(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  A.max_level = domain.tree.height() - 1;
  dual_tree_traversal(A, domain.tree, domain.tree, domain, opts);
  A.min_level = 0;
}