#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include "Hatrix/Hatrix.hpp"
#include "distributed/distributed.hpp"

using namespace Hatrix;

// This routine is not designed for multi-process.
void
factorize_noparsec(SymmetricSharedBasisMatrix& A,
                   const Domain& domain,
                   const Args& opts);
