#pragma once

#include "franklin/franklin.hpp"

Hatrix::SymmetricSharedBasisMatrix
dense_cholesky_test(const Hatrix::SymmetricSharedBasisMatrix& A,
                    const Hatrix::Domain& domain, const Hatrix::Args& opts);
