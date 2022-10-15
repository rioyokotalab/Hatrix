#pragma once

#include "franklin/franklin.hpp"

void factorize(Hatrix::SymmetricSharedBasisMatrix& A);
Hatrix::Matrix solve(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);

// matvec routine
Hatrix::Matrix matmul(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);
