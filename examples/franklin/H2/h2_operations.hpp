#pragma once

#include "franklin/franklin.hpp"

extern Hatrix::RowLevelMap US;

void factorize(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts);
Hatrix::Matrix solve(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);

// matvec routine
Hatrix::Matrix matmul(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);
