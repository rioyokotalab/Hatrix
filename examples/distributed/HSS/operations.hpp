#pragma once

#include "distributed/distributed.hpp"

void factorize(Hatrix::SymmetricSharedBasisMatrix& A);
Hatrix::Matrix solve(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);
Hatrix::Matrix matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);
