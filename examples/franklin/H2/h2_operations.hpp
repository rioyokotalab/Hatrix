#pragma once

#include "franklin/franklin.hpp"

// matvec routine
Hatrix::Matrix matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);
