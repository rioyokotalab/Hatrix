#pragma once

#include "franklin/franklin.hpp"

Hatrix::SymmetricSharedBasisMatrix
dense_cholesky_test(const Hatrix::SymmetricSharedBasisMatrix& A,
                    const Hatrix::Args& opts);

void
vector_permute_test(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x);

void
dense_factorize_and_solve_test(const Hatrix::SymmetricSharedBasisMatrix& A,
                               const Hatrix::Matrix& x, const Hatrix::Args& opts);

void
cholesky_fill_in_recompress_check(const Hatrix::SymmetricSharedBasisMatrix& A,
                                  const Hatrix::Args& opts);
