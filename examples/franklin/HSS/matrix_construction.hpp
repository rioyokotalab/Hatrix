#pragma once

#include "franklin/franklin.hpp"
#include "SymmetricSharedBasisMatrix.hpp"

void init_diagonal_admis(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Domain& domain, const Hatrix::Args& opts);

// Init geometry-based admissibility with dual tree traversal.
void init_geometry_admis(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Domain& domain,
                         const Hatrix::Args& opts);

void construct_h2_matrix_miro(Hatrix::SymmetricSharedBasisMatrix& A,
                              const Hatrix::Domain& domain,
                              const Hatrix::Args& args);
void print_h2_structure(const Hatrix::SymmetricSharedBasisMatrix& A);
double reconstruct_accuracy(const Hatrix::SymmetricSharedBasisMatrix& A,
                            const Hatrix::Domain& domain,
                            const Hatrix::Matrix& dense,
                            const Hatrix::Args& opts);
