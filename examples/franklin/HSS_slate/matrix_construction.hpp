#pragma once

#include <random>

#include "franklin/franklin.hpp"

#include "MPISymmSharedBasisMatrix.hpp"
#include "MPIWrapper.hpp"

extern std::mt19937 random_generator;
extern std::uniform_real_distribution<double> uniform_distribution;

void init_diagonal_admis(MPISymmSharedBasisMatrix& A, const Hatrix::Args& opts);
void construct_h2_mpi_miro(MPISymmSharedBasisMatrix& A, const Hatrix::Domain& domain,
                           const Hatrix::Args& opts);
