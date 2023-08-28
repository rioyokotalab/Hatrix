#include "Hatrix/Hatrix.hpp"

#include <cassert>
#include <cmath>

using namespace Hatrix;

void SymmetricSharedBasisMatrix::actually_print_structure(int64_t level) {
  int64_t nblocks = pow(2, level);
  std::cout << "LEVEL:" << level << " NBLOCKS: " << nblocks << " MIN: " << min_level << std::endl;
  for (int64_t i = 0; i < nblocks; ++i) {
    std::cout << "| " ;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, level)) {
        std::cout << is_admissible(i, j, level) << " | " ;
      }
      else {
        std::cout << "  | ";
      }
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  if (level <= min_level) { return; }

  actually_print_structure(level-1);
}

void
SymmetricSharedBasisMatrix::print_structure() {
  actually_print_structure(max_level);
}

double
SymmetricSharedBasisMatrix::Csp(int64_t level) {
  assert(level >= min_level && level <= max_level);

  int64_t nblocks = pow(2, level);
  double avg_csp = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    int64_t i_dense = 0;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
        avg_csp++;
      }
    }
  }

  return avg_csp / nblocks;
}

int64_t
SymmetricSharedBasisMatrix::leaf_dense_blocks() {
  int64_t nblocks = pow(2, max_level);
  int64_t ndense_blocks = 0;

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (is_admissible.exists(i, j, max_level) && !is_admissible(i, j, max_level)) {
        ndense_blocks += 1;
      }
    }
  }

  return ndense_blocks;
}

SymmetricSharedBasisMatrix::SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A) :
  min_level(A.min_level), max_level(A.max_level) {
  is_admissible.deep_copy(A.is_admissible);
  S.deep_copy(A.S);
  D.deep_copy(A.D);
  U.deep_copy(A.U);
}

SymmetricSharedBasisMatrix::SymmetricSharedBasisMatrix() {}
