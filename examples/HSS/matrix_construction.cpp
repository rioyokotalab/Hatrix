#include <exception>

#include "matrix_construction.hpp"
#include "SymmetricSharedBasisMatrix.hpp"
#include "Domain.hpp"
#include "Args.hpp"

using namespace Hatrix;

static void coarsen_blocks(SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j <= i; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (A.is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !A.is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            A.is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      A.is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

static void diagonal_admis_init(SymmetricSharedBasisMatrix& A, const Args& opts, int64_t level) {
  int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
  A.level_blocks.push_back(nblocks);
  if (level == 0) { return; }
  if (level == A.height) {
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j <= i; ++j) {
        A.is_admissible.insert(i, j, level, std::abs(i - j) > opts.admis);
      }
    }
  }
  else {
    coarsen_blocks(A, level);
  }

  diagonal_admis_init(A, opts, level-1);
}

void init_diagonal_admis(SymmetricSharedBasisMatrix& A, const Args& opts) {
  A.height = int64_t(log2(opts.N / opts.nleaf));
  diagonal_admis_init(A, opts, A.height);
}

void init_geometry_admis(SymmetricSharedBasisMatrix& A, const Args& opts) {
  throw std::exception();
}

void construct_h2_matrix_miro(SymmetricSharedBasisMatrix& A,
                              const Domain& domain,
                              const Args& args) {

}
