#include "Hatrix/Hatrix.h"

#include "SymmetricSharedBasisMatrix.hpp"

using namespace Hatrix;

Matrix
SymmetricSharedBasisMatrix::Ubig(int64_t node, int64_t level) const {
  if (level == max_level) {
    return U(node, level);
  }

  int64_t node_rank = U(node, level).cols;
  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;

  Matrix Ubig_child1 = Ubig(child1, level+1);
  Matrix Ubig_child2 = Ubig(child2, level+1);

  int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;

  Matrix Ubig(block_size, node_rank);

  std::vector<Matrix> Ubig_splits = Ubig.split(
                                               std::vector<int64_t>(1,
                                                                    Ubig_child1.rows),
                                               {});

  std::vector<Matrix> U_splits = U(node, level).split(std::vector<int64_t>(1,
                                                                           Ubig_child1.cols),
                                                      {});

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

  return Ubig;
}

int64_t SymmetricSharedBasisMatrix::max_rank() {
  int64_t m_rank = 0;
  for (int64_t level = min_level+1; level <= max_level; ++level) {
    for (int64_t node = 0; node < pow(2, level); ++node) {
      if (ranks(node, level) > m_rank) {
        m_rank = ranks(node, level);
      }
    }
  }

  return m_rank;
}
