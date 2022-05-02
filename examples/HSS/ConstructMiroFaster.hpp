#pragma once

#include "Hatrix/Hatrix.h"

#include "SharedBasisMatrix.hpp"
#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  class ConstructMiroFaster : public ConstructAlgorithm {
  private:
    void generate_leaf_nodes(const Domain& domain);
    std::tuple<Matrix, Matrix> generate_column_bases(int64_t block,
                                                     int64_t block_size,
                                                     int64_t level);
    Matrix generate_column_block(int64_t block, int64_t block_size,
                                 int64_t level);
    void generate_transfer_matrices(int64_t level);
  public:
    ConstructMiroFaster(SharedBasisMatrix* context);
    void construct();
  };
}
