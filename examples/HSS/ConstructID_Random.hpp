#pragma once

#include "Hatrix/Hatrix.h"

#include "SharedBasisMatrix.hpp"
#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  class ConstructID_Random : public ConstructAlgorithm {
    const int64_t p = 100;
    std::tuple<std::vector<std::vector<int64_t>>, std::vector<Matrix>, std::vector<Matrix>>
    generate_leaf_blocks(const Matrix& samples, const Matrix& OMEGA);
    std::tuple<std::vector<std::vector<int64_t>>, std::vector<Matrix>, std::vector<Matrix>>
    generate_transfer_blocks(const std::vector<std::vector<int64_t>>& row_indices,
                             const std::vector<Matrix>& S_loc_blocks,
                             const std::vector<Matrix>& OMEGA_blocks,
                             int level);
  public:
    ConstructID_Random(SharedBasisMatrix* context);
    void construct();
  };
}
