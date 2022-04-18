#pragma once

#include "Hatrix/Hatrix.h"

#include "SharedBasisMatrix.hpp"
#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
// Construct HSS matrix using the algorithm written in the miro board.
  class ConstructMiro : public ConstructAlgorithm {
  private:
    Matrix generate_row_block(int64_t block, int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_row_bases(int64_t block, int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_column_bases(int64_t block, int64_t block_size, int64_t level);
    Matrix generate_column_block(int64_t block, int64_t block_size, int64_t level);
    void generate_leaf_nodes(const Domain& domain);

    bool row_has_admissible_blocks(int64_t row, int64_t level);
    bool col_has_admissible_blocks(int64_t col, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                              int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                                              int64_t block_size, int64_t level);
    std::tuple<RowLevelMap, ColLevelMap>
    generate_transfer_matrices(int64_t level, RowLevelMap& Uchild, ColLevelMap& Vchild);

  public:
    ConstructMiro(SharedBasisMatrix* context);
    void construct();
  };
}
