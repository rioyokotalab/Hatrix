#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "ConstructMiroFaster.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructMiroFaster::ConstructMiroFaster(SharedBasisMatrix* context) :
    ConstructAlgorithm(context) {}

  Matrix
  ConstructMiroFaster::generate_column_block(int64_t block, int64_t block_size,
                                       int64_t level) {
    int ncols = 0;
    int num_blocks = 0;
    for (int64_t j = 0; j < context->level_blocks[level]; ++j) {
      if (context->is_admissible.exists(block, j, level) &&
          !context->is_admissible(block, j, level)) { continue; }
      ncols += context->get_block_size(j, level);
      num_blocks++;
    }

    Matrix AY(block_size, ncols);


    return AY;
  }

  std::tuple<Matrix, Matrix>
  ConstructMiroFaster::generate_column_bases(int64_t block, int64_t block_size, int64_t level) {
    // Row slice since column bases should be cutting across the columns.
    Matrix AY = generate_column_block(block, block_size, level);
    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = truncated_svd(AY, context->rank);

    return {std::move(Ui), std::move(Si)};
  }

  void
  ConstructMiroFaster::generate_leaf_nodes(const Domain& domain) {
    int64_t nblocks = context->level_blocks[context->height];

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (context->is_admissible.exists(i, j, context->height) &&
            !context->is_admissible(i, j, context->height)) {
        }
      }
    }

    // Generate symmetric U and V leaf blocks
    for (int64_t i = 0; i < nblocks; ++i) {
    }

    // Generate S coupling matrices
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (context->is_admissible.exists(i, j, context->height) &&
            context->is_admissible(i, j, context->height)) {
        }
      }
    }
  }

  void ConstructMiroFaster::generate_transfer_matrices(int64_t level) {
    int64_t nblocks = context->level_blocks[level];
    int64_t child_level = level + 1;

    for (int64_t node = 0; node < nblocks; ++node) {
    }

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < i; ++j) {
        if (context->is_admissible.exists(i, j, level) &&
            context->is_admissible(i, j, level)) {

        }
      }
    }
  }

  void
  ConstructMiroFaster::construct() {
    generate_leaf_nodes(context->domain);

    for (int64_t level = context->height-1; level > 0; --level) {
      generate_transfer_matrices(level);
    }
  }
}
