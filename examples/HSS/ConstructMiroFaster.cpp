#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "ConstructMiroFaster.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructMiroFaster::ConstructMiroFaster(SharedBasisMatrix* context) : ConstructAlgorithm(context) {}

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
    auto AY_splits = AY.split(1, num_blocks);

    int index = 0;
    for (int64_t j = 0; j < context->level_blocks[level]; ++j) {
      if (context->is_admissible.exists(block, j, level) &&
          !context->is_admissible(block, j, level)) { continue; }
      Hatrix::generate_p2p_interactions(context->domain, block, j, level, context->height,
                                        context->kernel, AY_splits[index++]);
    }

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
          context->D.insert(i, j, context->height,
                            generate_p2p_interactions(context->domain, i, j, context->kernel));
        }
      }
    }

    // Generate symmetric U and V leaf blocks
    for (int64_t i = 0; i < nblocks; ++i) {
      Matrix Utemp, Stemp;
      std::tie(Utemp, Stemp) =
        generate_column_bases(i, domain.boxes[i].num_particles, context->height);
      Matrix Vtemp(Utemp);

      context->U.insert(i, context->height, std::move(Utemp));
      context->V.insert(i, context->height, std::move(Vtemp));

      Matrix Stempcol(Stemp);
      context->Srow.insert(i, context->height, std::move(Stemp));
      context->Scol.insert(i, context->height, std::move(Stempcol));
    }

    // Generate S coupling matrices
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < i; ++j) {
        if (context->is_admissible.exists(i, j, context->height) &&
            context->is_admissible(i, j, context->height)) {
          Matrix dense = generate_p2p_interactions(context->domain, i, j, context->kernel);
          Matrix Sblock = matmul(matmul(context->U(i, context->height), dense, true, false),
                                 context->V(j, context->height));
          Matrix SblockT(transpose(Sblock));

          context->S.insert(i, j, context->height, std::move(Sblock));
          context->S.insert(j, i, context->height, std::move(SblockT));
        }
      }
    }
  }

  void
  ConstructMiroFaster::construct() {
    generate_leaf_nodes(context->domain);
  }
}
