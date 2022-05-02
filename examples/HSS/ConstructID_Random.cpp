#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "ConstructMiro.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructID_Random::ConstructID_Random(SharedBasisMatrix* context) :
    ConstructAlgorithm(context) {}

  std::tuple<std::vector<std::vector<int64_t>>, std::vector<Matrix>, std::vector<Matrix>>
  ConstructID_Random::generate_leaf_blocks(const Matrix& samples, const Matrix& OMEGA) {
    std::vector<std::vector<int64_t>> row_indices;
    std::vector<Matrix> S_loc_blocks, OMEGA_blocks;
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

    for (int64_t node = 0; node < nblocks; ++node) {
      // gather indices for leaf nodes. line 1.
      std::vector<int64_t> indices;
      for (int64_t i = node * context->nleaf; i < (node + 1) * context->nleaf; ++i) {
        indices.push_back(i);
      }
      row_indices.push_back(indices);

      // obtain a slice of the random matrix. line 2.
      Matrix OMEGA_loc(indices.size(), p);
      for (int64_t i = 0; i < indices.size(); ++i) {
        int64_t row = row_indices[node][i];
        for (int64_t j = 0; j < p; ++j) {
          OMEGA_loc(i, j) = OMEGA(row, j);
        }
      }

      OMEGA_blocks.push_back(OMEGA_loc);

      Matrix S_loc(indices.size(), p);
      // Copy the samples into its own matrix
      for (int64_t i = 0; i < indices.size(); ++i) {
        int64_t row = row_indices[node][i];
        for (int64_t j = 0; j < p; ++j) {
          S_loc(i, j) = samples(row, j);
        }
      }

      // Remove the dense part from the LR block
      S_loc -= matmul(context->D(node, node, context->height), OMEGA_loc);

      S_loc_blocks.push_back(S_loc);
    }

    return {std::move(row_indices), std::move(S_loc_blocks), std::move(OMEGA_blocks)};
  }

  std::tuple<std::vector<std::vector<int64_t>>, std::vector<Matrix>, std::vector<Matrix>>
  ConstructID_Random::generate_transfer_blocks(const std::vector<std::vector<int64_t>>&
                                               child_row_indices,
                                               const std::vector<Matrix>& child_S_loc_blocks,
                                               const std::vector<Matrix>& child_OMEGA_blocks,
                                               int level) {
    std::vector<std::vector<int64_t>> row_indices;
    std::vector<Matrix> S_loc_blocks, OMEGA_blocks;
    int64_t nblocks = context->level_blocks[level];
    int64_t child_level = level + 1;

    for (int64_t node = 0; node < nblocks; ++node) {
      int64_t c1 = node * 2, c2 = node * 2 + 1;

      // line 5. Store absolute row indices from the child nodes.
      std::vector<int64_t> indices;
      for (int64_t i : child_row_indices[c1]) { indices.push_back(i); }
      for (int64_t i : child_row_indices[c2]) { indices.push_back(i); }
      row_indices.push_back(indices);

      int64_t c1_size = child_row_indices[c1].size();
      int64_t c2_size = child_row_indices[c2].size();

      // line 6. Concat the random matrices.
      Matrix OMEGA_loc(c1_size + c2_size, p);
      auto OMEGA_loc_splits = OMEGA_loc.split(std::vector<int64_t>(1, c1_size), {});
      OMEGA_loc_splits[0] = child_OMEGA_blocks[c1];
      OMEGA_loc_splits[1] = child_OMEGA_blocks[c2];
      OMEGA_blocks.push_back(OMEGA_loc);

      // line 7.Combine samples from the children.
      Matrix Sv1(child_S_loc_blocks[c1]), Sv2(child_S_loc_blocks[c2]);
      Matrix Sloc(c1_size + c2_size, p);
      auto Sloc_splits = Sloc.split(std::vector<int64_t>(1, c1_size), {});

      Sv2 -= matmul(context->S(c2, c1, child_level), child_OMEGA_blocks[c1]);
      Sv1 -= matmul(context->S(c2, c1, child_level), child_OMEGA_blocks[c2], true, false);

      Sloc_splits[0] = Sv1;
      Sloc_splits[1] = Sv2;

      S_loc_blocks.push_back(Sloc);
    }

    return {std::move(row_indices), std::move(S_loc_blocks), std::move(OMEGA_blocks)};
  }


  void
  ConstructID_Random::construct() {
    Matrix dense = generate_p2p_matrix(context->domain, context->kernel);
    Matrix OMEGA = generate_random_matrix(context->N, p);
    // TODO: perform this multiplication with a transpose so that it is possible
    // to perform the interpolation without neeeded to transpose the samples.
    Matrix samples = Hatrix::matmul(dense, OMEGA);

    // begin construction procedure using randomized samples.
    std::vector<std::vector<int64_t>> row_indices(context->level_blocks[context->height]);
    std::vector<Matrix> S_loc_blocks, OMEGA_blocks;
    std::vector<Matrix> pivot_store(context->level_blocks[context->height]);

    for (int64_t level = context->height; level > 0; --level) {
      if (level == context->height) {
        std::tie(row_indices, S_loc_blocks, OMEGA_blocks) =
          generate_leaf_blocks(samples, OMEGA);
      }
      else {
        std::tie(row_indices, S_loc_blocks, OMEGA_blocks) =
          generate_transfer_blocks(row_indices, S_loc_blocks, OMEGA_blocks, level);
      }
      int64_t nblocks = context->level_blocks[level];

      for (int64_t node = 0; node < nblocks; ++node) {
        Matrix interp, pivots;
        int64_t rank;

        // TODO: use rvalues with transpose.
        Matrix sT(transpose(S_loc_blocks[node]));
        std::tie(interp, pivots, rank) = error_interpolate(sT, context->accuracy);
        // TODO: avoid storing both the U and V.
        Matrix Vinterp(interp);
        context->U.insert(node, level, std::move(interp));
        context->V.insert(node, level, std::move(Vinterp));

        // apply the interpolation matrix on the previous random vectors
        OMEGA_blocks[node] = matmul(interp, OMEGA_blocks[node], true, false);

        // choose the rows of the samples that correspond to the interpolation.
        Matrix S_loc(rank, p);
        for (int64_t i = 0; i < rank; ++i) {
          int64_t row = pivots(i, 0)-1;
          for (int64_t j = 0; j < p; ++j) {
            S_loc(i, j) = S_loc_blocks[node](row, j);
          }
        }
        S_loc_blocks[node] = std::move(S_loc);

        // keep the absolute indicies of the rows that span the row of
        // this index set.
        std::vector<int64_t> indices;
        for (int64_t i = 0; i < rank; ++i) {
          indices.push_back(row_indices[node][pivots(i, 0)-1]);
        }
        row_indices[node] = std::move(indices);
      }

      // generate S blocks for the lower triangular region only since this
      // is a symmetric matrix.
      for (int64_t brow = 0; brow < nblocks; ++brow) {
        for (int64_t bcol = 0; bcol < brow; ++bcol) {
          if (context->is_admissible.exists(brow, bcol, level) &&
              context->is_admissible(brow, bcol, level)) {
            int64_t row_size = row_indices[brow].size();
            int64_t col_size = row_indices[bcol].size();

            Matrix Stemp(row_size, col_size);
            for (int64_t i = 0; i < row_size; ++i) {
              for (int64_t j = 0; j < col_size; ++j) {
                Stemp(i, j) = context->kernel(context->domain.particles[brow].coords,
                                              context->domain.particles[bcol].coords);
              }
            }

            context->S.insert(brow, bcol, level, std::move(Stemp));
          }
        }
      }
    }
  }
}
