#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "ConstructMiro.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructMiro::ConstructMiro(SharedBasisMatrix* context) : ConstructAlgorithm(context) {}

  Matrix
  ConstructMiro::generate_column_block(int64_t block, int64_t block_size,
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
  ConstructMiro::generate_column_bases(int64_t block, int64_t block_size, int64_t level) {
    // Row slice since column bases should be cutting across the columns.
    Matrix AY = generate_column_block(block, block_size, level);
    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = truncated_svd(AY, context->rank);

    return {std::move(Ui), std::move(Si)};
  }

  Matrix
  ConstructMiro::generate_row_block(int64_t block, int64_t block_size, int64_t level) {
    int nrows = 0;
    int num_blocks = 0;
    for (int64_t i = 0; i < context->level_blocks[level]; ++i) {
      if (context->is_admissible.exists(i, block, level) &&
          !context->is_admissible(i, block, level)) { continue; }
      nrows += context->get_block_size(i, level);
      num_blocks++;
    }

    Hatrix::Matrix YtA(nrows, block_size);
    auto YtA_splits = YtA.split(num_blocks, 1);

    int index = 0;
    for (int64_t i = 0; i < context->level_blocks[level]; ++i) {
      if (context->is_admissible.exists(i, block, level) &&
          !context->is_admissible(i, block, level)) { continue; }
      Hatrix::generate_p2p_interactions(context->domain, i, block, level,
                                        context->height, context->kernel,
                                        YtA_splits[index++]);
    }

    return YtA;
  }


  std::tuple<Matrix, Matrix>
  ConstructMiro::generate_row_bases(int64_t block, int64_t block_size, int64_t level) {
    Matrix YtA = generate_row_block(block, block_size, level);

    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(YtA, context->rank);

    return {std::move(Si), transpose(Vi)};
  }

  void
  ConstructMiro::generate_leaf_nodes(const Domain& domain) {
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

    if (context->is_symmetric) {
      // Generate U leaf blocks
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
    else {
      for (int64_t i = 0; i < nblocks; ++i) {
        Matrix Utemp, Stemp;
        std::tie(Utemp, Stemp) =
          generate_column_bases(i, domain.boxes[i].num_particles, context->height);
        context->U.insert(i, context->height, std::move(Utemp));
      }
      // Generate V leaf blocks
      for (int64_t j = 0; j < nblocks; ++j) {
        Matrix Stemp, Vtemp;
        std::tie(Stemp, Vtemp) =
          generate_row_bases(j, domain.boxes[j].num_particles, context->height);
        context->V.insert(j, context->height, std::move(Vtemp));
      }

      // Generate S coupling matrices
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if (context->is_admissible.exists(i, j, context->height) &&
              context->is_admissible(i, j, context->height)) {
            Matrix dense = generate_p2p_interactions(context->domain, i, j, context->kernel);

            context->S.insert(i, j, context->height,
                              matmul(matmul(context->U(i, context->height), dense, true, false),
                                     context->V(j, context->height)));
          }
        }
      }
    }
  }

  bool
  ConstructMiro::row_has_admissible_blocks(int64_t row, int64_t level) {
    bool has_admis = false;
    for (int64_t i = 0; i < context->level_blocks[level]; ++i) {
      if (!context->is_admissible.exists(row, i, level) ||
          (context->is_admissible.exists(row, i, level) && context->is_admissible(row, i, level))) {
        has_admis = true;
        break;
      }
    }

    return has_admis;
  }

  bool
  ConstructMiro::col_has_admissible_blocks(int64_t col, int64_t level) {
    bool has_admis = false;
    for (int64_t j = 0; j < context->level_blocks[level]; ++j) {
      if (!context->is_admissible.exists(j, col, level) ||
          (context->is_admissible.exists(j, col, level) && context->is_admissible(j, col, level))) {
        has_admis = true;
        break;
      }
    }

    return has_admis;
  }

  std::tuple<Matrix, Matrix>
  ConstructMiro::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                            int64_t block_size, int64_t level) {
    Matrix col_block = generate_column_block(node, block_size, level);
    auto col_block_splits = col_block.split(2, 1);

    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, col_block.cols);
    auto temp_splits = temp.split(2, 1);

    matmul(Ubig_child1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

    Matrix Utransfer, Si, Vi; double error;
    std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, context->rank);

    return {std::move(Utransfer), std::move(Si)};
  }

  std::tuple<Matrix, Matrix>
  ConstructMiro::generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                                            int64_t block_size, int64_t level) {
    Matrix row_block = generate_row_block(node, block_size, level);
    auto row_block_splits = row_block.split(1, 2);

    Matrix temp(row_block.rows, Vbig_child1.cols + Vbig_child2.cols);
    auto temp_splits = temp.split(1, 2);

    matmul(row_block_splits[0], Vbig_child1, temp_splits[0]);
    matmul(row_block_splits[1], Vbig_child2, temp_splits[1]);

    Matrix Ui, Si, Vtransfer; double error;
    std::tie(Ui, Si, Vtransfer, error) = truncated_svd(temp, context->rank);

    return {std::move(Si), transpose(Vtransfer)};
  }

  std::tuple<RowLevelMap, ColLevelMap>
  ConstructMiro::generate_transfer_matrices(int64_t level, RowLevelMap& Uchild,
                                            ColLevelMap& Vchild) {
    int64_t nblocks = context->level_blocks[level];

    // Generate the actual bases for the upper level and pass it to this
    // function again for generating transfer matrices at successive levels.
    RowLevelMap Ubig_parent;
    ColLevelMap Vbig_parent;

    for (int64_t node = 0; node < nblocks; ++node) {
      int64_t child1 = node * 2;
      int64_t child2 = node * 2 + 1;
      int64_t child_level = level + 1;
      int64_t block_size = context->get_block_size(node, level);

      if (row_has_admissible_blocks(node, level) && context->height != 1) {
        Matrix& Ubig_child1 = Uchild(child1, child_level);
        Matrix& Ubig_child2 = Uchild(child2, child_level);

        Matrix Utransfer, Stemp;
        std::tie(Utransfer, Stemp) = generate_U_transfer_matrix(Ubig_child1,
                                                                Ubig_child2,
                                                                node,
                                                                block_size,
                                                                level);

        context->U.insert(node, level, std::move(Utransfer));
        context->Srow.insert(node, level, std::move(Stemp));

        // Generate the full bases to pass onto the parent.
        auto Utransfer_splits = context->U(node, level).split(2, 1);
        Matrix Ubig(block_size, context->rank);
        auto Ubig_splits = Ubig.split(2, 1);

        matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
        matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

        Ubig_parent.insert(node, level, std::move(Ubig));
      }

      if (col_has_admissible_blocks(node, level) && context->height != 1) {
        // Generate V transfer matrix.
        Matrix& Vbig_child1 = Vchild(child1, child_level);
        Matrix& Vbig_child2 = Vchild(child2, child_level);

        Matrix Vtransfer, Stemp;
        std::tie(Stemp, Vtransfer) = generate_V_transfer_matrix(Vbig_child1,
                                                                Vbig_child2,
                                                                node,
                                                                block_size,
                                                                level);
        context->V.insert(node, level, std::move(Vtransfer));
        context->Scol.insert(node, level, std::move(Stemp));

        // Generate the full bases for passing onto the upper level.
        std::vector<Matrix> Vtransfer_splits = context->V(node, level).split(2, 1);
        Matrix Vbig(context->rank, block_size);
        std::vector<Matrix> Vbig_splits = Vbig.split(1, 2);

        matmul(Vtransfer_splits[0], Vbig_child1, Vbig_splits[0], true, true, 1, 0);
        matmul(Vtransfer_splits[1], Vbig_child2, Vbig_splits[1], true, true, 1, 0);

        Vbig_parent.insert(node, level, transpose(Vbig));
      }
    }

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (context->is_admissible.exists(row, col, level) &&
            context->is_admissible(row, col, level)) {
          int64_t row_block_size = context->get_block_size(row, level);
          int64_t col_block_size = context->get_block_size(col, level);

          Matrix dense = generate_p2p_interactions(context->domain, row, col, level,
                                                   context->height, context->kernel);

          context->S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level),
                                                           dense, true, false),
                                                    Vbig_parent(col, level)));
        }
      }
    }

    return {Ubig_parent, Vbig_parent};
  }


  void
  ConstructMiro::construct() {
    generate_leaf_nodes(context->domain);
    RowLevelMap Uchild = context->U;
    ColLevelMap Vchild = context->V;

    for (int64_t level = context->height-1; level > 0; --level) {
      std::tie(Uchild, Vchild) = generate_transfer_matrices(level, Uchild, Vchild);
    }
  }
}
