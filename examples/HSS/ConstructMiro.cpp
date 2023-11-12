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
                                       int64_t level, const Matrix& A, const Matrix& rand) {
    int nblocks = context->level_blocks[level];

    auto A_splits = A.split(nblocks, nblocks);
    auto rand_splits = rand.split(nblocks, 1);
    Matrix AY(block_size, rand.cols);

    int index = 0;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (context->is_admissible.exists(block, j, level) &&
          !context->is_admissible(block, j, level)) { continue; }
      matmul(A_splits[block * nblocks + j], rand_splits[j], AY, false, false, 1, 1);
    }

    return AY;
  }

  Matrix
  ConstructMiro::generate_column_bases(int64_t block, int64_t block_size, int64_t level,
                                       const Matrix& A, const Matrix& rand) {
    // Row slice since column bases should be cutting across the columns.
    Matrix AY = generate_column_block(block, block_size, level, A, rand);
    Matrix Ui;
    if (context->rank > 0) {             // constant rank compression
      std::vector<int64_t> pivots;
      std::tie(Ui, pivots) = pivoted_qr(AY, context->rank);
    }
    else {                      // constant accuracy compression
      std::vector<int64_t> pivots; int64_t rank;
      std::tie(Ui, pivots, rank) = error_pivoted_qr(AY, context->accuracy);
    }

    return {std::move(Ui)};
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
  ConstructMiro::generate_leaf_nodes(const Domain& domain, const Matrix& A, const Matrix& rand) {
    int64_t nblocks = context->level_blocks[context->height];
    auto A_splits = A.split(nblocks, nblocks);

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (context->is_admissible.exists(i, j, context->height) &&
            !context->is_admissible(i, j, context->height)) {
          Matrix Aij(A_splits[i * nblocks + j], true);
          context->D.insert(i, j, context->height, std::move(Aij));
        }
      }
    }

    if (context->is_symmetric) {
      // Generate U leaf blocks
      for (int64_t i = 0; i < nblocks; ++i) {
        Matrix Utemp = generate_column_bases(i, domain.boxes[i].num_particles, context->height,
                                             A, rand);
        Matrix Vtemp(Utemp);

        context->U.insert(i, context->height, std::move(Utemp));
        context->V.insert(i, context->height, std::move(Vtemp));
      }

      // Generate S coupling matrices
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if (context->is_admissible.exists(i, j, context->height) &&
              context->is_admissible(i, j, context->height)) {
            Matrix Sblock = matmul(matmul(context->U(i, context->height),
                                          A_splits[i * nblocks + j], true, false),
                                   context->V(j, context->height));
            context->S.insert(i, j, context->height, std::move(Sblock));
          }
        }
      }
    }
    // else {
    //   for (int64_t i = 0; i < nblocks; ++i) {
    //     Matrix Utemp, Stemp;
    //     std::tie(Utemp, Stemp) =
    //       generate_column_bases(i, domain.boxes[i].num_particles, context->height);
    //     context->U.insert(i, context->height, std::move(Utemp));
    //   }
    //   // Generate V leaf blocks
    //   for (int64_t j = 0; j < nblocks; ++j) {
    //     Matrix Stemp, Vtemp;
    //     std::tie(Stemp, Vtemp) =
    //       generate_row_bases(j, domain.boxes[j].num_particles, context->height);
    //     context->V.insert(j, context->height, std::move(Vtemp));
    //   }

    //   // Generate S coupling matrices
    //   for (int64_t i = 0; i < nblocks; ++i) {
    //     for (int64_t j = 0; j < nblocks; ++j) {
    //       if (context->is_admissible.exists(i, j, context->height) &&
    //           context->is_admissible(i, j, context->height)) {
    //         Matrix dense = generate_p2p_interactions(context->domain, i, j, context->kernel);

    //         context->S.insert(i, j, context->height,
    //                           matmul(matmul(context->U(i, context->height), dense, true, false),
    //                                  context->V(j, context->height)));
    //       }
    //     }
    //   }
    // }
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

  Matrix
  ConstructMiro::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                            int64_t block_size, int64_t level,
                                            const Matrix& A, const Matrix& rand) {
    Matrix col_block = generate_column_block(node, block_size, level, A, rand);
    auto col_block_splits = col_block.split(2, 1);

    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, col_block.cols);
    auto temp_splits = temp.split(std::vector<int64_t>(1, Ubig_child1.cols),
                                  {});

    matmul(Ubig_child1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

    Matrix Utransfer;
    std::vector<int64_t> pivots;
    if (context->rank > 0) {    // constant rank
      std::tie(Utransfer, pivots) = pivoted_qr(temp, context->rank);
    }
    else {
      int64_t rank;
      std::tie(Utransfer, pivots, rank) = error_pivoted_qr(temp, context->accuracy);
    }

    return std::move(Utransfer);
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
                                            ColLevelMap& Vchild, const Matrix& A,
                                            const Matrix& rand) {
    int64_t nblocks = context->level_blocks[level];
    auto A_splits = A.split(nblocks, nblocks);

    // Generate the actual bases for the upper level and pass it to this
    // function again for generating transfer matrices at successive levels.
    RowLevelMap Ubig_parent;
    ColLevelMap Vbig_parent;

    if (context->is_symmetric) {
      for (int64_t node = 0; node < nblocks; ++node) {
        int64_t child1 = node * 2;
        int64_t child2 = node * 2 + 1;
        int64_t child_level = level + 1;

        if (row_has_admissible_blocks(node, level) && context->height != 1) {
          Matrix& Ubig_child1 = Uchild(child1, child_level);
          Matrix& Ubig_child2 = Uchild(child2, child_level);
          int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;

          Matrix Utransfer  = generate_U_transfer_matrix(Ubig_child1,
                                                         Ubig_child2,
                                                         node,
                                                         block_size,
                                                         level,
                                                         A,
                                                         rand);


          auto Utransfer_splits = Utransfer.split(std::vector<int64_t>(1, Ubig_child1.cols),
                                                  {});
          Matrix Ubig(block_size, Utransfer.cols);

          auto Ubig_splits = Ubig.split(2, 1);

          matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
          matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

          // add to context
          Matrix Vtransfer(Utransfer);
          context->U.insert(node, level, std::move(Utransfer));
          context->V.insert(node, level, std::move(Vtransfer));

          Matrix Vbig(Ubig);
          Ubig_parent.insert(node, level, std::move(Ubig));
          Vbig_parent.insert(node, level, std::move(Vbig));
        }
      }

      for (int64_t row = 0; row < nblocks; ++row) {
        for (int64_t col = 0; col < nblocks; ++col) {
          if (context->is_admissible.exists(row, col, level) &&
              context->is_admissible(row, col, level)) {
            Matrix Sdense = matmul(matmul(Ubig_parent(row, level),
                                          A_splits[row * nblocks + col], true, false),
                                   Vbig_parent(col, level));

            context->S.insert(row, col, level, std::move(Sdense));
          }
        }
      }
    }
    // else {
    //   for (int64_t node = 0; node < nblocks; ++node) {
    //     int64_t child1 = node * 2;
    //     int64_t child2 = node * 2 + 1;
    //     int64_t child_level = level + 1;
    //     int64_t block_size = context->get_block_size(node, level);

    //     if (row_has_admissible_blocks(node, level) && context->height != 1) {
    //       Matrix& Ubig_child1 = Uchild(child1, child_level);
    //       Matrix& Ubig_child2 = Uchild(child2, child_level);

    //       Matrix Utransfer, Stemp;
    //       std::tie(Utransfer, Stemp) = generate_U_transfer_matrix(Ubig_child1,
    //                                                               Ubig_child2,
    //                                                               node,
    //                                                               block_size,
    //                                                               level);

    //       context->U.insert(node, level, std::move(Utransfer));
    //       context->Srow.insert(node, level, std::move(Stemp));

    //       // Generate the full bases to pass onto the parent.
    //       auto Utransfer_splits = context->U(node, level).split(2, 1);
    //       Matrix Ubig(block_size, context->rank);
    //       auto Ubig_splits = Ubig.split(2, 1);

    //       matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0], false, false, 1, 0);
    //       matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1], false, false, 1, 0);

    //       Ubig_parent.insert(node, level, std::move(Ubig));
    //     }

    //     if (col_has_admissible_blocks(node, level) && context->height != 1) {
    //       // Generate V transfer matrix.
    //       Matrix& Vbig_child1 = Vchild(child1, child_level);
    //       Matrix& Vbig_child2 = Vchild(child2, child_level);

    //       Matrix Vtransfer, Stemp;
    //       std::tie(Stemp, Vtransfer) = generate_V_transfer_matrix(Vbig_child1,
    //                                                               Vbig_child2,
    //                                                               node,
    //                                                               block_size,
    //                                                               level);
    //       context->V.insert(node, level, std::move(Vtransfer));
    //       context->Scol.insert(node, level, std::move(Stemp));

    //       // Generate the full bases for passing onto the upper level.
    //       std::vector<Matrix> Vtransfer_splits = context->V(node, level).split(2, 1);
    //       Matrix Vbig(context->rank, block_size);
    //       std::vector<Matrix> Vbig_splits = Vbig.split(1, 2);

    //       matmul(Vtransfer_splits[0], Vbig_child1, Vbig_splits[0], true, true, 1, 0);
    //       matmul(Vtransfer_splits[1], Vbig_child2, Vbig_splits[1], true, true, 1, 0);

    //       Vbig_parent.insert(node, level, transpose(Vbig));
    //     }
    //   }

    //   for (int64_t row = 0; row < nblocks; ++row) {
    //     for (int64_t col = 0; col < nblocks; ++col) {
    //       if (context->is_admissible.exists(row, col, level) &&
    //           context->is_admissible(row, col, level)) {
    //         Matrix dense = generate_p2p_interactions(context->domain, row, col, level,
    //                                                  context->height, context->kernel);

    //         context->S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level),
    //                                                          dense, true, false),
    //                                                   Vbig_parent(col, level)));
    //       }
    //     }
    //   }
    // }

    return {Ubig_parent, Vbig_parent};
  }


  void
  ConstructMiro::construct() {
    int64_t p = 100;
    Matrix A = generate_p2p_matrix(context->domain, context->kernel);
    Matrix rand = generate_random_matrix(context->N, p);
    generate_leaf_nodes(context->domain, A, rand);
    RowLevelMap Uchild = context->U;
    ColLevelMap Vchild = context->V;

    for (int64_t level = context->height-1; level > 0; --level) {
      std::tie(Uchild, Vchild) = generate_transfer_matrices(level, Uchild, Vchild, A, rand);
    }
  }
}