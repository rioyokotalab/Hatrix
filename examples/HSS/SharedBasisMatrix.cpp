#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructAlgorithm::ConstructAlgorithm(SharedBasisMatrix* context) : context(context) {}

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
    std::vector<Hatrix::Matrix> Y;

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (context->is_admissible.exists(i, j, context->height) &&
            !context->is_admissible(i, j, context->height)) {
          context->D.insert(i, j, context->height,
                            generate_p2p_interactions(context->domain, i, j, context->kernel));
        }
      }
    }

    for (int64_t i = 0; i < nblocks; ++i) {
      Y.push_back(generate_random_matrix(domain.boxes[i].num_particles,
                                         context->rank + oversampling));
    }

    // Generate U leaf blocks
    for (int64_t i = 0; i < nblocks; ++i) {
      Matrix Utemp, Stemp;
      std::tie(Utemp, Stemp) =
        generate_column_bases(i, domain.boxes[i].num_particles, context->height);
      context->U.insert(i, context->height, std::move(Utemp));
    }

    // Generate V leaf blocks
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix Stemp, Vtemp;
      std::tie(Stemp, Vtemp) = generate_row_bases(j, domain.boxes[j].num_particles, context->height);
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

    std::vector<Matrix> Y;
    // Generate the actual bases for the upper level and pass it to this
    // function again for generating transfer matrices at successive levels.
    RowLevelMap Ubig_parent;
    ColLevelMap Vbig_parent;

    for (int64_t i = 0; i < nblocks; ++i) {
      int64_t block_size = context->get_block_size(i, level);
      Y.push_back(generate_random_matrix(block_size, context->rank + oversampling));
    }

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

  ConstructID_Random::ConstructID_Random(SharedBasisMatrix* context) :
    ConstructAlgorithm(context) {}

  std::tuple<std::vector<std::vector<int64_t>>, std::vector<Matrix>, std::vector<Matrix>>
  ConstructID_Random::generate_leaf_blocks(const Matrix& samplesT, const Matrix& OMEGA) {
    std::vector<std::vector<int64_t>> row_indices;
    std::vector<Matrix> S_loc_blocks, OMEGA_blocks;

    return {std::move(row_indices), std::move(S_loc_blocks), std::move(OMEGA_blocks)};
  }

  void
  ConstructID_Random::construct() {
    Matrix dense = generate_p2p_matrix(context->domain, context->kernel);
    Matrix OMEGA = generate_random_matrix(p, context->N);
    // obtain the transposed samples so that we dont need to transpose for the ID.
    Matrix samplesT = Hatrix::matmul(OMEGA, dense, false, true);

    // begin construction procedure using randomized samples.
    std::vector<std::vector<int64_t>> row_indices;
    std::vector<Matrix> S_loc_blocks, OMEGA_blocks;

    for (int64_t level = context->height; level > 0; --level) {
      if (level == context->height) {
        std::tie(row_indices, S_loc_blocks, OMEGA_blocks) =
          generate_leaf_blocks(samplesT, OMEGA);
      }
    }
  }

  int64_t
  SharedBasisMatrix::get_block_size(int64_t parent, int64_t level) {
    if (level == height) {
      return domain.boxes[parent].num_particles;
    }
    int64_t child_level = level + 1;
    int64_t child1 = parent * 2;
    int64_t child2 = parent * 2 + 1;

    return get_block_size(child1, child_level) + get_block_size(child2, child_level);
  }

  void
  SharedBasisMatrix::coarsen_blocks(int64_t level) {
    int64_t child_level = level + 1;
    int64_t nblocks = pow(2, level);
    for (int64_t i = 0; i < nblocks; ++i) {
      std::vector<int64_t> row_children({i * 2, i * 2 + 1});
      for (int64_t j = 0; j < nblocks; ++j) {
        std::vector<int64_t> col_children({j * 2, j * 2 + 1});

        bool admis_block = true;
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
                !is_admissible(row_children[c1], col_children[c2], child_level)) {
              admis_block = false;
            }
          }
        }

        if (admis_block) {
          for (int64_t c1 = 0; c1 < 2; ++c1) {
            for (int64_t c2 = 0; c2 < 2; ++c2) {
              is_admissible.erase(row_children[c1], col_children[c2], child_level);
            }
          }
        }

        is_admissible.insert(i, j, level, std::move(admis_block));
      }
    }
  }

  void
  SharedBasisMatrix::calc_diagonal_based_admissibility(int64_t level) {
    int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
    level_blocks.push_back(nblocks);
    if (level == 0) { return; }
    if (level == height) {
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, level, std::abs(i - j) > admis);
        }
      }
    }
    else {
      coarsen_blocks(level);
    }

    calc_diagonal_based_admissibility(level-1);
  }

  Matrix
  SharedBasisMatrix::get_Ubig(int64_t node, int64_t level) {
    if (level == height) {
      return U(node, level);
    }

    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;

    Matrix Ubig_child1 = get_Ubig(child1, level+1);
    Matrix Ubig_child2 = get_Ubig(child2, level+1);

    int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;

    Matrix Ubig(block_size, rank);

    std::vector<Matrix> Ubig_splits = Ubig.split(
                                                 std::vector<int64_t>(1,
                                                                      Ubig_child1.rows),
                                                 {});

    std::vector<Matrix> U_splits = U(node, level).split(2, 1);

    matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
    matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

    return Ubig;
  }

  Matrix
  SharedBasisMatrix::get_Vbig(int64_t node, int64_t level) {
    if (level == height) {
      return V(node, height);
    }

    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;

    Matrix Vbig_child1 = get_Vbig(child1, level+1);
    Matrix Vbig_child2 = get_Vbig(child2, level+1);

    int64_t block_size = Vbig_child1.rows + Vbig_child2.rows;

    Matrix Vbig(block_size, rank);

    std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
    std::vector<Matrix> V_splits = V(node, level).split(2, 1);

    matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
    matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

    return Vbig;
  }


  double
  SharedBasisMatrix::construction_error() {
    double error = 0;
    double dense_norm = 0;
    int64_t nblocks = level_blocks[height];

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
          Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j, kernel);
          Matrix expected = D(i, j, height);
          error += pow(norm(actual - expected), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }
    }

    for (int64_t level = height; level > 0; --level) {
      int64_t nblocks = level_blocks[level];

      for (int64_t row = 0; row < nblocks; ++row) {
        for (int64_t col = 0; col < nblocks; ++col) {
          if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
            Matrix Ubig = get_Ubig(row, level);
            Matrix Vbig = get_Vbig(col, level);

            Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)),
                                            Vbig, false, true);
            Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, row, col, level,
                                                height, kernel);

            dense_norm += pow(norm(actual_matrix), 2);
            error += pow(norm(expected_matrix - actual_matrix), 2);
          }
        }
      }
    }

    return std::sqrt(error / dense_norm);
  }

  Matrix
  SharedBasisMatrix::matvec(const Matrix& x) {
    int leaf_nblocks = level_blocks[height];
    std::vector<Matrix> x_hat;
    auto x_splits = x.split(leaf_nblocks, 1);

    // V leaf nodes
    for (int i = 0; i < leaf_nblocks; ++i) {
      x_hat.push_back(matmul(V(i, height), x_splits[i], true, false, 1.0));
    }

    int x_hat_offset = 0;

    // V non-leaf nodes
    for (int level = height-1; level > 0; --level) {
      int nblocks = level_blocks[level];
      int child_level = level + 1;
      for (int i = 0; i < nblocks; ++i) {
        int child1 = i * 2;
        int child2 = i * 2 + 1;

        Matrix xtemp = Matrix(V(i, level).rows, 1);
        auto xtemp_splits =
          xtemp.split(std::vector<int64_t>(1, V(child1, child_level).cols),
                      {});
        xtemp_splits[0] = x_hat[x_hat_offset + child1];
        xtemp_splits[1] = x_hat[x_hat_offset + child2];

        x_hat.push_back(matmul(V(i, level), xtemp, true, false, 1.0));
      }

      x_hat_offset += level_blocks[level+1];
    }
    int level = 1;

    // b_hat does the product in reverse so matrices are pushed from the back.
    std::vector<Matrix> b_hat;

    // Multiply the S blocks at the top-most level with the corresponding xhat.
    Matrix b1_1 = matmul(S(0, 1, level), x_hat[x_hat_offset+1]);
    Matrix b1_2 = matmul(S(1, 0, level), x_hat[x_hat_offset]);
    b_hat.push_back(b1_1);
    b_hat.push_back(b1_2);
    int b_hat_offset = 0;

    for (int level = 1; level < height; ++level) {
      int nblocks = level_blocks[level];
      int child_level = level + 1;
      x_hat_offset -= level_blocks[child_level];

      for (int row = 0; row < nblocks; ++row) {
        int c_r1 = row * 2, c_r2 = row * 2 + 1;

        Matrix Ub = matmul(U(row, level),
                           b_hat[b_hat_offset + row]);
        auto Ub_splits = Ub.split(2, 1);

        Matrix b_r1_cl = matmul(S(c_r1, c_r2, child_level),
                                x_hat[x_hat_offset + c_r2]);
        b_hat.push_back(b_r1_cl + Ub_splits[0]);

        Matrix b_r2_cl = matmul(S(c_r2, c_r1, child_level),
                                x_hat[x_hat_offset + c_r1]);
        b_hat.push_back(b_r2_cl + Ub_splits[1]);
      }

      b_hat_offset += level_blocks[level];
    }

    // multiply the leaf level U block with the generated b_hat vectors
    // and add the product with the corresponding x blocks.
    Matrix b(x.rows, 1);
    auto b_splits = b.split(leaf_nblocks, 1);
    for (int i = 0; i < leaf_nblocks; ++i) {
      Matrix temp = matmul(U(i, height), b_hat[b_hat_offset + i]) +
        matmul(D(i, i, height), x_splits[i]);
      b_splits[i] = temp;
    }

    return b;
  }

  SharedBasisMatrix::SharedBasisMatrix(int64_t N, int64_t nleaf, int64_t rank,
                                       double accuracy, double admis,
                                       ADMIS_KIND admis_kind,
                                       CONSTRUCT_ALGORITHM construct_algorithm,
                                       bool use_shared_basis,
                                       const Domain& domain,
                                       const kernel_function& kernel) :
    N(N), nleaf(nleaf), rank(rank), accuracy(accuracy), admis(admis),
    admis_kind(admis_kind), construct_algorithm(construct_algorithm),
    use_shared_basis(use_shared_basis), domain(domain), kernel(kernel)
  {
    if (use_shared_basis) {
      height = int64_t(log2(N / nleaf));
      calc_diagonal_based_admissibility(height);
      std::reverse(std::begin(level_blocks), std::end(level_blocks));
    }
    else {
      height = 1;
      int64_t nblocks = domain.boxes.size();
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, height, std::abs(i - j) > admis);
        }
      }
      level_blocks.push_back(1);
      level_blocks.push_back(nblocks);
    }
    is_admissible.insert(0, 0, 0, false);

    ConstructAlgorithm *construct_algo;
    if (construct_algorithm == MIRO) {
      construct_algo = new ConstructMiro(this);
    }
    else if (construct_algorithm == ID_RANDOM) {
      construct_algo = new ConstructID_Random(this);
    }

    construct_algo->construct();

    delete construct_algo;
  };
}
