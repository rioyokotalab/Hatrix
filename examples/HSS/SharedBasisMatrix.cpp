#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "ConstructMiro.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructAlgorithm::ConstructAlgorithm(SharedBasisMatrix* context) : context(context) {}

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
    Matrix b1_2 = matmul(S(1, 0, level), x_hat[x_hat_offset], true, false);
    Matrix b1_1 = matmul(S(1, 0, level), x_hat[x_hat_offset]);
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
        auto Ub_splits = Ub.split(std::vector<int64_t>(1, U(c_r1, child_level).cols),
                                  {});

        Matrix b_r1_cl = matmul(S(c_r2, c_r1, child_level),
                                x_hat[x_hat_offset + c_r2],
                                true, false);
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
                                       const kernel_function& kernel,
                                       const bool is_symmetrix) :
    N(N), nleaf(nleaf), rank(rank), accuracy(accuracy), admis(admis),
    admis_kind(admis_kind), construct_algorithm(construct_algorithm),
    use_shared_basis(use_shared_basis), domain(domain), kernel(kernel),
    is_symmetric(is_symmetric)
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
