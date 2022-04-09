#include <cmath>

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "SharedBasisMatrix.hpp"
#include "functions.hpp"

namespace Hatrix {
  ConstructAlgorithm::ConstructAlgorithm(SharedBasisMatrix* context) : context(context) {}

  ConstructMiro::ConstructMiro(SharedBasisMatrix* context) : ConstructAlgorithm(context) {}

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
      Y.push_back(generate_random_matrix(domain.boxes[i].num_particles, context->rank + oversampling));
    }

    // Generate U leaf blocks
    for (int64_t i = 0; i < nblocks; ++i) {
      Matrix Utemp, Stemp;
      std::tie(Utemp, Stemp) =
        generate_column_bases(i, domain.boxes[i].num_particles, context->domain, Y, context->height);
      context->U.insert(i, context->height, std::move(Utemp));
    }

    // Generate V leaf blocks
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix Stemp, Vtemp;
      std::tie(Stemp, Vtemp) =
        generate_row_bases(j, domain.boxes[j].num_particles,
                           context->domain, Y, context->height);
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


  void
  ConstructMiro::construct() {
    generate_leaf_nodes(context->domain);
    RowLevelMap Uchild = context->U;
    ColLevelMap Vchild = context->V;

    for (int64_t level = context->height-1; level > 0; --level) {
      std::tie(Uchild, Vchild) = generate_transfer_matrices(context->domain, level, Uchild, Vchild);
    }
  }

  ConstructID_Random::ConstructID_Random(SharedBasisMatrix* context) : ConstructAlgorithm(context) {}

  void
  ConstructID_Random::construct() {

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

  double
  SharedBasisMatrix::construction_error() {

  }

  SharedBasisMatrix::SharedBasisMatrix(int64_t N, int64_t nleaf, int64_t rank, double accuracy,
                                       double admis, ADMIS_KIND admis_kind,
                                       CONSTRUCT_ALGORITHM construct_algorithm, bool use_shared_basis,
                                       const Domain& domain, const kernel_function& kernel) :
    N(N), nleaf(nleaf), rank(rank), accuracy(accuracy), admis(admis), admis_kind(admis_kind),
    construct_algorithm(construct_algorithm), use_shared_basis(use_shared_basis),
    domain(domain), kernel(kernel)
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
  };
}
