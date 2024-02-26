#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;
static Hatrix::greens_functions::kernel_function_t kernel;

static bool
row_has_admissible_blocks(const Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t row, const int64_t level) {
  bool has_admis = false;
  for (int64_t col = 0; col < pow(2, level); col++) {
    if ((!A.is_admissible.exists(row, col, level)) || // part of upper level admissible block
        (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level))) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

static std::tuple<Matrix, Matrix, Matrix, int64_t>
svd_like_compression(Matrix& matrix,
                     const int64_t max_rank,
                     const double accuracy) {
  Matrix Ui, Si, Vi;
  int64_t rank;
  std::tie(Ui, Si, Vi, rank) = error_svd(matrix, accuracy, false, false);

  // Assume fixed rank if accuracy==0.
  rank = accuracy == 0. ? max_rank : std::min(max_rank, rank);

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi), std::move(rank));
}


static RowLevelMap
generate_H2_strong_transfer_matrices(Hatrix::SymmetricSharedBasisMatrix& A,
                                     RowLevelMap Uchild,
                                     const Hatrix::Domain& domain,
                                     const int64_t N, const int64_t nleaf, const int64_t max_rank,
                                     const int64_t level, const double accuracy) {
  Matrix Ui, Si, _Vi; double error; int64_t rank;
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = N / nblocks;
  Matrix AY(block_size, block_size);
  RowLevelMap Ubig_parent;
  const int64_t child_level = level + 1;

  for (int64_t row = 0; row < nblocks; ++row) {
    if (row_has_admissible_blocks(A, row, level)) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (A.is_admissible.exists(row, col, level) &&
            A.is_admissible(row, col, level)) {
          Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                           row * block_size, block_size,
                                                           col * block_size, block_size,
                                                           kernel);
          AY += dense;
        }
      }

      int64_t child1 = row * 2;
      int64_t child2 = row * 2 + 1;

      // Generate U transfer matrix.
      Matrix& Ubig_child1 = Uchild(child1, child_level);
      Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
      std::vector<Matrix> temp_splits = temp.split(std::vector<int64_t>{Ubig_child1.cols},
                                                   std::vector<int64_t>{});
      std::vector<Matrix> AY_splits = AY.split(std::vector<int64_t>{Ubig_child1.rows},
                                               std::vector<int64_t>{});

      matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
      matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

      // std::tie(Ui, Si, _Vi, error) = truncated_svd(temp, rank);
      std::tie(Ui, Si, _Vi, rank) = svd_like_compression(temp, max_rank, accuracy);
      Ui.shrink(Ui.rows, rank);
      std::cout << "rank: " << rank << std::endl;

      // Generate the full basis to pass to the next level.
      auto Utransfer_splits = Ui.split(std::vector<int64_t>{Ubig_child1.cols}, {});

      Matrix Ubig(block_size, rank);
      auto Ubig_splits = Ubig.split(std::vector<int64_t>{Ubig_child1.rows},
                                    {});
      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

      A.U.insert(row, level, std::move(Ui));
      Ubig_parent.insert(row, level, std::move(Ubig));
    }
  }

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < row; ++col) {
      if (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level)) {
        Matrix& Urow_actual = Ubig_parent(row, level);
        Matrix& Ucol_actual = Ubig_parent(col, level);

        Matrix dense = generate_p2p_interactions(domain,
                                                 row * block_size, block_size,
                                                 col * block_size, block_size,
                                                 kernel);
        Matrix S_block = matmul(matmul(Urow_actual, dense, true), Ucol_actual);
        A.S.insert(row, col, level, std::move(S_block));
      }
    }
  }

  return Ubig_parent;
}

static void
construct_H2_strong_leaf_nodes(Hatrix::SymmetricSharedBasisMatrix& A,
                               const Hatrix::Domain& domain,
                               const int64_t N, const int64_t nleaf,
                               const int64_t max_rank, const double accuracy) {
  Hatrix::Matrix Utemp, Stemp, Vtemp;
  int64_t rank;
  const int64_t nblocks = pow(2, A.max_level);


  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) && !A.is_admissible(i, j, A.max_level)) {
        Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
        A.D.insert(i, j, A.max_level, std::move(Aij));
      }
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix AY(nleaf, nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (!A.is_admissible.exists(i, j, A.max_level) ||
          (A.is_admissible.exists(i, j, A.max_level) && A.is_admissible(i, j, A.max_level))) {
        Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
        AY += Aij;
      }
    }

    // std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
    std::tie(Utemp, Stemp, Vtemp, rank) = svd_like_compression(AY, max_rank, accuracy);
    Utemp.shrink(Utemp.rows, rank);
    std::cout << "leaf rank: " << rank << std::endl;
    A.U.insert(i, A.max_level, std::move(Utemp));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
      A.S.insert(i, j, A.max_level,
                   Hatrix::matmul(Hatrix::matmul(A.U(i, A.max_level), dense, true),
                                  A.U(j, A.max_level)));
    }
  }
}

static void
construct_H2_strong(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Domain& domain,
                    const int64_t N, const int64_t nleaf, const int64_t max_rank, const double accuracy) {
  construct_H2_strong_leaf_nodes(A, domain, N, nleaf, max_rank, accuracy);
  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level - 1; level > 0; --level) {
    Uchild = generate_H2_strong_transfer_matrices(A, Uchild, domain, N, nleaf, max_rank, level, accuracy);
  }
}

static Matrix get_Ubig(const Hatrix::SymmetricSharedBasisMatrix& A,
                       const int64_t node, const int64_t level) {
  if (level == A.max_level) {
    return A.U(node, level);
  }

  const int64_t child1 = node * 2;
  const int64_t child2 = node * 2 + 1;
  const Matrix Ubig_child1 = get_Ubig(A, child1, level + 1);
  const Matrix Ubig_child2 = get_Ubig(A, child2, level + 1);

  const int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, A.U(node, level).cols);
  auto Ubig_splits = Ubig.split(std::vector<int64_t>{Ubig_child1.rows},
                                std::vector<int64_t>{});
  auto U_splits = A.U(node, level).split(std::vector<int64_t>{Ubig_child1.cols},
                                         std::vector<int64_t>{});

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

static double construction_absolute_error(const Hatrix::SymmetricSharedBasisMatrix& A,
                                          const int64_t nleaf,
                                          const Domain& domain) {
  double error = 0;
  const int64_t leaf_nblocks = pow(2, A.max_level);
  // Inadmissible blocks (only at leaf level)
  for (int64_t i = 0; i < leaf_nblocks; i++) {
    for (int64_t j = 0; j < leaf_nblocks; j++) {
      if (A.is_admissible.exists(i, j, A.max_level) && !A.is_admissible(i, j, A.max_level)) {
        const Matrix actual = generate_p2p_interactions(domain,
                                                        i * nleaf, nleaf,
                                                        j * nleaf, nleaf,
                                                        kernel);
        const Matrix expected = A.D(i, j, A.max_level);
        error += pow(norm(actual - expected), 2);
      }
    }
  }
  // Admissible blocks
  for (int64_t level = A.max_level; level > 0; level--) {
    const int64_t nblocks = pow(2, level);
    for (int64_t i = 0; i < nblocks; i++) {
      for (int64_t j = 0; j < i; j++) {
        if (A.is_admissible.exists(i, j, level) &&
            A.is_admissible(i, j, level)) {
          const int64_t block_size = domain.N / pow(2, level);
          const Matrix Ubig = get_Ubig(A, i, level);
          const Matrix Vbig = get_Ubig(A, j, level);
          const Matrix expected_matrix = matmul(matmul(Ubig, A.S(i, j, level)), Vbig, false, true);
          const Matrix actual_matrix =
            generate_p2p_interactions(domain,
                                      i * block_size, block_size,
                                      j * block_size, block_size,
                                      kernel);
          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }
  return std::sqrt(error);
}

int main(int argc, char ** argv) {
  if (argc == 1) {
    std::cout << "HELP SCREEN FOR H2_strong_CON.cpp" << std::endl;
    std::cout << "Specify arguments as follows: " << std::endl;
    std::cout << "N leaf_size accuracy max_rank admis kernel_type geom_type ndim matrix_type" << std::endl;
    return 0;
  }

  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-9;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const double admis = argc > 5 ? atof(argv[5]) : 1.0;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 6 ? atol(argv[6]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  const int64_t geom_type = argc > 7 ? atol(argv[7]) : 0;
  const int64_t ndim  = argc > 8 ? atol(argv[8]) : 2;
  assert(ndim >= 1 && ndim <= 3);

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 9 ? atol(argv[9]) : 1;

  const double add_diag = 1e-6 / N;
  const double alpha = 1;
  // Setup the kernel.
  switch (kernel_type) {
  case 1:                       // yukawa
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return Hatrix::greens_functions::yukawa_kernel(c_row, c_col, alpha, add_diag);
    };
    break;
  default:                      // laplace
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      if (ndim == 1) {
        return Hatrix::greens_functions::laplace_1d_kernel(c_row, c_col, add_diag);
      }
      else if (ndim == 2) {
        return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, add_diag);
      }
      else {
        return Hatrix::greens_functions::laplace_3d_kernel(c_row, c_col, add_diag);
      }
    };
  }

  // Setup the Domain
  Hatrix::Domain domain(N, ndim);
  switch(geom_type) {
  case 1:                       // cube mesh
    domain.generate_grid_particles();
  default:                      // circle / sphere mesh
    domain.generate_circular_particles();
  }
  domain.cardinal_sort_and_cell_generation(leaf_size);

  // Initialize H2 matrix class.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / leaf_size);

  A.generate_admissibility(domain, matrix_type == 1, Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);
  A.print_structure();

  construct_H2_strong(A, domain, N, leaf_size, max_rank, accuracy);

  double error = construction_absolute_error(A, leaf_size, domain);

  std::cout << "Construction absolute error: " << error << std::endl;

  return 0;
}
