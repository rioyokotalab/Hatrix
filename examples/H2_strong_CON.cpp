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
  std::tie(Ui, Si, Vi, rank) = error_svd(matrix, accuracy, true, false);

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
    const int64_t child1 = row * 2;
    const int64_t child2 = row * 2 + 1;

    // Generate U transfer matrix.
    const Matrix& Ubig_child1 = Uchild(child1, child_level);
    const Matrix& Ubig_child2 = Uchild(child2, child_level);

    if (row_has_admissible_blocks(A, row, level)) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (!A.is_admissible.exists(row, col, level) ||
            (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level))) {
          Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                           row * block_size, block_size,
                                                           col * block_size, block_size,
                                                           kernel);
          AY += dense;
        }
      }

      Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
      std::vector<Matrix> temp_splits = temp.split(std::vector<int64_t>{Ubig_child1.cols},
                                                   std::vector<int64_t>{});
      std::vector<Matrix> AY_splits = AY.split(2, 1);

      matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
      matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

      std::tie(Ui, Si, _Vi, rank) = svd_like_compression(temp, max_rank, accuracy);
      Ui.shrink(Ui.rows, rank);
      A.U.insert(row, level, std::move(Ui));

      // Generate the full basis to pass to the next level.
      auto Utransfer_splits = A.U(row, level).split(std::vector<int64_t>{Ubig_child1.cols}, {});

      Matrix Ubig(block_size, rank);
      auto Ubig_splits = Ubig.split(std::vector<int64_t>{Ubig_child1.rows},
                                    {});
      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

      Ubig_parent.insert(row, level, std::move(Ubig));
    }
    else {                      // insert an identity block if there are no admissible blocks.
      A.U.insert(row, level, generate_identity_matrix(max_rank * 2, max_rank));
      Ubig_parent.insert(row, level, generate_identity_matrix(block_size, max_rank));
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
        Matrix S_block = matmul(matmul(Urow_actual, dense, true, false), Ucol_actual);
        A.S.insert(row, col, level, std::move(S_block));
      }
    }
  }

  return std::move(Ubig_parent);
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

    std::tie(Utemp, Stemp, Vtemp, rank) = svd_like_compression(AY, max_rank, accuracy);
    Utemp.shrink(Utemp.rows, rank);
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
                    const int64_t N, const int64_t nleaf, const int64_t max_rank,
                    const double accuracy) {
  construct_H2_strong_leaf_nodes(A, domain, N, nleaf, max_rank, accuracy);
  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    Uchild = generate_H2_strong_transfer_matrices(A, Uchild, domain, N, nleaf,
                                                  max_rank, level, accuracy);
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

static void
multiply_S(const Hatrix::SymmetricSharedBasisMatrix& A,
           std::vector<Matrix>& x_hat, std::vector<Matrix>& b_hat,
           int64_t x_hat_offset, int64_t b_hat_offset, int64_t level) {
  int64_t nblocks = pow(2, level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
        matmul(A.S(i, j, level), x_hat[x_hat_offset + j], b_hat[b_hat_offset + i]);
        matmul(A.S(i, j, level), x_hat[x_hat_offset + i], b_hat[b_hat_offset + j],
               true, false);
      }
    }
  }
}


static Matrix
matmul(const SymmetricSharedBasisMatrix& A, Matrix& x, int64_t N, int64_t rank) {
  std::vector<Matrix> x_hat;
  const int64_t leaf_nblocks = pow(2, A.max_level);
  std::vector<int64_t> split_indices;
  for (int64_t i = 0; i < leaf_nblocks-1; ++i) {
    int64_t prev = i == 0 ? 0 : split_indices[i-1];
    split_indices.push_back(A.U(i, A.max_level).rows + prev);
  }

  auto x_splits = x.split(split_indices, {});

  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  int64_t x_hat_offset = 0;
  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level + 1;

    for (int64_t i = 0; i < nblocks; ++i) {
      int64_t c1 = i * 2;
      int64_t c2 = i * 2 + 1;

      Matrix xtemp = Matrix(A.U(c1, child_level).cols + A.U(c2, child_level).cols, 1);
      auto xtemp_splits = xtemp.split(std::vector<int64_t>{A.U(c1, child_level).cols},
                                      {});
      xtemp_splits[0] = x_hat[x_hat_offset + c1];
      xtemp_splits[1] = x_hat[x_hat_offset + c2];

      int64_t pad_rows = A.U(i, level).rows - xtemp.rows;
      int64_t pad_cols = 0;
      const Matrix xtemp_pad = xtemp.pad(pad_rows, pad_cols);
      x_hat.push_back(matmul(A.U(i, level), xtemp_pad, true, false, 1.0));
    }

    x_hat_offset += pow(2, child_level);
  }

  std::vector<Matrix> b_hat;
  for (int64_t i = 0; i < pow(2, A.min_level); ++i) {
    b_hat.push_back(Matrix(A.U(i, A.min_level).cols, 1));
  }

  int64_t b_hat_offset = 0;
  multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset, A.min_level);

  for (int64_t level = A.min_level; level < A.max_level; ++level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level+1;

    x_hat_offset -= pow(2, child_level);

    for (int64_t row = 0; row < nblocks; ++row) {
      int c_r1 = row * 2, c_r2 = row * 2 + 1;
      Matrix Ub = matmul(A.U(row, level),
                         b_hat[b_hat_offset + row]);
      auto Ub_splits = Ub.split(std::vector<int64_t>(1, A.U(c_r1, child_level).cols),
                                {});

      b_hat.push_back(Matrix(Ub_splits[0], true));
      b_hat.push_back(Matrix(Ub_splits[1], true));
    }
    multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset + nblocks, child_level);
    b_hat_offset += nblocks;
  }

  // Multiply with the leaf level transfer matrices to generate the product matrix.
  Matrix b(x.rows, 1);
  auto b_splits = b.split(split_indices, {});
  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.U(i, A.max_level), b_hat[b_hat_offset + i], b_splits[i]);
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        // TODO: make the diagonal tringular and remove this.
        if (i == j) {
          matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
        }
        else {
          matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
          matmul(A.D(i, j, A.max_level), x_splits[i], b_splits[j], true, false);
        }
      }
    }
  }

  return b;
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

  const double add_diag = 1e-3 / N;
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
    break;
  default:                      // circle / sphere mesh
    domain.generate_circular_particles();
  }
  domain.cardinal_sort_and_cell_generation(leaf_size);

  // Initialize H2 matrix class.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / leaf_size);
  A.generate_admissibility(domain, matrix_type == 1,
                           Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);
  // Construct H2 strong admis matrix.
  construct_H2_strong(A, domain, N, leaf_size, max_rank, accuracy);

  // Verification of the construction with a matvec product.
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);

  // Low rank matrix-vector product.
  Hatrix::Matrix b_lowrank = matmul(A, x, N, max_rank);

  // Generate a dense matrix.
  Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
  Matrix b_dense = Hatrix::matmul(A_dense, x);
  Matrix diff = b_dense - b_lowrank;
  double rel_error = Hatrix::norm(diff) / Hatrix::norm(b_dense);

  std::cout << "N=" << N << " nleaf=" << leaf_size << " acc=" << accuracy
            << " max_rank=" << max_rank << " admis=" << admis
            << " kernel= " << (kernel_type == 0 ? "laplace" : "yukawa")
            << " geom_type= " << (geom_type == 0 ? "circle" : "grid")
            << " ndim= " << ndim
            << " matrix_type= " << (matrix_type == 1 ? "H2" : "BLR2")
            << " Error : " << rel_error
            << std::endl;

  return 0;
}
