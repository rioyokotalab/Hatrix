#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;
Hatrix::greens_functions::kernel_function_t kernel;

void
construct_H2_weak_2level_leaf_nodes(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                                    int64_t N, int64_t nleaf, int64_t rank) {
  Hatrix::Matrix Utemp, Stemp, Vtemp;
  double error;
  const int64_t nblocks = N / nleaf;
  // populate diagonal blocks.
  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                   i * nleaf, nleaf,
                                                   i * nleaf, nleaf,
                                                   kernel);
    A.D.insert(i, i, A.max_level, std::move(Aij));
  }

  // populate leaf level shared bases.
  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix AY(nleaf, nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (i != j) {
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         i * nleaf, nleaf,
                                                         j * nleaf, nleaf,
                                                         kernel);
        AY += dense;
      }
    }
    std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, A.max_level, std::move(Utemp));
  }

  // Update S blocks.
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (A.is_admissible(i, j, A.max_level)) {
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         i * nleaf, nleaf,
                                                         j * nleaf, nleaf,
                                                         kernel);
        A.S.insert(i, j, A.max_level,
                   Hatrix::matmul(Hatrix::matmul(A.U(i, A.max_level), dense, true), A.U(j, A.max_level)));
      }
    }
  }
}

Matrix
generate_actual_bases(Hatrix::SymmetricSharedBasisMatrix& A, const int64_t N,
                      const int64_t p, const int64_t rank) {
  int64_t child1 = p * 2;
  int64_t child2 = p * 2 + 1;
  int64_t leaf_size = N / 2;
  Matrix Ubig(leaf_size, rank);

  std::vector<Matrix> Ubig_splits = Ubig.split(2, 1);
  std::vector<Matrix> U_splits = A.U(p, A.max_level-1).split(2, 1);

  matmul(A.U(child1, A.max_level), U_splits[0], Ubig_splits[0]);
  matmul(A.U(child2, A.max_level), U_splits[1], Ubig_splits[1]);

  return Ubig;
}

void
generate_transfer_matrices(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                           const int64_t N, const int64_t nleaf, const int64_t rank) {
  Matrix Ui, Si, _Vi; double error;
  int64_t block_size = N / 2;
  int64_t level = A.max_level - 1;
  int64_t nblocks = pow(2, level);
  Matrix AY(block_size, block_size);

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < nblocks; ++col) {
      if (row != col) {             // admit only admissible blocks.
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         row * block_size, block_size,
                                                         col * block_size, block_size,
                                                         kernel);
        AY += dense;
      }
    }

    int child1 = row * 2;
    int child2 = row * 2 + 1;

    // Generate U transfer matrix.
    Matrix& Ubig_child1 = A.U(child1, A.max_level);
    Matrix& Ubig_child2 = A.U(child2, A.max_level);
    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
    std::vector<Matrix> temp_splits = temp.split(2, 1);
    std::vector<Matrix> AY_splits = AY.split(2, 1);

    matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

    std::tie(Ui, Si, _Vi, error) = truncated_svd(temp, rank);
    A.U.insert(row, level, std::move(Ui));
  }

  for (int64_t row = 0; row < 2; ++row) {
    int64_t col = (row % 2 == 0) ? row + 1 : row - 1;
    Matrix Urow_actual = generate_actual_bases(A, N, row, rank);
    Matrix Ucol_actual = generate_actual_bases(A, N, col, rank);

    Matrix dense = generate_p2p_interactions(domain,
                                             row * block_size, block_size,
                                             col * block_size, block_size,
                                             kernel);

    Matrix S_block = matmul(matmul(Urow_actual, dense, true), Ucol_actual);
    A.S.insert(row, col, level, std::move(dense));
  }
}

void
construct_H2_weak_2level(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                         int64_t N, int64_t nleaf, int64_t rank, double admis) {
  construct_H2_weak_2level_leaf_nodes(A, domain, N, nleaf, rank);
  generate_transfer_matrices(A, domain, N, nleaf, rank);
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
matmul(SymmetricSharedBasisMatrix& A, Matrix& x, int64_t N, int64_t rank) {
  std::vector<Matrix> x_hat;
  const int64_t leaf_nblocks = pow(2, A.max_level);
  auto x_splits = x.split(leaf_nblocks, 1);

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

      Matrix xtemp = Matrix(A.U(i, level).rows, 1);
      auto xtemp_splits = xtemp.split(std::vector<int64_t>(1, rank),
                                      {});
      xtemp_splits[0] = x_hat[x_hat_offset + c1];
      xtemp_splits[1] = x_hat[x_hat_offset + c2];

      x_hat.push_back(matmul(A.U(i, level), xtemp, true, false, 1.0));
    }

    x_hat_offset += pow(2, child_level);
  }

  std::vector<Matrix> b_hat;
  for (int64_t i = 0; i < pow(2, A.min_level); ++i) {
    b_hat.push_back(Matrix(rank, 1));
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
  auto b_splits = b.split(leaf_nblocks, 1);
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

int main(int argc, char *argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  const double admis = 0;
  const int64_t height = 2;

  assert(N / nleaf == 4);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  // Assign kernel function
  double add_diag = 1e-4;
  kernel = [&](const std::vector<double>& c_row,
               const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_1d_kernel(c_row, c_col, add_diag);
  };

  // Define a 1D grid geometry using the Domain class.
  const int64_t ndim = 1;
  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  // Initialize a Symmetric shared basis matrix container.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / nleaf);

  // Use a simple distance from diagonal based admissibility condition. admis is kept
  // at 0 since this is a weakly admissible code.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DIAGONAL_ADMIS, admis);

  // Construct a 2-level weak admissibility H2 matrix.
  construct_H2_weak_2level(A, domain, N, nleaf, rank, admis);

  // Verification of the construction with a matvec product.
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);

  // Low rank matrix-vector product.
  Hatrix::Matrix b_lowrank = matmul(A, x, N, rank);

  // Generate a dense matrix.
  Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
  Matrix b_dense = Hatrix::matmul(A_dense, x);
  Matrix diff = b_dense - b_lowrank;
  double rel_error = Hatrix::norm(diff) / Hatrix::norm(b_dense);

  std::cout << "Error : " << rel_error << std::endl;

  return 0;
}
