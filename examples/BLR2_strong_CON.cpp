#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>
#include <random>
#include <string>

#include "omp.h"

#include "Hatrix/Hatrix.hpp"

using namespace Hatrix;

Hatrix::greens_functions::kernel_function_t kernel;

// Construction of BLR2 strong admis matrix based on geometry based admis condition.
void
construct_strong_BLR2(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                      int64_t N, int64_t nleaf, int64_t rank, double admis) {
  int64_t nblocks = N / nleaf;
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (!A.is_admissible(i, j, A.max_level)) {
        Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                               i * nleaf, nleaf,
                                               j * nleaf, nleaf,
                                               kernel);
        A.D.insert(i, j, A.max_level, std::move(Aij));
      }
    }
  }

  Hatrix::Matrix Utemp, Stemp, Vtemp;
  double error;

  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix AY(nleaf, nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible(i, j, A.max_level)) {
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

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
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

Hatrix::Matrix
matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x,
       const int64_t N, const int64_t rank) {
  int64_t leaf_nblocks = pow(2, A.max_level);
  Matrix b(N, 1);

  std::vector<Matrix> x_hat, b_hat;
  auto x_splits = x.split(leaf_nblocks, 1);
  auto b_splits = b.split(leaf_nblocks, 1);

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    b_hat.push_back(Hatrix::Matrix(rank, 1));
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j < leaf_nblocks; ++j) {
      if (A.is_admissible(i, j, A.max_level)) {
        matmul(A.S(i, j, A.max_level), x_hat[j], b_hat[i]);
      }
    }
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.U(i, A.max_level), b_hat[i], b_splits[i]);
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j < leaf_nblocks; ++j) {
      if (!A.is_admissible(i, j, A.max_level)) {
        matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
      }
    }
  }

  return b;
}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  // Assign kernel function
  double add_diag = 1e-9;
  kernel = [&](const std::vector<double>& c_row,
               const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_3d_kernel(c_row, c_col, add_diag);
  };

  // Define a 2D grid geometry using the Domain class.
  const int64_t ndim = 1;
  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  // Initialize a Symmetric shared basis matrix container.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / nleaf);
  A.min_level = log2(N / nleaf);

  // Perform a dual tree traversal and initialize the is_admissibile property of the matrix.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);

  A.print_structure();

  // Call a custom construction routine.
  construct_strong_BLR2(A, domain, N, nleaf, rank, admis);

  // Call a custom matvec routine.
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);

  // Low rank matrix-vector product.
  Hatrix::Matrix b_lowrank = matmul(A, x, N, rank);

  // Generate a dense matrix for verification.
  Hatrix::Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
  Hatrix::Matrix b_dense = Hatrix::matmul(A_dense, x);
  Hatrix::Matrix diff = b_dense - b_lowrank;
  double rel_error = Hatrix::norm(diff) / Hatrix::norm(b_dense);

  std::cout << "Error : " << rel_error << std::endl;
}
