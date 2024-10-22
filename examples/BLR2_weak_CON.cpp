#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;

Hatrix::greens_functions::kernel_function_t kernel;
// BLR2 compression scheme using randomization and multiplying each row/col block
// into another to generate the shared bases.

// Accuracy:
// N: 1000 rank: 10 nblocks: 10 admis: 0 construct error: 0.00280282
// N: 2000 rank: 10 nblocks: 20 admis: 0 construct error: 0.00211956
// N: 4000 rank: 10 nblocks: 40 admis: 0 construct error: 0.00172444
// N: 1000 rank: 10 nblocks: 10 admis: 1 construct error: 6.48274e-11
// N: 2000 rank: 10 nblocks: 20 admis: 1 construct error: 1.67429e-10
// N: 4000 rank: 10 nblocks: 40 admis: 1 construct error: 2.84801e-10
// N: 1000 rank: 10 nblocks: 10 admis: 2 construct error: 9.38326e-14
// N: 2000 rank: 10 nblocks: 20 admis: 2 construct error: 4.61331e-13
// N: 4000 rank: 10 nblocks: 40 admis: 2 construct error: 1.39515e-12
// N: 1000 rank: 10 nblocks: 10 admis: 3 construct error: 2.01124e-14
// N: 2000 rank: 10 nblocks: 20 admis: 3 construct error: 1.39202e-13
// N: 4000 rank: 10 nblocks: 40 admis: 3 construct error: 5.52532e-13
constexpr double PV = 1e-3;
using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

void
construct_weak_BLR2(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                    int64_t N, int64_t nleaf, int64_t rank, double admis) {
  int64_t nblocks = N / nleaf;
  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                   i * nleaf, nleaf,
                                                   i * nleaf, nleaf,
                                                   kernel);
    A.D.insert(i, i, A.max_level, std::move(Aij));
  }

  Hatrix::Matrix Utemp, Stemp, Vtemp;
  double error;

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

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (i != j) {
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
    for (int64_t j = 0; j < i; ++j) {
      if (i != j) {
        matmul(A.S(i, j, A.max_level), x_hat[j], b_hat[i]);
        matmul(A.S(i, j, A.max_level), x_hat[i], b_hat[j], true, false, 1, 1);
      }
    }
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.U(i, A.max_level), b_hat[i], b_splits[i]);
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.D(i, i, A.max_level), x_splits[i], b_splits[i]);
  }

  return b;
}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  const double admis = 0;

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  // Assign kernel function
  double add_diag = 1e-9;
  kernel = [&](const std::vector<double>& c_row,
               const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, add_diag);
  };

  // Define a 1D grid geometry using the Domain class.
  const int64_t ndim = 1;
  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  // Initialize a Symmetric shared basis matrix container.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / nleaf);
  A.min_level = log2(N / nleaf);

  // Use a simple distance from diagonal based admissibility condition. admis is kept
  // at 0 since this is a weakly admissible code.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DIAGONAL_ADMIS, admis);

  A.print_structure();

  // Call a custom construction routine.
  construct_weak_BLR2(A, domain, N, nleaf, rank, admis);

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

  return 0;
}
