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

namespace Hatrix {
  class BLR2 {
  private:
    RowMap<Matrix> U;
    ColMap<Matrix> V;
    RowColMap<bool> is_admissible;
    RowColMap<Matrix> D, S;
    int64_t N, nblocks, rank, admis;

  public:
    BLR2(const randvec_t& randpts, int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::abs(i - j) > admis);

          if (!is_admissible(i, j)) {
            D.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts,
                                                       block_size, block_size,
                                                       i*block_size, j*block_size, PV));
          }
        }
      }

      int64_t oversampling = 5;
      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      // Generate a bunch of random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(block_size, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Hatrix::Matrix AY(block_size, rank + oversampling);
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size, PV);
            Hatrix::matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, std::move(Utemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size, PV);
            Hatrix::matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, std::move(transpose(Vtemp)));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size, PV);
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U[i], dense, true), V[j]));
          }
        }
      }
    }

    double construction_error(const randvec_t& randpts) {
      double error = 0, fnorm = 0;
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(i, j)) {
            double dense_error = Hatrix::norm(D(i, j) -
                                           Hatrix::generate_laplacend_matrix(randpts,
                                                                             block_size,
                                                                             block_size,
                                                                             block_size * i,
                                                                             block_size * j,
                                                                             PV));
            error += pow(dense_error, 2);
          }
          else {
            Matrix& Ubig = U(i);
            Matrix& Vbig = V(j);
            Matrix expected = matmul(matmul(Ubig, S(i, j)), Vbig, false, true);
            Matrix actual = Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                              i * block_size, j * block_size, PV);
            // std::cout << "rel err: " << Hatrix::norm(actual - expected) / Hatrix::norm(expected) << std::endl;

            error += pow(Hatrix::norm(expected - actual), 2);
          }
        }
      }
      return std::sqrt(error / N / N);
    }
  };

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


  // std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
  //           << " construct error: " << construct_error << "\n";

  A.print_structure();

  // Call a custom construction routine.
  construct_weak_BLR2(A, domain, N, nleaf, rank, admis);

  return 0;
}
