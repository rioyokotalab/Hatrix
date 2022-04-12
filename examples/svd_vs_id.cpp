// Performance comparison of SVD vs. one sided ID. Note that the interpolation
// matrix of the ID is not orthgonalized.
#include <cstdlib>
#include <iostream>
#include <chrono>

#include "Hatrix/Hatrix.h"

using namespace Hatrix;

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);

  Matrix A = Hatrix::generate_random_matrix(N, N);
  Matrix U, S, V; double error;

  auto svd_start = std::chrono::system_clock::now();
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A, rank);
  auto svd_stop = std::chrono::system_clock::now();
  double svd_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(svd_stop - svd_start).count();

  Matrix Arank = matmul(matmul(U, S), V);

  Matrix interp, pivots;
  auto id_start = std::chrono::system_clock::now();
  std::tie(interp, pivots) = Hatrix::truncated_interpolate(Arank, false, rank);
  auto id_stop = std::chrono::system_clock::now();
  double id_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(id_stop - id_start).count();

  std::cout << "N= " << N << " rank= " << rank << " svd= "
            << svd_time << " id= " << id_time << std::endl;
}
