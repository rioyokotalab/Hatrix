// Performance comparison of SVD vs. one sided ID. Note that the interpolation
// matrix of the ID is not orthgonalized.
#include <cstdlib>
#include <iostream>
#include <chrono>

#include "Hatrix/Hatrix.h"

// Results on a 2021 macbook pro with M1.
// N= 1000 rank= 10 svd= 2421 id= 497
// N= 1000 rank= 20 svd= 2422 id= 531
// N= 1000 rank= 40 svd= 2420 id= 499
// N= 1000 rank= 50 svd= 2421 id= 499
// N= 1000 rank= 100 svd= 2424 id= 500
// N= 2000 rank= 10 svd= 23111 id= 4101
// N= 2000 rank= 20 svd= 23172 id= 4102
// N= 2000 rank= 40 svd= 23109 id= 4103
// N= 2000 rank= 50 svd= 23118 id= 4102
// N= 2000 rank= 100 svd= 23115 id= 4109
// N= 3000 rank= 10 svd= 83671 id= 13970
// N= 3000 rank= 20 svd= 84293 id= 13991
// N= 3000 rank= 40 svd= 84303 id= 14010
// N= 3000 rank= 50 svd= 84455 id= 14015
// N= 3000 rank= 100 svd= 84505 id= 14027
// N= 4000 rank= 10 svd= 523282 id= 80411
// N= 4000 rank= 20 svd= 294784 id= 33707
// N= 4000 rank= 40 svd= 276772 id= 33750
// N= 4000 rank= 50 svd= 274316 id= 33666

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
