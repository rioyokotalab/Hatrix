#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <string>

#include "Hatrix/Hatrix.hpp"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.

void
construct_strong_BLR2(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                      int64_t N, int64_t nleaf, int64_t rank, double admis) {

}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);

  if (N % nleaf == 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  // Define a 2D grid geometry using the Domain class.
  const int64_t ndim = 2;
  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  // Initialize a Symmetric shared basis matrix container.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / nleaf);

  // Perform a dual tree traversal and initialize the is_admissibile property of the matrix.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);

  // Call a custom construction routine.
  construct_strong_BLR2(A, domain, N, nleaf, rank, admis);


}
