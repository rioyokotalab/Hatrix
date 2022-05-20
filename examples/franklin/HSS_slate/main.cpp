#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iomanip>


#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "mpi.h"
#include <slate/slate.hh>


typedef struct MPISymmSharedBasisMatrix {
  int64_t min_level, max_level;
  Hatrix::ColLevelMap U;
  Hatrix::RowColLevelMap<Hatrix::Matrix> D, S;
  Hatrix::RowColLevelMap<bool> is_admissible;
  Hatrix::RankMap rank_map;
} MPISymmSharedBasisMatrix;

int main(int argc, char* argv[]) {
  Hatrix::Args opts(argc, argv);

  int provided;
  assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == 0);

  // generate the points on all the processes.
  auto start_domain = std::chrono::system_clock::now();
  Hatrix::Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == Hatrix::GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == Hatrix::CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  domain.divide_domain_and_create_particle_boxes(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  double construct_time;

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();

    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();
  }

  assert(MPI_Finalize() == 0);

  return 0;
}
