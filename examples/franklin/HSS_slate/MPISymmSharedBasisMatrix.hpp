#pragma once

#include <functional>

#include "Hatrix/Hatrix.h"
#include "MPIWrapper.hpp"

typedef struct MPISymmSharedBasisMatrix {
  int64_t min_level, max_level;
  Hatrix::ColLevelMap U;
  Hatrix::RowColLevelMap<Hatrix::Matrix> D, S;
  Hatrix::RowColLevelMap<bool> is_admissible;
  Hatrix::RankMap rank_map;     // rank of each level

  int rank_1d(const int row) {
    return row % mpi_world.MPISIZE;
  }

  int rank_2d(const int row, const int col) {
    return (row % mpi_world.MPIGRID[0]) + (col % mpi_world.MPIGRID[1]) * mpi_world.MPIGRID[0];
  }
} MPISymmSharedBasisMatrix;
