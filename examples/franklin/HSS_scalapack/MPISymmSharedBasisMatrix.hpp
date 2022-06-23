#pragma once

#include <functional>

#include "Hatrix/Hatrix.h"
#include "MPIWrapper.hpp"

typedef struct MPISymmSharedBasisMatrix {
  int64_t min_level, max_level;
  Hatrix::ColLevelMap U;
  Hatrix::RowColLevelMap<Hatrix::Matrix> D, S;
  Hatrix::RowColLevelMap<bool> is_admissible;
  Hatrix::RowMap<std::vector<int64_t>> rank_map;     // rank of each level

  int rank_1d(const int row) const {
    return row % mpi_world.MPISIZE;
  }

  int rank_2d(const int row, const int col) const {
    return (row % mpi_world.MPIGRID[0]) +
      (col % mpi_world.MPIGRID[1]) * mpi_world.MPIGRID[0];
  }
} MPISymmSharedBasisMatrix;
