#pragma once

typedef struct MPISymmSharedBasisMatrix {
  int64_t min_level, max_level;
  Hatrix::ColLevelMap U;
  Hatrix::RowColLevelMap<Hatrix::Matrix> D, S;
  Hatrix::RowColLevelMap<bool> is_admissible;
  Hatrix::RankMap rank_map;
} MPISymmSharedBasisMatrix;
