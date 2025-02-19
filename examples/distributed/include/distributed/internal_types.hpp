#pragma once

#include "Hatrix/Hatrix.hpp"

#include <functional>

namespace Hatrix {
  constexpr int INIT_VALUE = -1;
  constexpr int oversampling = 5;
  enum KERNEL_FUNC {LAPLACE, SQR_EXP, SINE, GSL_MATERN, YUKAWA, ELSES_C60};
  enum KIND_OF_GEOMETRY {GRID, CIRCULAR, COL_FILE, ELSES_C60_GEOMETRY};
  enum ADMIS_KIND {DIAGONAL, GEOMETRY};
  enum CONSTRUCT_ALGORITHM {MIRO, MIRO_FASTER, ID_RANDOM};

  using kernel_function =
    std::function<double(const std::vector<double>& coords_row,
                         const std::vector<double>& coords_col)>;
}
