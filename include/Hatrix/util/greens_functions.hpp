#pragma once

#include "Hatrix/Hatrix.hpp"

namespace Hatrix {
  namespace greens_functions {
    double laplace_2d_kernel(const std::vector<double>& coords_row,
                          const std::vector<double>& coords_col,
                          const double eta);

    double laplace_3d_kernel(const std::vector<double>& coords_row,
                          const std::vector<double>& coords_col,
                          const double eta);

    double matern_kernel(const std::vector<double>& coords_row,
                         const std::vector<double>& coords_col,
                         const double sigma, const double nu, const double smoothness);

    double yukawa_kernel(const std::vector<double>& coords_row,
                  const std::vector<double>& coords_col,
                  const double alpha, double singularity);
  }
}
