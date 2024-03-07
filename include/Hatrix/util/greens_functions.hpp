#pragma once

#include <vector>
#include <functional>

namespace Hatrix {
  namespace greens_functions {
    // Common type for kernel functions that can be used with lambda for currying the
    // constants of the Green's function.
    using kernel_function_t =
      std::function<double(const std::vector<double>& coords_row,
                           const std::vector<double>& coords_col)>;

    double laplace_1d_kernel(const std::vector<double>& coords_row,
                             const std::vector<double>& coords_col,
                             const double eta);

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
