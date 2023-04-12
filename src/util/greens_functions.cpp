#include <cmath>
#include <cstdint>
#include <random>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

#include "Hatrix/util/greens_functions.hpp"

namespace Hatrix {
  static double distance(const std::vector<double>& coords_row,
                         const std::vector<double>& coords_col) {
    int64_t ndim = coords_row.size();
    double rij = 0;
    for (int64_t k = 0; k < ndim; ++k) {
      rij += pow(coords_row[k] - coords_col[k], 2);
    }

    return std::sqrt(rij);
  }

  double laplace_kernel(const std::vector<double>& coords_row,
                        const std::vector<double>& coords_col,
                        const double eta) {
    double dist = distance(coords_row, coords_col);
    double out = 1 / (dist + eta);
    return out;
  }

  double matern_kernel(const std::vector<double>& coords_row,
                       const std::vector<double>& coords_col,
                       const double sigma, const double nu, const double smoothness) {
    double expr = 0.0;
    double con = 0.0;
    double sigma_square = sigma*sigma;
    double dist = distance(coords_row, coords_col);

    con = pow(2, (smoothness - 1)) * gsl_sf_gamma(smoothness);
    con = 1.0 / con;
    con = sigma_square * con;

    if (dist != 0) {
      expr = dist / nu;
      return con * pow(expr, smoothness) * gsl_sf_bessel_Knu(smoothness, expr);
    }
    else {
      return sigma_square;
    }
  }

  double yukawa(const std::vector<double>& coords_row,
                const std::vector<double>& coords_col,
                const double alpha, const double singularity) {
    double dist = distance(coords_row, coords_col);
    double r = dist + singularity;
    return exp(alpha * -r) / r;
  }
}
