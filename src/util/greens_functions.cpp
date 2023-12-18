#include <cmath>
#include <cassert>
#include <cstdint>
#include <random>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

#include "Hatrix/util/greens_functions.hpp"

namespace Hatrix {
  namespace greens_functions {
    static double distance(const std::vector<double>& coords_row,
                           const std::vector<double>& coords_col) {
      int64_t ndim = coords_row.size();
      double rij = 0;
      for (int64_t k = 0; k < ndim; ++k) {
        rij += pow(coords_row[k] - coords_col[k], 2);
      }

      return std::sqrt(rij);
    }

    double laplace_1d_kernel(const std::vector<double>& coords_row,
                             const std::vector<double>& coords_col,
                             const double eta) {
      assert(coords_row.size() == coords_col.size());
      const double dist = distance(coords_row, coords_col);
      return (dist + eta) / 2;
    }

    double laplace_2d_kernel(const std::vector<double>& coords_row,
                          const std::vector<double>& coords_col,
                          const double eta) {
      assert(coords_row.size() == coords_col.size());
      double dist = distance(coords_row, coords_col);
      return -log(dist + eta);
    }

    double laplace_3d_kernel(const std::vector<double>& coords_row,
                          const std::vector<double>& coords_col,
                          const double eta) {
      assert(coords_row.size() == coords_col.size());
      double dist = distance(coords_row, coords_col);
      double out = 1 / (dist + eta);
      return out;
    }

    double matern_kernel(const std::vector<double>& coords_row,
                         const std::vector<double>& coords_col,
                         const double sigma, const double nu, const double smoothness) {
      assert(coords_row.size() == coords_col.size());
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

    double yukawa_kernel(const std::vector<double>& coords_row,
                  const std::vector<double>& coords_col,
                  const double alpha, const double singularity) {
      assert(coords_row.size() == coords_col.size());
      double dist = distance(coords_row, coords_col);
      double r = dist + singularity;
      return exp(alpha * -r) / r;
    }
  }
}
