#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Body.hpp"
#include "Cell.hpp"
#include "Domain.hpp"
#include "Hatrix/Hatrix.h"

namespace Hatrix {

// Kernel constants
double PV = 1e-3;
double alpha = 1.0;

void set_kernel_constants(double _PV, double _alpha) {
  PV = _PV;
  alpha = _alpha;
}

using kernel_func_t =
    std::function<double(const Body& source, const Body& target)>;

kernel_func_t kernel_function;

void set_kernel_function(kernel_func_t _kernel_function) {
  kernel_function = _kernel_function;
}

double p2p_distance(const Body& source, const Body& target) {
  double r = 0;
  for (uint64_t axis = 0; axis < 3; axis++) {
    r += (source.X[axis] - target.X[axis]) *
         (source.X[axis] - target.X[axis]);
  }
  return std::sqrt(r);
}

double laplace_kernel(const Body& source, const Body& target) {
  const double r = p2p_distance(source, target) + PV;
  const double out = 1. / r;
  return out;
}

double yukawa_kernel(const Body& source, const Body& target) {
  const double r = p2p_distance(source, target) + PV;
  const double out = std::exp(alpha * -r) / r;
  return out;
}

Matrix generate_p2p_matrix(const Domain& domain,
                           const int64_t row, const int64_t col, const int64_t level) {
  const auto source_loc = domain.get_cell_loc(row, level);
  const auto target_loc = domain.get_cell_loc(col, level);
  const auto& source = domain.cells[source_loc];
  const auto& target = domain.cells[target_loc];

  // Prepare output matrix
  int64_t nrows = source.nbodies;
  int64_t ncols = target.nbodies;
  Matrix out(nrows, ncols);
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++) {
      out(i, j) = kernel_function(domain.bodies[source.body + i],
                                  domain.bodies[target.body + j]);
    }
  }
  return out;
}

Matrix generate_p2p_matrix(const Domain& domain) {
  return generate_p2p_matrix(domain, 0, 0, 0);
}

Matrix prepend_complement_basis(const Matrix &Q) {
  Matrix Q_F(Q.rows, Q.rows);
  Matrix Q_full, R;
  std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

  for (int64_t i = 0; i < Q_F.rows; i++) {
    for (int64_t j = 0; j < Q_F.cols - Q.cols; j++) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }
  for (int64_t i = 0; i < Q_F.rows; i++) {
    for (int64_t j = 0; j < Q.cols; j++) {
      Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
    }
  }
  return Q_F;
}

} // namespace Hatrix

