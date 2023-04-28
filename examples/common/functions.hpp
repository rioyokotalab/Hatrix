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

#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

namespace Hatrix {

using kernel_func_t =
    std::function<double(const Body& source, const Body& target)>;

// Kernel constants
double PV = 1e-3;
double alpha = 1.0;

void set_kernel_constants(double _PV, double _alpha) {
  PV = _PV;
  alpha = _alpha;
}

kernel_func_t kernel_function;

void set_kernel_function(kernel_func_t _kernel_function) {
  kernel_function = _kernel_function;
}

double p2p_distance(const Body& source, const Body& target) {
  double r = 0;
  for (int64_t axis = 0; axis < 3; axis++) {
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

double matern_kernel(const Body& source, const Body& target) {
  double sigma = 1.0, nu = 0.03, smoothness = 0.5;

  double expr = 0.0;
  double con = 0.0;
  double sigma_square = sigma*sigma;
  double dist = p2p_distance(source, target);

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

Matrix generate_p2p_matrix(const Domain& domain,
                           const std::vector<int64_t>& source,
                           const std::vector<int64_t>& target,
                           const int64_t source_offset = 0,
                           const int64_t target_offset = 0) {
  const int64_t nrows = source.size();
  const int64_t ncols = target.size();
  Matrix out(nrows, ncols);
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++) {
      out(i, j) = kernel_function(domain.bodies[source_offset + source[i]],
                                  domain.bodies[target_offset + target[j]]);
    }
  }
  return out;
}

Matrix generate_p2p_matrix(const Domain& domain,
                           const int64_t row, const int64_t col,
                           const int64_t level) {
  const auto source_idx = domain.get_cell_idx(row, level);
  const auto target_idx = domain.get_cell_idx(col, level);
  const auto& source = domain.cells[source_idx];
  const auto& target = domain.cells[target_idx];

  return generate_p2p_matrix(domain, source.get_bodies(), target.get_bodies());
}

Matrix generate_p2p_matrix(const Domain& domain) {
  return generate_p2p_matrix(domain, 0, 0, 0);
}

// Source: https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c
int64_t adaptive_anchor_grid_size(const Domain& domain,
                                  kernel_func_t kernel_function,
                                  const int64_t leaf_size,
                                  const double theta,
                                  const double ID_tol,
                                  const double stop_tol) {
  double L = 0;
  for (int64_t axis = 0; axis < domain.ndim; axis++) {
    const auto Xmin = Domain::get_Xmin(domain.bodies,
                                       domain.cells[0].get_bodies(), axis);
    const auto Xmax = Domain::get_Xmax(domain.bodies,
                                       domain.cells[0].get_bodies(), axis);
    L = std::max(L, Xmax - Xmin);
  }
  L /= 6.0; // Extra step taken from H2Pack MATLAB reference

  // Create two sets of points in unit boxes
  const auto box_nbodies = 2 * leaf_size;
  const auto shift = L * (theta == 0 ? 1 : theta);
  std::vector<Body> box1, box2;
  std::vector<int64_t> box_idx;
  box_idx.resize(box_nbodies);
  box1.resize(box_nbodies);
  box2.resize(box_nbodies);
  std::mt19937 gen(1234); // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(0, 1);
  for (int64_t i = 0; i < box_nbodies; i++) {
    box_idx[i] = i;
    for (int64_t axis = 0; axis < domain.ndim; axis++) {
      box1[i].X[axis] = L * dist(gen);
      box2[i].X[axis] = L * dist(gen) + shift;
    }
  }

  auto generate_matrix = [kernel_function]
                         (const std::vector<Body>& source,
                          const std::vector<Body>& target,
                          const std::vector<int64_t>& source_idx,
                          const std::vector<int64_t>& target_idx) {
    Matrix out(source_idx.size(), target_idx.size());
    for (int64_t i = 0; i < out.rows; i++) {
      for (int64_t j = 0; j < out.cols; j++) {
        out(i, j) = kernel_function(source[source_idx[i]],
                                    target[target_idx[j]]);
      }
    }
    return out;
  };
  Matrix A = generate_matrix(box1, box2, box_idx, box_idx);
  // Find anchor grid size r by checking approximation error to A
  double box2_xmin[MAX_NDIM], box2_xmax[MAX_NDIM];
  for (int64_t i = 0; i < box_nbodies; i++) {
    for (int64_t axis = 0; axis < domain.ndim; axis++) {
      if (i == 0) {
        box2_xmin[axis] = box2[i].X[axis];
        box2_xmax[axis] = box2[i].X[axis];
      }
      else {
        box2_xmin[axis] = std::min(box2_xmin[axis], box2[i].X[axis]);
        box2_xmax[axis] = std::max(box2_xmax[axis], box2[i].X[axis]);
      }
    }
  }
  double box2_size[MAX_NDIM];
  for (int64_t axis = 0; axis < domain.ndim; axis++) {
    box2_size[axis] = box2_xmax[axis] - box2_xmin[axis];
  }
  int64_t r = 1;
  bool stop = false;
  while (!stop) {
    const auto box2_sample =
        Domain::select_sample_bodies(domain.ndim, box2, box_idx, r, 3, 0);
    // A1 = kernel(box1, box2_sample)
    Matrix A1 = generate_matrix(box1, box2, box_idx, box2_sample);
    Matrix U;
    std::vector<int64_t> ipiv_rows;
    std::tie(U, ipiv_rows) = error_id_row(A1, ID_tol, false);
    int64_t rank = U.cols;
    // A2 = A(ipiv_rows[:rank], :)
    Matrix A2(rank, A.cols);
    for (int64_t i = 0; i < rank; i++) {
      for (int64_t j = 0; j < A.cols; j++) {
        A2(i, j) = A(ipiv_rows[i], j);
      }
    }
    Matrix UxA2 = matmul(U, A2);
    const double error = norm(A - UxA2);
    if ((error < stop_tol) ||
        ((box_nbodies - box2_sample.size()) < (box_nbodies / 10))) {
      stop = true;
    }
    else {
      r++;
    }
  }
  return r;
}

Matrix prepend_complement_basis(const Matrix &Q) {
  Matrix Q_F(Q.rows, Q.rows);
  Matrix Q_full, R;
  std::tie(Q_full, R) =
      qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

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
