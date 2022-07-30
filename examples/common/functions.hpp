#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Particle.hpp"
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
    std::function<double(const std::vector<double>& coords_row,
                         const std::vector<double>& coords_col)>;

kernel_func_t kernel_function;

void set_kernel_function(kernel_func_t _kernel_function) {
  kernel_function = _kernel_function;
}

double p2p_distance(const std::vector<double>& coords_row,
                    const std::vector<double>& coords_col) {
  double r = 0;
  const int64_t ndim = coords_row.size();
  for (int64_t k = 0; k < ndim; ++k) {
    r += (coords_row[k] - coords_col[k]) *
         (coords_row[k] - coords_col[k]);
  }
  return std::sqrt(r);
}

double laplace_kernel(const std::vector<double>& coords_row,
                      const std::vector<double>& coords_col) {
  const double r = p2p_distance(coords_row, coords_col) + PV;
  const double out = 1. / r;
  return out;
}

double yukawa_kernel(const std::vector<double>& coords_row,
                     const std::vector<double>& coords_col) {
  const double r = p2p_distance(coords_row, coords_col) + PV;
  const double out = std::exp(alpha * -r) / r;
  return out;
}

std::vector<int64_t> leaf_indices(const int64_t node, const int64_t level,
                                  const int64_t height) {
  std::vector<int64_t> indices;
  if (level == height) {
    indices.push_back(node);
  }
  else {
    auto c1_indices = leaf_indices(node * 2 + 0, level + 1, height);
    auto c2_indices = leaf_indices(node * 2 + 1, level + 1, height);
    indices.insert(indices.end(), c1_indices.begin(), c1_indices.end());
    indices.insert(indices.end(), c2_indices.begin(), c2_indices.end());
  }

  return indices;
}

Matrix generate_p2p_interactions(const Domain& domain,
                                 const int64_t row, const int64_t col,
                                 const int64_t level, const int64_t height) {
  // Get source and target particles by gathering leaf level boxes
  const auto source_leaf_indices = leaf_indices(row, level, height);
  const auto target_leaf_indices = leaf_indices(col, level, height);
  std::vector<Particle> source_particles, target_particles;
  for (int64_t i = 0; i < source_leaf_indices.size(); i++) {
    const auto source_box_idx = source_leaf_indices[i];
    const auto offset = domain.boxes[source_box_idx].begin_index;
    for (int64_t k = 0; k < domain.boxes[source_box_idx].num_particles; k++) {
      source_particles.push_back(domain.particles[offset + k]);
    }
  }
  for (int64_t i = 0; i < target_leaf_indices.size(); i++) {
    const auto target_box_idx = target_leaf_indices[i];
    const auto offset = domain.boxes[target_box_idx].begin_index;
    for (int64_t k = 0; k < domain.boxes[target_box_idx].num_particles; k++) {
      target_particles.push_back(domain.particles[offset + k]);
    }
  }

  // Prepare output matrix
  int64_t nrows = 0, ncols = 0;
  for (int64_t i = 0; i < source_leaf_indices.size(); i++) {
    const auto source_box_idx = source_leaf_indices[i];
    nrows += domain.boxes[source_box_idx].num_particles;
  }
  for (int64_t i = 0; i < target_leaf_indices.size(); i++) {
    const auto target_box_idx = target_leaf_indices[i];
    ncols += domain.boxes[target_box_idx].num_particles;
  }
  Matrix out(nrows, ncols);
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++) {
      out(i, j) = kernel_function(source_particles[i].coords,
                                  target_particles[j].coords);
    }
  }
  return out;
}

Matrix generate_p2p_matrix(const Domain& domain) {
  // Gather all particles
  std::vector<Particle> particles;
  for (int64_t i = 0; i < domain.boxes.size(); i++) {
    const auto box_offset = domain.boxes[i].begin_index;
    for (int64_t k = 0; k < domain.boxes[i].num_particles; k++) {
      particles.push_back(domain.particles[box_offset + k]);
    }
  }

  // Prepare output matrix
  const int64_t nrows =  domain.particles.size();
  const int64_t ncols =  domain.particles.size();
  Matrix out(nrows, ncols);
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++) {
      out(i, j) = kernel_function(particles[i].coords,
                                  particles[j].coords);
    }
  }
  return out;
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

