#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "Body.hpp"
#include "Cell.hpp"
#include "Domain.hpp"
#include "functions.hpp"

namespace {

// Taken from: H2Pack GitHub
// Decompose an integer into the sum of multiple integers which are approximately proportional
// to another floating point array.
// Input parameters:
//   nelem      : Number of integers after decomposition
//   decomp_sum : The target number to be decomposed
//   prop       : Floating point array as the proportions
// Output parameter:
//   decomp : Decomposed values, decomp[i] ~= prop[i] / sum(prop[0:nelem-1]) * decomp_sum
// Return value: product(decomp[0:nelem-1])
int64_t proportional_int_decompose(const int64_t nelem,
                                   const int64_t decomp_sum,
                                   const double* prop,
                                   int64_t* decomp) {
  double sum_prop = 0.0;
  for (int64_t i = 0; i < nelem; i++) {
    sum_prop += prop[i];
  }
  std::vector<double> decomp_prop(nelem, 0);
  int decomp_sum0 = 0;
  for (int64_t i = 0; i < nelem; i++) {
    decomp_prop[i] = (double)decomp_sum * prop[i] / sum_prop;
    decomp[i] = (int64_t)std::floor(decomp_prop[i]);
    decomp_sum0 += decomp[i];
  }
  for (int64_t k = decomp_sum0; k < decomp_sum; k++) {
    // Add 1 to the position that got hit most by floor
    int64_t min_idx = 0;
    double max_diff = decomp_prop[0] - (double)decomp[0];
    for (int64_t i = 1; i < nelem; i++) {
      const auto diff = decomp_prop[i] - (double)decomp[i];
      if (diff > max_diff) {
        max_diff = diff;
        min_idx = i;
      }
    }
    decomp[min_idx]++;
  }
  int64_t prod = 1;
  for (int64_t i = 0; i < nelem; i++) prod *= (decomp[i] + 1);
  return prod;
}

// Source: https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c
std::vector<Hatrix::Body> build_anchor_grid(const int64_t ndim,
                                            const double coord_min[/* ndim */],
                                            const double coord_max[/* ndim */],
                                            const double enbox_size[/* ndim */],
                                            const int64_t grid_size[/* ndim */],
                                            const int64_t grid_algo = 0) {
  int64_t max_grid_size = 0;
  int64_t anchor_npt = 1;
  for (int64_t i = 0; i < ndim; i++) {
    max_grid_size =
        (grid_size[i] > max_grid_size) ? grid_size[i] : max_grid_size;
    anchor_npt *= (grid_size[i] + 1);
  }
  max_grid_size++;

  // 1. Assign anchor points in each dimension
  std::vector<double> anchor_dim(ndim * max_grid_size);
  for (int64_t i = 0; i < ndim; i++) {
    if (grid_size[i] == 0) {
      const auto offset_i = i * max_grid_size;
      anchor_dim[offset_i] = (coord_min[i] + coord_max[i]) * 0.5;
    }
  }
  if (grid_algo == 0) {  // Default: grid_algo == 6 in H2Pack code
    const double c0 = 1.0;
    const double c1 = 0.5;
    const double c2 = 0.25;
    for (int64_t i = 0; i < ndim; i++) {
      const auto size_i = c0 * enbox_size[i] / ((double)grid_size[i] + c1);
      const auto offset_i = i * max_grid_size;
      for (int64_t j = 0; j <= grid_size[i]; j++) {
        anchor_dim[offset_i + j] =
            coord_min[i] + c2 * size_i + size_i * (double)j;
      }
    }
  }
  else {  // Chebyshev anchor points: grid_algo == 2 in H2Pack code
    for (int64_t i = 0; i < ndim; i++) {
      if (grid_size[i] == 0) continue;
      const auto offset_i = i * max_grid_size;
      const auto s0 = 0.5 * (coord_max[i] + coord_min[i]);
      const auto s1 = 0.5 * (coord_max[i] - coord_min[i]);
      const auto s2 = M_PI / (2.0 * (double)grid_size[i] + 2);
      for (int64_t j = 0; j <= grid_size[i]; j++) {
        const double v0 = 2.0 * (double)j + 1.0;
        const double v1 = std::cos(v0 * s2);
        anchor_dim[offset_i + j] = s0 + s1 * v1;
      }
    }
  }
  // 2. Do a tensor product to get all anchor points
  std::vector<Hatrix::Body> anchor_coord(anchor_npt);
  std::vector<int64_t> stride(ndim + 1);
  stride[0] = 1;
  for (int64_t i = 0; i < ndim; i++) {
    stride[i + 1] = stride[i] * (grid_size[i] + 1);
  }
  for (int64_t i = 0; i < anchor_npt; i++) {
    for (int64_t j = 0; j < ndim; j++) {
      const int64_t dim_idx = (i / stride[j]) % (grid_size[j] + 1);
      anchor_coord[i].X[j] = anchor_dim[j * max_grid_size + dim_idx];
    }
  }
  return anchor_coord;
}

}  // namespace

namespace Hatrix {
namespace Utility {

std::vector<int64_t> select_sample_points(const int64_t ndim,
                                          const std::vector<Hatrix::Body>& points_arr,
                                          const std::vector<int64_t>& points_idx,
                                          const int64_t sample_size,
                                          const int64_t sampling_algo,
                                          const int64_t grid_algo = 0) {
  static std::mt19937 g(1234);  // Use fixed seed for reproducibility
  const int64_t npoints = points_idx.size();
  if (sample_size >= npoints) {
    return points_idx;
  }
  std::vector<int64_t> sample_idx;
  switch (sampling_algo) {
    case 0: {  // Select based on equally spaced indices
      const int64_t d =
          (int64_t)std::floor((double)npoints / (double)sample_size);
      int64_t k = 0;
      sample_idx.reserve(sample_size);
      for (int64_t i = 0; i < sample_size; i++) {
        sample_idx.push_back(points_idx[k]);
        k += d;
      }
      break;
    }
    case 1: {  // Random sampling
      std::vector<int64_t> random_indices(npoints, 0);
      for (int64_t i = 0; i < npoints; i++) {
        random_indices[i] = i;
      }
      // Random shuffle 3 times
      for (int64_t i = 0; i < 3; i++) {
        std::shuffle(random_indices.begin(), random_indices.end(), g);
      }
      // Choose sample based on random indices
      sample_idx.reserve(sample_size);
      for (int64_t i = 0; i < sample_size; i++) {
        const auto ridx = random_indices[i];
        sample_idx.push_back(points_idx[ridx]);
      }
      break;
    }
    case 2: {  // Farthest Point Sampling (FPS)
      std::vector<bool> chosen(npoints, false);
      // Find centroid of the considered points
      std::vector<double> center(ndim);
      for (int64_t axis = 0; axis < ndim; axis++) {
        const auto Xsum = Domain::get_Xsum(points_arr, points_idx, axis);
        center[axis] = Xsum / (double)points_idx.size();
      }
      // Start with point closest to the centroid as pivot
      int64_t pivot = -1;
      double min_dist = std::numeric_limits<double>::max();
      for (int64_t i = 0; i < npoints; i++) {
        const auto dist2_i = Domain::dist2(ndim,
                                           center.data(),
                                           points_arr[points_idx[i]].X);
        if (dist2_i < min_dist) {
          min_dist = dist2_i;
          pivot = i;
        }
      }
      chosen[pivot] = true;
      for (int64_t k = 1; k < sample_size; k++) {
        // Add the farthest body from pivot into sample
        double max_dist = -1.;
        int64_t farthest_idx = -1;
        for (int64_t i = 0; i < npoints; i++) {
          if(!chosen[i]) {
            const auto dist2_i = Domain::dist2(ndim,
                                               points_arr[points_idx[pivot]].X,
                                               points_arr[points_idx[i]].X);
            if (dist2_i > max_dist) {
              max_dist = dist2_i;
              farthest_idx = i;
            }
          }
        }
        chosen[farthest_idx] = true;
        pivot = farthest_idx;
      }
      sample_idx.reserve(sample_size);
      for (int64_t i = 0; i < npoints; i++) {
        if (chosen[i]) {
          sample_idx.push_back(points_idx[i]);
        }
      }
      break;
    }
    case 3: {  // Anchor Net Method
      std::vector<double> coord_max(ndim), coord_min(ndim), diameter(ndim);
      for (int64_t axis = 0; axis < ndim; axis++) {
        coord_min[axis] = Domain::get_Xmin(points_arr, points_idx, axis);
        coord_max[axis] = Domain::get_Xmax(points_arr, points_idx, axis);
        diameter[axis] = coord_max[axis] - coord_min[axis];
      }
      std::vector<int64_t> grid_size(ndim);
      const auto anchor_npt = proportional_int_decompose(ndim, sample_size,
                                                         diameter.data(), grid_size.data());
      if (anchor_npt >= npoints) {
        sample_idx = points_idx;
      }
      else {
        const auto anchor_points =
            build_anchor_grid(ndim, coord_min.data(), coord_max.data(),
                              diameter.data(), grid_size.data(), grid_algo);
        std::vector<bool> chosen(npoints, false);
        for (int64_t i = 0; i < anchor_npt; i++) {
          int64_t min_idx = -1;
          double min_dist = std::numeric_limits<double>::max();
          for (int64_t j = 0; j < npoints; j++) {
            const auto dist2_ij = Domain::dist2(ndim,
                                                anchor_points[i].X,
                                                points_arr[points_idx[j]].X);
            if (dist2_ij < min_dist) {
              min_dist = dist2_ij;
              min_idx = j;
            }
          }
          chosen[min_idx] = true;
        }
        sample_idx.reserve(anchor_npt);
        for (int64_t i = 0; i < npoints; i++) {
          if (chosen[i]) {
            sample_idx.push_back(points_idx[i]);
          }
        }
      }
      break;
    }
    default: { // No sampling
      sample_idx = points_idx;
      break;
    }
  }
  return sample_idx;
}

}  // namespace Utility
}  // namespace Hatrix
