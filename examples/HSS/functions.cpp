#include <vector>

#include "Domain.hpp"
#include "functions.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  std::vector<int64_t>
  leaf_indices(int64_t node, int64_t level, int64_t height) {
    std::vector<int64_t> indices;
    if (level == height) {
      std::vector<int64_t> leaf_index{node};
      return leaf_index;
    }

    auto c1_indices = leaf_indices(node * 2, level + 1, height);
    auto c2_indices = leaf_indices(node * 2 + 1, level + 1, height);

    c1_indices.insert(c1_indices.end(), c2_indices.begin(), c2_indices.end());

    return c1_indices;
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 int64_t level, int64_t height,
                                 const kernel_function& kernel,
                                 Matrix& out) {
    if (level == height) {
      generate_p2p_interactions(domain, irow, icol, kernel, out);
    }

    std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
    std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

    int64_t nrows = 0, ncols = 0;
    for (int64_t i = 0; i < leaf_rows.size(); ++i) {
      nrows += domain.boxes[leaf_rows[i]].num_particles; }
    for (int64_t i = 0; i < leaf_cols.size(); ++i) {
      ncols += domain.boxes[leaf_cols[i]].num_particles; }

    assert(out.rows == nrows);
    assert(out.cols == ncols);

    std::vector<Particle> source_particles, target_particles;
    for (int64_t i = 0; i < leaf_rows.size(); ++i) {
      int64_t source_box = leaf_rows[i];
      int64_t source = domain.boxes[source_box].start_index;
      for (int64_t n = 0; n < domain.boxes[source_box].num_particles; ++n) {
        source_particles.push_back(domain.particles[source + n]);
      }
    }

    for (int64_t i = 0; i < leaf_cols.size(); ++i) {
      int64_t target_box = leaf_cols[i];
      int64_t target = domain.boxes[target_box].start_index;
      for (int64_t n = 0; n < domain.boxes[target_box].num_particles; ++n) {
        target_particles.push_back(domain.particles[target + n]);
      }
    }

    for (int64_t i = 0; i < nrows; ++i) {
      for (int64_t j = 0; j < ncols; ++j) {
        out(i, j) = kernel(source_particles[i].coords,
                           target_particles[j].coords);
      }
    }
  }

  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   int64_t level, int64_t height,
                                   const kernel_function& kernel) {
    if (level == height) {
      return generate_p2p_interactions(domain, irow, icol, kernel);
    }

    std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
    std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

    int64_t nrows = 0, ncols = 0;
    for (int64_t i = 0; i < leaf_rows.size(); ++i) {
      nrows += domain.boxes[leaf_rows[i]].num_particles; }
    for (int64_t i = 0; i < leaf_cols.size(); ++i) {
      ncols += domain.boxes[leaf_cols[i]].num_particles; }

    Matrix out(nrows, ncols);
    generate_p2p_interactions(domain, irow, icol, level, height,
                              kernel, out);

    return out;
  }

  Matrix generate_p2p_matrix(const Domain& domain,
                             const kernel_function& kernel) {
    int64_t rows =  domain.particles.size();
    int64_t cols =  domain.particles.size();
    Matrix out(rows, cols);

    std::vector<Particle> particles;

    for (int64_t irow = 0; irow < domain.boxes.size(); ++irow) {
      int64_t source = domain.boxes[irow].start_index;
      for (int64_t n = 0; n < domain.boxes[irow].num_particles; ++n) {
        particles.push_back(domain.particles[source + n]);
      }
    }


    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        out(i, j) = kernel(particles[i].coords,
                           particles[j].coords);
      }
    }

    return out;
  }

  // Generates p2p interactions between the particles of two boxes specified by irow
  // and icol. ndim specifies the dimensionality of the particles present in domain.
  // Uses a laplace kernel for generating the interaction.
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   const kernel_function& kernel) {
    Matrix out(domain.boxes[irow].num_particles, domain.boxes[icol].num_particles);
    generate_p2p_interactions(domain, irow, icol, kernel, out);
    return out;
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 const kernel_function& kernel,
                                 Matrix& out) {
    assert(out.rows == domain.boxes[irow].num_particles);
    assert(out.cols == domain.boxes[icol].num_particles);
    for (int64_t i = 0; i < domain.boxes[irow].num_particles; ++i) {
      for (int64_t j = 0; j < domain.boxes[icol].num_particles; ++j) {
        int64_t source = domain.boxes[irow].start_index;
        int64_t target = domain.boxes[icol].start_index;

        out(i, j) = kernel(domain.particles[source+i].coords,
                           domain.particles[target+j].coords);
      }
    }
  }
}
