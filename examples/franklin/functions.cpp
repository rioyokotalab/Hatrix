#include <vector>
#include <cassert>
#include <cmath>

#include "franklin/franklin.hpp"

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

  void
  search_tree_for_nodes(const Cell& tree, const int64_t level_index, const int64_t level,
                        int64_t &pstart, int64_t &pend) {
    if (tree.level == level && tree.level_index == level_index) {
      pstart = tree.start_index;
      pend = tree.end_index;
      return;
    }

    if (tree.cells.size() == 0) {
      throw std::exception();
    }

    search_tree_for_nodes(tree.cells[0], level_index, level, pstart, pend);
    search_tree_for_nodes(tree.cells[1], level_index, level, pstart, pend);
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 int64_t level, int64_t height,
                                 const kernel_function& kernel,
                                 Matrix& out) {
    int64_t source_pstart, source_pend, target_pstart, target_pend;

    search_tree_for_nodes(domain.tree, irow, level, source_pstart, source_pend);
    search_tree_for_nodes(domain.tree, icol, level, target_pstart, target_pend);


    std::cout << "s pstart: " << source_pstart << " s pstop: " << source_pend << std::endl;
    // if (level == height) {
    //   generate_p2p_interactions(domain, irow, icol, kernel, out);
    // }

    // std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
    // std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

    // int64_t nrows = 0, ncols = 0;
    // for (unsigned i = 0; i < leaf_rows.size(); ++i) {
    //   nrows += domain.boxes[leaf_rows[i]].num_particles; }
    // for (unsigned i = 0; i < leaf_cols.size(); ++i) {
    //   ncols += domain.boxes[leaf_cols[i]].num_particles; }

    // assert(out.rows == nrows);
    // assert(out.cols == ncols);

    // std::vector<Particle> source_particles, target_particles;
    // for (unsigned i = 0; i < leaf_rows.size(); ++i) {
    //   int64_t source_box = leaf_rows[i];
    //   int64_t source = domain.boxes[source_box].start_index;
    //   for (unsigned n = 0; n < domain.boxes[source_box].num_particles; ++n) {
    //     source_particles.push_back(domain.particles[source + n]);
    //   }
    // }

    // for (unsigned i = 0; i < leaf_cols.size(); ++i) {
    //   int64_t target_box = leaf_cols[i];
    //   int64_t target = domain.boxes[target_box].start_index;
    //   for (int64_t n = 0; n < domain.boxes[target_box].num_particles; ++n) {
    //     target_particles.push_back(domain.particles[target + n]);
    //   }
    // }

    // for (int64_t i = 0; i < nrows; ++i) {
    //   for (int64_t j = 0; j < ncols; ++j) {
    //     out(i, j) = kernel(source_particles[i].coords,
    //                        target_particles[j].coords);
    //   }
    // }
  }

  // use
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   int64_t level, int64_t height,
                                   const kernel_function& kernel) {
    // if (level == height) {
    //   return generate_p2p_interactions(domain, irow, icol, kernel);
    // }

    // std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
    // std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

    // int64_t nrows = 0, ncols = 0;
    // for (unsigned i = 0; i < leaf_rows.size(); ++i) {
    //   nrows += domain.boxes[leaf_rows[i]].num_particles; }
    // for (unsigned i = 0; i < leaf_cols.size(); ++i) {
    //   ncols += domain.boxes[leaf_cols[i]].num_particles; }

    // Matrix out(nrows, ncols);
    // generate_p2p_interactions(domain, irow, icol, level, height,
    //                           kernel, out);

    // return out;
  }

  // use
  Matrix generate_p2p_matrix(const Domain& domain,
                             const kernel_function& kernel) {
    int64_t rows =  domain.particles.size();
    int64_t cols =  domain.particles.size();
    Matrix out(rows, cols);

    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
      #pragma omp parallel for
      for (int64_t j = 0; j < cols; ++j) {
        out(i, j) = kernel(domain.particles[i].coords,
                           domain.particles[j].coords);
      }
    }

    return out;
  }

  // use
  // Generates p2p interactions between the particles of two boxes specified by irow
  // and icol. ndim specifies the dimensionality of the particles present in domain.
  // Uses a laplace kernel for generating the interaction.
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   const kernel_function& kernel) {
    // Matrix out(domain.boxes[irow].num_particles, domain.boxes[icol].num_particles);
    // generate_p2p_interactions(domain, irow, icol, kernel, out);
    // return out;
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 const kernel_function& kernel,
                                 const int64_t rows, const int64_t cols,
                                 double* out,
                                 const int64_t ld) {
    // for (int64_t i = 0; i < rows; ++i) {
    //   for (int64_t j = 0; j < cols; ++j) {
    //     int64_t source = domain.boxes[irow].start_index;
    //     int64_t target = domain.boxes[icol].start_index;

    //     out[i + j * ld] = kernel(domain.particles[source+i].coords,
    //                              domain.particles[target+j].coords);
    //   }
    // }
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 const kernel_function& kernel,
                                 Matrix& out) {
    // assert(out.rows == domain.boxes[irow].num_particles);
    // assert(out.cols == domain.boxes[icol].num_particles);
    // generate_p2p_interactions(domain, irow, icol, kernel,
    //                           out.rows, out.cols, &out, out.stride);
  }


  // double
  // block_sin(const std::vector<double>& coords_row,
  //           const std::vector<double>& coords_col) {
  //   double dist = 0, temp;
  //   int64_t ndim = coords_row.size();

  //   for (int64_t k = 0; k < ndim; ++k) {
  //     dist += pow(coords_row[k] - coords_col[k], 2);
  //   }
  //   if (dist == 0) {
  //     return add_diag;
  //   }
  //   else {
  //     dist = std::sqrt(dist);
  //     return sin(wave_k * dist) / dist;
  //   }
  // }

  // double
  // sqrexp_kernel(const std::vector<double>& coords_row,
  //               const std::vector<double>& coords_col) {
  //   int64_t ndim = coords_row.size();
  //   double dist = 0;
  //   double local_beta = -2 * pow(beta, 2);
  //   // Copied from kernel_sqrexp.c in stars-H.
  //   for (int64_t k = 0; k < ndim; ++k) {
  //     dist += pow(coords_row[k] - coords_col[k], 2);
  //   }
  //   dist = dist / local_beta;
  //   if (std::abs(dist) < 1e-10) {
  //     return sigma + noise;
  //   }
  //   else {
  //     return sigma * exp(dist);
  //   }
  // }

  double laplace_kernel(const std::vector<double>& coords_row,
                        const std::vector<double>& coords_col,
                        const double eta) {
    int64_t ndim = coords_row.size();
    double rij = 0;
    for (int64_t k = 0; k < ndim; ++k) {
      rij += pow(coords_row[k] - coords_col[k], 2);
    }
    double out = 1 / (std::sqrt(rij) + eta);

    return out;
  }
}
