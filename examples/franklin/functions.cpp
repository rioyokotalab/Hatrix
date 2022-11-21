#include <vector>
#include <cassert>
#include <cmath>

#include "franklin/franklin.hpp"

namespace Hatrix {
  Matrix
  make_complement(const Matrix& Q) {
    Hatrix::Matrix Q_F(Q.rows, Q.rows);
    Hatrix::Matrix Q_full, R;
    std::tie(Q_full, R) = qr(Q,
                             Hatrix::Lapack::QR_mode::Full,
                             Hatrix::Lapack::QR_ret::OnlyQ);

    for (int64_t i = 0; i < Q_F.rows; ++i) {
      for (int64_t j = 0; j < Q_F.cols - Q.cols; ++j) {
        Q_F(i, j) = Q_full(i, j + Q.cols);
      }
    }

    for (int64_t i = 0; i < Q_F.rows; ++i) {
      for (int64_t j = 0; j < Q.cols; ++j) {
        Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
      }
    }
    return Q_F;
  }

  void
  search_tree_for_nodes(const Cell& tree, const int64_t level_index, const int64_t level,
                        int64_t &pstart, int64_t &pend) {
    if (tree.level == level && tree.level_index == level_index) {
      pstart = tree.start_index;
      pend = tree.end_index;
      return;
    }

    if (tree.cells.size() > 0) {
      search_tree_for_nodes(tree.cells[0], level_index, level, pstart, pend);
      search_tree_for_nodes(tree.cells[1], level_index, level, pstart, pend);
    }
  }

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow,
                                 int64_t icol,
                                 int64_t level,
                                 const kernel_function& kernel,
                                 Matrix& out) {
    int64_t source_pstart, source_pend, target_pstart, target_pend;

    search_tree_for_nodes(domain.tree, irow, level, source_pstart, source_pend);
    search_tree_for_nodes(domain.tree, icol, level, target_pstart, target_pend);

#pragma omp parallel for
    for (int64_t i = 0; i < source_pend - source_pstart; ++i) {
#pragma omp parallel for
      for (int64_t j = 0; j < target_pend - target_pstart; ++j) {
        out(i, j) = kernel(domain.particles[i + source_pstart].coords,
                           domain.particles[j + target_pstart].coords);
      }
    }
  }

  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow,
                                   int64_t icol,
                                   int64_t level,
                                   const kernel_function& kernel) {
    int64_t nrows = domain.cell_size(irow, level);
    int64_t ncols = domain.cell_size(icol, level);
    Matrix out(nrows, ncols);

    generate_p2p_interactions(domain, irow, icol, level, kernel, out);
    return out;
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
}
