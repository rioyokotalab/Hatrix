#include <vector>
#include <cassert>
#include <cmath>

#include "distributed/distributed.hpp"

// This is actually RowLevelMap
Hatrix::RowColMap<std::vector<int64_t>> near_neighbours, far_neighbours;

namespace Hatrix {

  std::vector<Hatrix::Matrix>
  split_dense(const Hatrix::Matrix& dense, int64_t row_split, int64_t col_split) {
    return dense.split(std::vector<int64_t>(1, row_split),
                       std::vector<int64_t>(1, col_split));
  }

  bool
  exists_and_inadmissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t i, const int64_t j, const int64_t level) {
    return A.is_admissible.exists(i, j, level) && !A.is_admissible(i, j, level);
  }

  bool
  exists_and_admissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                        const int64_t i, const int64_t j, const int64_t level) {
    return A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level);
  }
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

  static void
  dual_tree_traversal(SymmetricSharedBasisMatrix& A, const Cell& Ci, const Cell& Cj,
                      const Domain& domain, const Args& opts) {
    int64_t i_level = Ci.level;
    int64_t j_level = Cj.level;

    bool well_separated = false;
    if (i_level == j_level &&
        ((!opts.use_nested_basis && i_level == A.max_level) ||
         opts.use_nested_basis)) {
      double distance = 0;
      for (int64_t k = 0; k < opts.ndim; ++k) {
        distance += pow(Ci.center[k] - Cj.center[k], 2);
      }
      distance = sqrt(distance);

      if (distance >= ((Ci.radius + Cj.radius) * opts.admis + opts.perturbation)) {
        // well-separated blocks.
        well_separated = true;
      }

      bool val = well_separated;
      A.is_admissible.insert(Ci.level_index, Cj.level_index, i_level, std::move(val));
    }

    // Only descend down the tree if you are currently at a higher level and the blocks
    // at the current level are inadmissible. You then want to refine the tree further
    // since it has been found that the higher blocks are inadmissible.
    //
    // Alternatively, to create a BLR2 matrix you want to down to the finest level of granularity
    // anyway and populate the blocks at that level. So that puts another OR condition to check
    // if the use of nested basis is enabled.
    if (i_level <= j_level && Ci.cells.size() > 0 && (!well_separated || !opts.use_nested_basis)) {
      // j is at a higher level and i is not leaf.
      dual_tree_traversal(A, Ci.cells[0], Cj, domain, opts);
      dual_tree_traversal(A, Ci.cells[1], Cj, domain, opts);
    }
    else if (j_level <= i_level && Cj.cells.size() > 0 && (!well_separated || !opts.use_nested_basis)) {
      // i is at a higheer level and j is not leaf.
      dual_tree_traversal(A, Ci, Cj.cells[0], domain, opts);
      dual_tree_traversal(A, Ci, Cj.cells[1], domain, opts);
    }
  }

  static void
  build_dense_level(SymmetricSharedBasisMatrix& A) {
    int64_t level = A.max_level - 1;
    int64_t nblocks = pow(2, level);

    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        A.is_admissible.insert(i, j, level, false);
      }
    }
  }

  void init_geometry_admis(SymmetricSharedBasisMatrix& A,
                           const Domain& domain, const Args& opts) {
    A.max_level = domain.tree.height() - 1;
    dual_tree_traversal(A, domain.tree, domain.tree, domain, opts);
    // Using BLR2 so need an 'artificial' dense matrix level at max_level-1
    // for accumulation of the partial factorization.
    if (!opts.use_nested_basis) {
      build_dense_level(A);
    }
    A.min_level = 0;
    for (int64_t l = A.max_level; l > 0; --l) {
      int64_t nblocks = pow(2, l);
      bool all_dense = true;
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if ((A.is_admissible.exists(i, j, l) && A.is_admissible(i, j, l)) || !A.is_admissible.exists(i, j, l)) {
            all_dense = false;
          }
        }
      }

      if (all_dense) {
        A.min_level = l;
        break;
      }
    }

    if (A.max_level != A.min_level) { A.min_level++; }

    // populate near and far lists. comment out when doing H2.
    for (int64_t level = A.max_level; level >= A.min_level; --level) {
      int64_t nblocks = pow(2, level);

      for (int64_t i = 0; i < nblocks; ++i) {
        far_neighbours.insert(i, level, std::vector<int64_t>());
        near_neighbours.insert(i, level, std::vector<int64_t>());
        for (int64_t j = 0; j <= i; ++j) {
          if (A.is_admissible.exists(i, j, level)) {
            if (A.is_admissible(i, j, level)) {
              far_neighbours(i, level).push_back(j);
            }
            else {
              near_neighbours(i, level).push_back(j);
            }
          }
        }
      }
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