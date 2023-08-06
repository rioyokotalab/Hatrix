#include <vector>
#include <iostream>
#include <iomanip>
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
  dual_tree_traversal(SymmetricSharedBasisMatrix& A, const int64_t& Ci_index, const int64_t& Cj_index,
                      const Domain& domain, const Args& opts) {
    const Cell& Ci = domain.tree_list[Ci_index];
    const Cell& Cj = domain.tree_list[Cj_index];
    const int64_t i_level = Ci.level;
    const int64_t j_level = Cj.level;


    bool well_separated = false;
    if (i_level == j_level &&
        ((!opts.use_nested_basis && i_level == A.max_level) || opts.use_nested_basis)) {
      double distance = 0;

      for (int64_t k = 0; k < opts.ndim; ++k) {
        distance += pow(Ci.center[k] - Cj.center[k], 2);
      }
      distance = sqrt(distance);

      double ci_size = 0, cj_size = 0;
      for (int axis = 0; axis < opts.ndim; ++axis) {
        ci_size += pow(Ci.radii[axis], 2);
        cj_size += pow(Cj.radii[axis], 2);
      }

      if (distance > ((ci_size + cj_size) * opts.admis)) {
        well_separated = true;
      }

      bool val = well_separated;
      A.is_admissible.insert(Ci.key, Cj.key, i_level, std::move(val));
    }

    // Only descend down the tree if you are currently at a higher level and the blocks
    // at the current level are inadmissible. You then want to refine the tree further
    // since it has been found that the higher blocks are inadmissible.
    //
    // Alternatively, to create a BLR2 matrix you want to down to the finest level of granularity
    // anyway and populate the blocks at that level. So that puts another OR condition to check
    // if the use of nested basis is enabled.
    if (!well_separated || !opts.use_nested_basis) {
      if (i_level <= j_level && Ci.nchild > 0) {
        // j is at a higher level and i is not leaf.
        const int64_t c1_index = pow(2, i_level+1) - 1 + Ci.key * 2;
        const int64_t c2_index = pow(2, i_level+1) - 1 + Ci.key * 2 + 1;
        dual_tree_traversal(A, c1_index, Cj_index, domain, opts);
        dual_tree_traversal(A, c2_index, Cj_index, domain, opts);
      }
      else if (j_level <= i_level && Cj.nchild > 0) {
        // i is at a higheer level and j is not leaf.
        const int64_t c1_index = pow(2, j_level+1) - 1 + Cj.key * 2;
        const int64_t c2_index = pow(2, j_level+1) - 1 + Cj.key * 2 + 1;
        dual_tree_traversal(A, Ci_index, c1_index, domain, opts);
        dual_tree_traversal(A, Ci_index, c2_index, domain, opts);
      }
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

  static void
  populate_near_far_lists(SymmetricSharedBasisMatrix& A) {
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

  void
  init_geometry_admis(SymmetricSharedBasisMatrix& A,
                      const Domain& domain, const Args& opts) {
    // A.max_level = domain.tree.height() - 1;
    for (int i = 63; i < domain.tree_list.size(); ++i) {
      const Cell& c_i = domain.tree_list[i];
      for (int j = 63; j < domain.tree_list.size(); ++j) {
        const Cell& c_j = domain.tree_list[j];

        double dist = 0;
        for (int axis = 0; axis < opts.ndim; ++axis) {
          dist += pow(c_i.center[axis] - c_j.center[axis], 2);
          // std::cout << std::setw(20) << c.center[axis] << " ";
        }
        dist = std::sqrt(dist);

        std::cout << std::setw(10) << std::setprecision(4) << dist << " ";
      }
      std::cout << std::endl;
    }
    dual_tree_traversal(A, 0, 0, domain, opts);
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
          if ((A.is_admissible.exists(i, j, l) &&
               A.is_admissible(i, j, l)) || !A.is_admissible.exists(i, j, l)) {
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
    populate_near_far_lists(A);
  }

  static void
  compute_matrix_structure(SymmetricSharedBasisMatrix& A, int64_t level, const Args& opts) {
    if (level == 0) { return; }
    int64_t nodes = pow(2, level);
    if (level == A.max_level) {
      for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
          A.is_admissible.insert(i, j, level, std::abs(i - j) > opts.admis);
        }
      }
    }
    else {
      int64_t child_level = level + 1;
      for (int i = 0; i < nodes; ++i) {
        std::vector<int> row_children({i * 2, i * 2 + 1});
        for (int j = 0; j < nodes; ++j) {
          std::vector<int> col_children({j * 2, j * 2 + 1});

          bool admis_block = true;
          for (int c1 = 0; c1 < 2; ++c1) {
            for (int c2 = 0; c2 < 2; ++c2) {
              if (A.is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
                  !A.is_admissible(row_children[c1], col_children[c2], child_level)) {
                admis_block = false;
              }
            }
          }

          if (admis_block) {
            for (int c1 = 0; c1 < 2; ++c1) {
              for (int c2 = 0; c2 < 2; ++c2) {
                A.is_admissible.erase(row_children[c1], col_children[c2], child_level);
              }
            }
          }

          A.is_admissible.insert(i, j, level, std::move(admis_block));
        }
      }
    }

    compute_matrix_structure(A, level-1, opts);
  }

  void init_diagonal_admis(SymmetricSharedBasisMatrix& A,
                           const Domain& domain, const Args& opts) {
    A.max_level = domain.tree.height() - 1;
    if (opts.use_nested_basis) {
      compute_matrix_structure(A, A.max_level, opts);
    }
    else {
      int64_t level = A.max_level;
      int64_t nodes = pow(2, level);
      for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
          A.is_admissible.insert(i, j, level, std::abs(i - j) > opts.admis);
        }
      }

      // dense level for BLR2 matrix.
      level--;
      nodes = pow(2, level);
      for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
          A.is_admissible.insert(i, j, level, false);
        }
      }
    }
    A.min_level = -1;

    for (int64_t l = A.max_level; l > 0; --l) {
      int64_t nblocks = pow(2, l);
      bool all_dense = true;
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if ((A.is_admissible.exists(i, j, l) &&
               A.is_admissible(i, j, l)) || !A.is_admissible.exists(i, j, l)) {
            all_dense = false;
          }
        }
      }

      if (all_dense) {
        A.min_level = l;
        break;
      }
    }

    if ((A.max_level != A.min_level) && A.min_level != -1) { A.min_level++; }
    if (A.min_level == -1) {
	    A.min_level = 1; // HSS matrix detected.
    }
    if (opts.use_nested_basis && A.min_level == 1) {
      A.is_admissible.insert(0, 0, 0, false);
    }
    populate_near_far_lists(A);
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
}
