#pragma once

#include "Hatrix/Hatrix.hpp"

#include "internal_types.hpp"
#include "SymmetricSharedBasisMatrix.hpp"

// storage for near and far blocks at each level.
extern Hatrix::RowColMap<std::vector<int64_t>> near_neighbours, far_neighbours;  // This is actually RowLevelMap

namespace Hatrix {

  // double sqrexp_kernel(const std::vector<double>& coords_row,
  //                      const std::vector<double>& coords_col);
  // double block_sin(const std::vector<double>& coords_row,
  //                  const std::vector<double>& coords_col);

  bool
  exists_and_inadmissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t i, const int64_t j, const int64_t level);

  bool
  exists_and_admissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                        const int64_t i, const int64_t j, const int64_t level);

  std::vector<Hatrix::Matrix>
  split_dense(const Hatrix::Matrix& dense, int64_t row_split, int64_t col_split);

  Matrix
  make_complement(const Matrix& Q);

  void
  search_tree_for_nodes(const Cell& tree,
                        const int64_t level_index,
                        const int64_t level,
                        int64_t &pstart, int64_t &pend);

  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow,
                                 int64_t icol,
                                 int64_t level,
                                 const kernel_function& kernel,
                                 Matrix& out);

  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow,
                                   int64_t icol,
                                   int64_t level,
                                   const kernel_function& kernel);

  Matrix generate_p2p_matrix(const Domain& domain,
                             const kernel_function& kernel);


  void init_geometry_admis(SymmetricSharedBasisMatrix& A,
                           const Domain& domain, const Args& opts);

  void init_diagonal_admis(SymmetricSharedBasisMatrix& A,
                           const Domain& domain, const Args& opts);
}
