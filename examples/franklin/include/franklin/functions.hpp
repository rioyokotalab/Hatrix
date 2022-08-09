#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {

  // double sqrexp_kernel(const std::vector<double>& coords_row,
  //                      const std::vector<double>& coords_col);
  // double block_sin(const std::vector<double>& coords_row,
  //                  const std::vector<double>& coords_col);

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

  double laplace_kernel(const std::vector<double>& coords_row,
                        const std::vector<double>& coords_col,
                        const double eta);

}
