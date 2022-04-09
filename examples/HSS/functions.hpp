#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 int64_t level, int64_t height,
                                 const kernel_function& kernel,
                                 Matrix& out);
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   int64_t level, int64_t height);
  Matrix generate_p2p_matrix(const Domain& domain,
                             const kernel_function& kernel);

  // Generates p2p interactions between the particles of two boxes specified by irow
  // and icol. ndim specifies the dimensionality of the particles present in domain.
  // Uses a laplace kernel for generating the interaction.
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol,
                                   const kernel_function& kernel);
  void generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 const kernel_function& kernel,
                                 Matrix& out);
}
