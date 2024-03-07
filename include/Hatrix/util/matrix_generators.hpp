#pragma once
#include <cstdint>
#include <vector>

#include "greens_functions.hpp"

#include "Hatrix/classes/Domain.hpp"
#include "Hatrix/classes/Matrix.hpp"

namespace Hatrix {

Matrix generate_random_spd_matrix(int64_t rows, double diag_scale=1.0);

Matrix generate_random_matrix(int64_t rows, int64_t cols);

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols);

Matrix generate_identity_matrix(int64_t rows, int64_t cols);

Matrix generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range);

// Generate P2P interactions for the entire domain and return a dense matrix.
Matrix generate_p2p_interactions(const Domain& domain,
                                 const greens_functions::kernel_function_t& kernel);

// Generate P2P interactions for a subset of particles in the rows and columns and
// return a block matrix of size nrows x ncols. The row of the matrix with be the
// irow'th particle in the domain, and the column of the matrix with the icol'th
// particle in the domain.
Matrix generate_p2p_interactions(const Domain& domain,
                                 const int64_t irow, const int64_t nrows,
                                 const int64_t icol, const int64_t ncols,
                                 const greens_functions::kernel_function_t& kernel);

Matrix generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv=1e-3);
}  // namespace Hatrix
