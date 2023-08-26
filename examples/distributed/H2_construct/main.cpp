#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <random>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "factorize_noparsec.hpp"
#include "factorize_ptg.hpp"

#include "parsec.h"
#include "mpi.h"
#include "omp.h"

#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef HAS_ELSES_ENABLED
extern "C" {
#include "elses.h"
}
#endif

using namespace Hatrix;

constexpr double EPS = 1e-13;// std::numeric_limits<double>::epsilon();
static const int BEGIN_PROW = 0, BEGIN_PCOL = 0;
const int ONE = 1;

int BLACS_CONTEXT;

int
indxg2l(int INDXGLOB, int NB, int NPROCS) {
  return NB * ((INDXGLOB - 1) / ( NB * NPROCS)) + (INDXGLOB - 1) % NB + 1;
}

int
indxl2g(int indxloc, int nb, int iproc, int nprocs) {
  return nprocs * nb * ((indxloc - 1) / nb) +
    (indxloc-1) % nb + ((nprocs + iproc) % nprocs) * nb + 1;
}

int
translate_rank_comm_world(int num_ranks, int right_comm_rank, MPI_Comm right_comm) {
  std::vector<int> left_rank(num_ranks), right_rank(num_ranks);
  MPI_Group left_group, right_group;

  MPI_Comm_group(MPI_COMM_WORLD, &left_group);
  MPI_Comm_group(right_comm, &right_group);
  for (int i = 0; i < num_ranks; ++i) {
    left_rank[i] = i;
  }

  MPI_Group_translate_ranks(left_group, num_ranks, left_rank.data(),
                            right_group, right_rank.data());

  if (MPIRANK == 0) {
    std::cout << "ranks: " << right_comm_rank <<  std::endl;
    for (int i = 0; i < num_ranks; ++i) {
      std::cout << right_rank[i] << " " << std::endl;
    }
  }

  return right_rank[right_comm_rank];
}

class ScaLAPACK_dist_matrix_t {
public:
  // scalapack storage for matrix descriptor.
  int nrows, ncols, block_nrows, block_ncols, local_stride;
  std::vector<double> data;
  std::vector<int> DESC;
  int local_nrows, local_ncols;

  ScaLAPACK_dist_matrix_t(int nrows, int ncols,
                          int block_nrows, int block_ncols,
                          int begin_prow, int begin_pcol,
                          int BLACS_CONTEXT) :
    nrows(nrows), ncols(ncols), block_nrows(block_nrows), block_ncols(block_ncols)
  {
    local_nrows = numroc_(&nrows, &block_nrows, &MYROW, &begin_prow, &MPIGRID[0]);
    local_ncols = numroc_(&ncols, &block_ncols, &MYCOL, &begin_pcol, &MPIGRID[1]);
    local_stride = local_nrows;

    int INFO;
    DESC.resize(9);
    descinit_(DESC.data(), &nrows, &ncols, &block_nrows, &block_ncols,
              &begin_prow, &begin_pcol, &BLACS_CONTEXT, &local_nrows, &INFO);

    try {
      data.resize((size_t)local_nrows * (size_t)local_ncols, 0);
    }
    catch (std::bad_alloc & exception) {
      std::cerr << "tried to allocate memory of size:  "
                << (size_t)local_nrows * (size_t)local_ncols
                << " " << exception.what() << std::endl;
    }

  }

  int glob_row(int local_row) {
    return indxl2g(local_row + 1, block_nrows, MYROW, MPIGRID[0]) - 1;
  }

  int glob_col(int local_col) {
    return indxl2g(local_col + 1, block_ncols, MYCOL, MPIGRID[1]) - 1;
  }

  void set_local(size_t local_row, size_t local_col, double value) {
    data[local_row + local_col * (size_t)local_nrows] = value;
  }
};

int64_t
get_basis_min_rank(const SymmetricSharedBasisMatrix& A, const Args& opts) {
  return opts.max_rank; // const rank code.
}

int64_t
get_basis_max_rank(const SymmetricSharedBasisMatrix& A, const Args& opts) {
  return opts.max_rank; // const rank code.
}

void
shift_diagonal(Matrix& dense, const double shift) {
  for(int64_t i = 0; i < dense.min_dim(); i++) {
    dense(i, i) += shift;
  }
}

void
factorize_dtd(SymmetricSharedBasisMatrix& A,
              const Domain& domain,
              const Args& opts) {
}


std::tuple<int64_t, int64_t, int64_t, bool>
inertia(const SymmetricSharedBasisMatrix& A,
        const Domain& domain,
        const Args& opts,
        const double lambda) {
  bool singular = false;
  SymmetricSharedBasisMatrix A_shifted(A);
  int64_t nblocks = pow(2, A.max_level);
  for (int64_t node = MPIRANK; node < nblocks; node += MPISIZE) {
    shift_diagonal(A_shifted.D(node, node, A_shifted.max_level), -lambda);
  }

  // SymmetricSharedBasisMatrix A_shifted_ptg(A_shifted);
  factorize_ptg(A_shifted, domain, opts);

  // factorize_noparsec(A_shifted, domain, opts);

  int64_t negative_elements_count = 0;

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    for (int64_t node = MPIRANK; node < nblocks; node += MPISIZE) {
      const Matrix& D_node = A_shifted.D(node, node, level);
      const auto D_node_splits =
        D_node.split(std::vector<int64_t>{D_node.rows - opts.max_rank},
                     std::vector<int64_t>{D_node.cols - opts.max_rank});
      const Matrix& D_lambda = D_node_splits[0];
      for(int64_t i = 0; i < D_lambda.min_dim(); i++) {
        negative_elements_count += (D_lambda(i, i) < 0 ? 1 : 0);
        if(std::isnan(D_lambda(i, i)) || std::abs(D_lambda(i, i)) < EPS) {
          singular = true;
        }
      }
    }
  }

  // Remaining blocks that are factorized as block-dense matrix
  const int64_t level = A.min_level - 1;
  const int64_t num_nodes = pow(2, level);
  for (int64_t node = MPIRANK; node < num_nodes; node += MPISIZE) {
    const Matrix& D_lambda = A_shifted.D(node, node, level);
    for(int64_t i = 0; i < D_lambda.min_dim(); i++) {
      negative_elements_count += (D_lambda(i, i) < 0 ? 1 : 0);
      if(std::isnan(D_lambda(i, i)) || std::abs(D_lambda(i, i)) < EPS) {
        singular = true;
      }
    }
  }

  const int64_t ldl_min_rank = get_basis_min_rank(A, opts);
  const int64_t ldl_max_rank = get_basis_max_rank(A, opts);

  return {negative_elements_count, ldl_min_rank, ldl_max_rank, singular};
}

std::tuple<double, int64_t, int64_t, double>
get_mth_eigenvalue(const SymmetricSharedBasisMatrix& A,
                   const Domain& domain, const Args& opts,
                   const int64_t k, const double ev_tol,
                   double left, double right) {
  int64_t shift_min_rank = get_basis_min_rank(A, opts);
  int64_t shift_max_rank = get_basis_max_rank(A, opts);
  double max_rank_shift = -1;
  bool singular;
  while((right - left) >= ev_tol) {
    const double mid = (left + right) / 2;
    // std::cout << "left: " << left << " right: " << right
    //           << " mid: " << mid << std::endl;
    int64_t num_negative_values, factor_min_rank, factor_max_rank;
    std::tie(num_negative_values, factor_min_rank, factor_max_rank, singular) =
      inertia(A, domain, opts, mid);

    if (factor_max_rank > shift_max_rank) {
      shift_min_rank = factor_min_rank;
      shift_max_rank = factor_max_rank;
      max_rank_shift = mid;
    }

    if (singular) {
      std::cout << "Shifted matrix became singular (shift=" << mid << ")" << std::endl;
      break;
    }

    if (num_negative_values >= k) {
      right = mid;
    }
    else {
      left = mid;
    }
  }

  return {(left + right) / 2, shift_min_rank, shift_max_rank, max_rank_shift};
}

// i, j, level -> block numbers.
Matrix
generate_p2p_interactions(int64_t i, int64_t j, int64_t level, const Args& opts,
                          const Domain& domain,
                          const SymmetricSharedBasisMatrix& A) {
  int64_t block_size = opts.N / A.num_blocks[level];
  Matrix dense(block_size, block_size);


  for (int64_t local_i = 0; local_i < block_size; ++local_i) {
    for (int64_t local_j = 0; local_j < block_size; ++local_j) {
      long int global_i = i * block_size + local_i;
      long int global_j = j * block_size + local_j;
      double value;
      if (opts.kernel_verbose == "elses_c60") {
        global_i += 1;
        global_j += 1;

#ifdef HAS_ELSES_ENABLED
        get_elses_matrix_value(&global_i, &global_j, &value);
#else
        abort();
#endif
      }
      else {
        value = opts.kernel(domain.particles[global_i].coords,
                            domain.particles[global_j].coords);
      }

      dense(local_i, local_j) = value;
    }
  }

  return dense;
}

void generate_leaf_nodes(SymmetricSharedBasisMatrix& A,
                         const Domain& domain, const Args& opts) {
  int64_t nblocks = A.num_blocks[A.max_level];
  int64_t block_size = opts.N / nblocks;

  // Generate dense blocks and store them in the appropriate structure.
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        Matrix Aij = generate_p2p_interactions(i, j, A.max_level, opts,
                                               domain, A);
        A.D.insert(i, j, A.max_level, std::move(Aij));
      }
    }
  }

  double ALPHA = 1.0, BETA = 1.0;

  // Accumulate admissible blocks from the large dist matrix.
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) { // row cylic distribution.
    Matrix AY(opts.nleaf, opts.nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {

#pragma omp parallel for collapse(2)
        for (int64_t local_i = 0; local_i < block_size; ++local_i) {
          for (int64_t local_j = 0; local_j < block_size; ++local_j) {
            long int global_i = i * block_size + local_i;
            long int global_j = j * block_size + local_j;
            double value;
            if (opts.kernel_verbose == "elses_c60") {
              global_i += 1;
              global_j += 1;
#ifdef HAS_ELSES_ENABLED
              get_elses_matrix_value(&global_i, &global_j, &value);
#else
              abort();
#endif
            }
            else {
              value = opts.kernel(domain.particles[global_i].coords,
                                  domain.particles[global_j].coords);
            }
            AY(local_i, local_j) += value;
          }
        }
      }
    }

    Matrix Ui, Si, _V; double error;
    std::tie(Ui, Si, _V, error) = truncated_svd(AY, opts.max_rank);

    A.U.insert(i, A.max_level, std::move(Ui));
    A.US.insert(i, A.max_level, std::move(Si));
  }

  // Allgather the bases for generation of skeleton blocks.
  int temp_blocks = nblocks < MPISIZE ? MPISIZE  : (nblocks / MPISIZE + 1) * MPISIZE;

  std::vector<double> temp_bases(opts.nleaf * opts.max_rank * temp_blocks, 0);
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    int64_t temp_bases_offset = ((i / MPISIZE) * MPISIZE) * opts.nleaf * opts.max_rank;

    MPI_Allgather(&A.U(i, A.max_level),
                  opts.nleaf * opts.max_rank,
                  MPI_DOUBLE,
                  temp_bases.data() + temp_bases_offset,
                  opts.nleaf * opts.max_rank,
                  MPI_DOUBLE,
                  MPI_COMM_WORLD);
  }

  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        int64_t temp_bases_offset = j * opts.nleaf * opts.max_rank;
        Matrix Uj(opts.nleaf, opts.max_rank);

#pragma omp parallel for collapse(2)
        for (int64_t Uj_i = 0; Uj_i < opts.nleaf; ++Uj_i) {
          for (int64_t Uj_j = 0; Uj_j < opts.max_rank; ++Uj_j) {
            Uj(Uj_i, Uj_j) = temp_bases[temp_bases_offset + Uj_i + Uj_j * opts.nleaf];
          }
        }

        Matrix temp = matmul(A.U(i, A.max_level),
                             generate_p2p_interactions(i, j, A.max_level, opts,
                                                       domain, A),
                             true, false);
        auto S_block = matmul(temp, Uj);
        A.S.insert(i, j, A.max_level, std::move(S_block));
      }
    }
  }
}

static RowLevelMap
generate_transfer_matrices(SymmetricSharedBasisMatrix& A,
                           const Args& opts, const RowLevelMap& Uchild) {
  RowLevelMap Ubig_parent;

  return Ubig_parent;
}

void construct_H2_matrix(SymmetricSharedBasisMatrix& A,
                         const Domain& domain,
                         const Args& opts) {
  generate_leaf_nodes(A, domain, opts);
  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    Uchild = generate_transfer_matrices(A, opts, Uchild);
  }

  // Final dense block for the factoirization.
  int64_t final_level = A.min_level - 1;
  int64_t final_nblocks = pow(2, final_level);
  for (int64_t i = 0; i < final_nblocks; ++i) {
    for (int64_t j = 0; j < final_nblocks; ++j) {
      A.D.insert(i, j, final_level,
                 Matrix(opts.max_rank * 2, opts.max_rank * 2));
    }
  }

  // Generate graphs for traversal.

}

// H2 matrix-vector product.
// b = H2_A * x
void
dist_matvec_h2(const SymmetricSharedBasisMatrix& A,
                  const Domain& domain,
                  const Args& opts,
                  const std::vector<Matrix>& x,
                  std::vector<Matrix>& b) {
  // Multiply V.T with x.
  int64_t nblocks = pow(2, A.max_level);
  int64_t x_hat_offset = 0;
  std::vector<Matrix> x_hat;
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    int64_t index_i = i / MPISIZE;
    Matrix x_hat_i = matmul(A.U(i, A.max_level), x[index_i], true, false);
    x_hat.push_back(x_hat_i);
    x_hat_offset++;
  }

  // Init temp blocks for intermediate products.
  std::vector<Matrix> b_hat;
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    b_hat.push_back(Matrix(opts.max_rank, 1));
  }

  // Multiply S blocks with the x_hat intermediate products.
  for (int64_t j = 0; j < nblocks; ++j) { // iterate over columns.
    int64_t index_j = j / MPISIZE;
    Matrix x_hat_j(opts.max_rank, 1);
    if (mpi_rank(j) == MPIRANK) {
      x_hat_j = x_hat[index_j];
    }
    MPI_Bcast(&x_hat_j, x_hat_j.numel(), MPI_DOUBLE, mpi_rank(j), MPI_COMM_WORLD);

    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      int64_t index_i = i / MPISIZE;
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        const Matrix& Sij = A.S(i, j, A.max_level);
        matmul(Sij, x_hat_j, b_hat[index_i]);
      }
    }
  }

  // Update with row bases.
  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    int64_t index_i = i / MPISIZE;
    matmul(A.U(i, A.max_level), b_hat[index_i], b[index_i], false, false, 1, 1);
  }

  for (int64_t j = 0; j < nblocks; ++j) {
    Matrix x_j(opts.nleaf, 1);
    if (mpi_rank(j) == MPIRANK) {
      int j_index = j / MPISIZE;
      x_j = x[j_index];
    }

    MPI_Bcast(&x_j, x_j.numel(), MPI_DOUBLE, mpi_rank(j), MPI_COMM_WORLD);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        int i_index = i / MPISIZE;
        matmul(A.D(i, j, A.max_level), x_j, b[i_index], false, false, 1, 1);
      }
    }
  }
}

// Distributed dense matrix vector product. Generates the dense matrix on the fly.
// b = dense_A * x
void
dist_matvec_dense(const SymmetricSharedBasisMatrix& A,
                  const Domain& domain,
                  const Args& opts,
                  const std::vector<Matrix>& x,
                  std::vector<Matrix>& b) {
  int64_t nblocks = pow(2, A.max_level);
  for (int64_t j = 0; j < nblocks; ++j) {
    Matrix xj(opts.nleaf, 1);
    if (mpi_rank(j) == MPIRANK) {
      int j_index = j / MPISIZE;
      xj = x[j_index];
    }

    MPI_Bcast(&xj, xj.numel(), MPI_DOUBLE, mpi_rank(j), MPI_COMM_WORLD);
    for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
      Matrix Aij = generate_p2p_interactions(i, j, A.max_level, opts,
                                             domain, A);
      int i_index = i / MPISIZE;
      matmul(Aij, xj, b[i_index]); //  b = 1 * b + 1 * A * x
    }
  }
}

void
construct_h2_matrix_graph_structures(const SymmetricSharedBasisMatrix& A,
                                     const Domain& domain,
                                     const Args& opts) {

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);

    for (int64_t i = 0; i < nblocks; ++i) {
      std::vector<int64_t> near_i, far_i;

      for (int64_t j = 0; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, level) &&
            !A.is_admissible(i, j, level)) {
          near_i.push_back(j);
        }

        if (A.is_admissible.exists(i, j, level) &&
            A.is_admissible(i, j, level)) {
          far_i.push_back(j);
        }
      }

      near_neighbours.insert(i, level, std::move(near_i));
      far_neighbours.insert(i, level, std::move(far_i));
    }
  }

  // Populate the last blocks of dense for the final factorization.
  int64_t level = A.min_level - 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> near_i;
    for (int64_t j = 0; j < nblocks; ++j) {
      near_i.push_back(j);
    }
    near_neighbours.insert(i, level, std::move(near_i));
  }
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  const double ev_tol = 1e-7;
  int N = opts.N;

  assert(opts.N % opts.nleaf == 0);

  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);

  SymmetricSharedBasisMatrix A;

  // Init domain decomposition for H2 matrix using dual tree traversal.
  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();

    if (opts.ndim == 1 || opts.ndim == 2) {
      A.max_level = log2(opts.N / opts.nleaf);
      domain.cardinal_sort_and_cell_generation(opts.nleaf);
    }
    else if (opts.ndim == 3) {
      abort();
      domain.sector_sort(opts.nleaf);
      domain.build_bottom_up_binary_tree(opts.nleaf);
    }
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
    A.max_level = log2(opts.N / opts.nleaf);

    if (opts.ndim == 1 || opts.ndim == 2) {
      A.max_level = log2(opts.N / opts.nleaf);
      domain.cardinal_sort_and_cell_generation(opts.nleaf);
    }
    else if (opts.ndim == 3) {
      abort();
      domain.sector_sort(opts.nleaf);
      domain.build_bottom_up_binary_tree(opts.nleaf);
    }
  }
  else if (opts.kind_of_geometry == COL_FILE) {
    domain.read_col_file_3d(opts.geometry_file);
    A.max_level = log2(opts.N / opts.nleaf);
    domain.sector_sort(opts.nleaf);
    domain.build_bottom_up_binary_tree(opts.nleaf);
  }
  else if (opts.kind_of_geometry == ELSES_C60_GEOMETRY) {
    const int64_t num_electrons_per_atom = 4;
    const int64_t num_atoms_per_molecule = 60;
#ifdef HAS_ELSES_ENABLED
    init_elses_state();
#else
    abort();
#endif
    domain.read_xyz_chemical_file(opts.geometry_file, num_electrons_per_atom);
    A.max_level = domain.build_elses_tree(num_electrons_per_atom * num_atoms_per_molecule);
    A.min_level = 0;
  }

  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  if (!MPIRANK)
    std::cout << "Domain setup time: " << domain_time << "ms"
              << " leaf: " << opts.nleaf
              << " ndim: " << opts.ndim
              << std::endl;

  // Generate a dense matrix and compute its eigenvalues for verification.
  std::vector<double> DENSE_EIGENVALUES(opts.N, 0);
#ifdef VERIFY_EIGEN
  {
    MPI_Dims_create(MPISIZE, 2, MPIGRID); // init 2D grid for scalapack
    Cblacs_get(-1, 0, &BLACS_CONTEXT );
    Cblacs_gridinit(&BLACS_CONTEXT, "Row", MPIGRID[0], MPIGRID[1]);
    Cblacs_pcoord(BLACS_CONTEXT, MPIRANK, &MYROW, &MYCOL);

    ScaLAPACK_dist_matrix_t dist_dense(opts.N, opts.N, opts.nleaf, opts.nleaf,
                                       BEGIN_PROW, BEGIN_PCOL, BLACS_CONTEXT);

#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < dist_dense.local_nrows; ++i) {
      for (int64_t j = 0; j < dist_dense.local_ncols; ++j) {
        int g_row = dist_dense.glob_row(i), g_col = dist_dense.glob_col(j);
        double value = opts.kernel(domain.particles[g_row].coords,
                                   domain.particles[g_col].coords);
        dist_dense.set_local(i, j, value);
      }
    }

    // Compute all the eigen values of the scalapack distributed matrix.
    const char JOBZ = 'N';
    const char UPLO = 'L';
    int LWORK = -1;
    std::vector<double> WORK(1);
    int INFO;
    pdsyev_(&JOBZ, &UPLO,
            &N, dist_dense.data.data(), &ONE, &ONE, dist_dense.DESC.data(),
            DENSE_EIGENVALUES.data(),
            NULL, NULL, NULL, NULL,
            WORK.data(), &LWORK, &INFO);

    LWORK = WORK[0];
    WORK.resize(LWORK);

    pdsyev_(&JOBZ, &UPLO,
            &N, dist_dense.data.data(), &ONE, &ONE, dist_dense.DESC.data(),
            DENSE_EIGENVALUES.data(),
            NULL, NULL, NULL, NULL,
            WORK.data(), &LWORK, &INFO);

    Cblacs_gridexit(BLACS_CONTEXT);
    Cblacs_exit(1);
  }

  // {
  //   Matrix AA(opts.N, opts.N);
  //   for (int i = 0; i < N; ++i) {
  //     for (int j = 0; j < N; ++j) {
  //       AA(i, j) = opts.kernel(domain.particles[i].coords,
  //                              domain.particles[j].coords);
  //     }
  //   }

  //   DENSE_EIGENVALUES = Hatrix::get_singular_values(AA);
  // }
#endif

  auto start_construct = std::chrono::system_clock::now();

  int64_t construct_max_rank;

  A.num_blocks.resize(A.max_level+1);
  A.num_blocks[A.max_level] = opts.N/opts.nleaf;
  if (opts.admis_kind == GEOMETRY) {
    init_geometry_admis(A, domain, opts); // init admissiblity conditions with DTT
  }
  else if (opts.admis_kind == DIAGONAL) {
    // init admissiblity conditions with diagonal condition.
    init_diagonal_admis(A, domain, opts);
  }

  global_is_admissible.deep_copy(A.is_admissible);
  // A.print_structure();
  construct_H2_matrix(A, domain, opts);
  auto stop_construct = std::chrono::system_clock::now();
  double construct_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_construct - start_construct).count();

  // Check the construction error.
  std::vector<Matrix> x, actual_b, expected_b;
  for (int i = MPIRANK; i < pow(2, A.max_level); i += MPISIZE) {
    x.push_back(Matrix(opts.nleaf, 1));
    actual_b.push_back(Matrix(opts.nleaf, 1));
    expected_b.push_back(Matrix(opts.nleaf, 1));
  }

  std::mt19937 gen(MPIRANK);
  std::uniform_real_distribution<double> dist(0, 1);
  for (int block = MPIRANK; block < pow(2, A.max_level); block += MPISIZE) {
    int index = block / MPISIZE;

    for (int i = 0; i < opts.nleaf; ++i) {
      int g_row = block * opts.nleaf + i + 1;
      double value = i;

      int l_row = indxg2l(g_row, opts.nleaf, MPISIZE) - 1;
      int l_col = 0;

      x[index](i, 0) = value;  // assign random data to x.
      actual_b[index](i, 0) = 0.0;    // set b to 0.
      expected_b[index](i, 0) = 0.0;    // set b to 0.
    }
  }

  // multiply dense matix with x. DENSE * x = actual_b
  dist_matvec_dense(A, domain, opts, x, actual_b);

  // multiply H2 matrix with x. H2 * x = expected_b
  dist_matvec_h2(A, domain, opts, x, expected_b);

  double local_norm[2] = {0, 0};
  for (int64_t i = MPIRANK; i < pow(2, A.max_level); i += MPISIZE) {
    int64_t index_i = i / MPISIZE;
    local_norm[0] += pow(norm(actual_b[index_i] - expected_b[index_i]), 2);
    local_norm[1] += pow(norm(actual_b[index_i]), 2);
  }
  double global_norm[2] = {0, 0};
  MPI_Reduce(&local_norm,
             &global_norm,
             2,
             MPI_DOUBLE,
             MPI_SUM,
             0,                 // root process
             MPI_COMM_WORLD);
  global_norm[0] = sqrt(global_norm[0]);
  global_norm[1] = sqrt(global_norm[1]);
  // Finish the construction verification.


  double kth_value_time;
  // Intervals within which the eigen values should be searched.
  {
#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif
    omp_set_num_threads(1);

    parsec = parsec_init( 1, NULL, NULL );
    construct_h2_matrix_graph_structures(A, domain, opts);

    bool singular = false;
    std::vector<int64_t> target_m;
    int64_t m_begin = N/2, m_end = N/2; // find eigen values from m_begin to m_end.
    for (int64_t m = m_begin; m <= m_end; ++m) {
      target_m.push_back(m);
    }

    double b = N * (1 / opts.param_1); // default values from ridwan.
    double a = -b;

    int64_t v_a = 0, v_b = N, temp1, temp2;
    // std::tie(v_a, temp1, temp2, singular) = inertia(A, domain, opts, a);
    // std::tie(v_b, temp1, temp2, singular) = inertia(A, domain, opts, b);

    if(v_a != 0 || v_b != N) {
      std::cout << std::endl
                << "Warning: starting interval does not contain the whole spectrum "
                << "(v(a)=v(" << a << ")=" << v_a << ","
                << " v(b)=v(" << b << ")=" << v_b << ")"
                << std::endl;
    }

    auto start_kth_value_time = std::chrono::system_clock::now();
    for (int64_t k : target_m) {
      double h2_mth_eigv, max_rank_shift;
      int64_t ldl_min_rank, ldl_max_rank;

      std::tie(h2_mth_eigv, ldl_min_rank, ldl_max_rank, max_rank_shift) =
        get_mth_eigenvalue(A, domain, opts, k, ev_tol, a, b);

      const double dense_mth_eigv = DENSE_EIGENVALUES[k-1];
      const double eigv_abs_error = std::abs(dense_mth_eigv - h2_mth_eigv);

      std::cout << "Compute eigenvalue k: " << k
                << " abs_error: " << eigv_abs_error
                << " check: " << (eigv_abs_error < 0.5 * ev_tol)
                << " dense: " << dense_mth_eigv
                << " H2   : " << h2_mth_eigv
                << std::endl;
    }
    auto stop_kth_value_time = std::chrono::system_clock::now();
    kth_value_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_kth_value_time - start_kth_value_time).count();

    parsec_fini(&parsec);
  }

  if (!MPIRANK) {
    double diff = global_norm[0];
    double actual =  global_norm[1];

    std::cout << "N       : " << opts.N << std::endl
              << "nleaf   : " << opts.nleaf << std::endl
              << "max rank: " << opts.max_rank << std::endl
              << "construct rel err: " << diff / actual << std::endl
              << "Csp              : " << A.Csp(A.max_level) << std::endl
              << "construct time   : " << construct_time << std::endl
              << "kth value time   : " << kth_value_time
              << std::endl;
  }


  MPI_Finalize();

  if (!MPIRANK) {
    std::cout << "Everything finished.\n";
  }

  return 0;
}
