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

#include "mpi.h"

extern "C" {
#include "elses.h"
}

using namespace Hatrix;

static const int BEGIN_PROW = 0, BEGIN_PCOL = 0;

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

// i, j, level -> block numbers.
Matrix
generate_p2p_interactions(int64_t i, int64_t j, int64_t level, const Args& opts,
                          const Domain& domain,
                          const SymmetricSharedBasisMatrix& A) {
  int64_t block_size = opts.N / A.num_blocks[level];
  Matrix dense(block_size, block_size);

#pragma omp parallel for collapse(2)
  for (int64_t local_i = 0; local_i < block_size; ++local_i) {
    for (int64_t local_j = 0; local_j < block_size; ++local_j) {
      long int global_i = i * block_size + local_i;
      long int global_j = j * block_size + local_j;
      double value;
      if (opts.kernel_verbose == "elses_c60") {
        global_i += 1;
        global_j += 1;
        get_elses_matrix_value(&global_i, &global_j, &value);
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
  for (int64_t i = 0; i < nblocks; ++i) {
    if (mpi_rank(i) == MPIRANK) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, A.max_level) &&
            !A.is_admissible(i, j, A.max_level)) {
          Matrix Aij = generate_p2p_interactions(i, i, A.max_level, opts,
                                                 domain, A);
          A.D.insert(i, i, A.max_level, std::move(Aij));
        }
      }
    }
  }

  double ALPHA = 1.0, BETA = 1.0;

  // Accumulate admissible blocks from the large dist matrix.
  for (int64_t i = MPIRANK; i < nblocks; i += MPIGRID[0]) { // row cylic distribution.
    Matrix AY(opts.nleaf, opts.nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) && A.is_admissible(i, j, A.max_level)) {

#pragma omp parallel for collapse(2)
        for (int64_t local_i = 0; local_i < block_size; ++local_i) {
          for (int64_t local_j = 0; local_j < block_size; ++local_j) {
            long int global_i = i * block_size + local_i;
            long int global_j = j * block_size + local_j;
            double value;
            if (opts.kernel_verbose == "elses_c60") {
              global_i += 1;
              global_j += 1;
              get_elses_matrix_value(&global_i, &global_j, &value);
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
    int64_t temp_bases_offset = (i / MPISIZE) * opts.nleaf * opts.max_rank;

    std::cout << "i : " << i << std::endl;

    MPI_Allgather(&A.U(i, A.max_level),
                  opts.nleaf * opts.max_rank,
                  MPI_DOUBLE,
                  temp_bases.data() + temp_bases_offset,
                  opts.nleaf * opts.max_rank,
                  MPI_DOUBLE,
                  MPI_COMM_WORLD);
  }

  for (int64_t i = MPIRANK; i < nblocks; i += MPISIZE) {
    int64_t temp_bases_offset = (i / MPISIZE) * opts.nleaf * opts.max_rank;
    Matrix Ui(opts.nleaf, opts.max_rank);

#pragma omp parallel for collapse(2)
    for (int64_t U_i = 0; U_i < opts.nleaf; ++U_i) {
      for (int64_t U_j = 0; U_j < opts.max_rank; ++U_j) {
        Ui(U_i, U_j) = temp_bases[temp_bases_offset + U_i + U_j * opts.nleaf];
      }
    }

    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
      }
    }
  }



  // // Calculate the skeleton blocks.
  // for (int i = 0; i < nblocks; ++i) {
  //   std::vector<double> Utemp_buffer(opts.nleaf * opts.max_rank * MPISIZE);

  //   int p_blocks = nblocks / MPISIZE;

  //   for (int p_block = 0; p_block < p_blocks; ++p_block) {
  //     int j = MPISIZE * p_block + MYROW;

  //     // send, recv
  //     MPI_Gather(&A.U(j, A.max_level),
  //                A.U(j, A.max_level).numel(),
  //                MPI_DOUBLE,
  //                Utemp_buffer.data(),
  //                opts.nleaf * opts.max_rank,
  //                MPI_DOUBLE,
  //                mpi_rank(i),
  //                MPI_COMM_WORLD);

  //     if (mpi_rank(i) == MPIRANK) {
  //       for (int j = 0; j < MPISIZE; ++j) {
  //         int real_j = p_block * MPISIZE + j;
  //         if (A.is_admissible.exists(i, j, A.max_level) && A.is_admissible(i, j, A.max_level)) {
  //           Matrix temp = matmul(A.U(i, A.max_level),
  //                                generate_p2p_interactions(i, real_j, A.max_level, opts,
  //                                                          domain, A),
  //                                true, false);

  //           Matrix Uj(opts.nleaf, opts.max_rank);
  //           array_copy(&Utemp_buffer[(opts.nleaf * opts.max_rank) * j], &Uj, Uj.numel());

  //           auto S_block = matmul(temp, Uj);

  //           A.S.insert(i, real_j, A.max_level, std::move(S_block));
  //         }
  //       }
  //     }
  //   }

  //   int leftover_nprocs = nblocks - p_blocks * MPISIZE;

  //   // Leftover block group communication. Second condition is needed to limit
  //   // the root process calculation to within the leftover processes?
  //   if (p_blocks * MPISIZE < nblocks && mpi_rank(i) < leftover_nprocs) {
  //     int j = MPISIZE * p_blocks + MYROW;

  //     MPI_Comm MPI_ROW_I_PROCESSES;
  //     int color = j < nblocks ? 1 : 0;
  //     int key = MYROW;

  //     MPI_Comm_split(MPI_COMM_WORLD,
  //                    color, key,
  //                    &MPI_ROW_I_PROCESSES);

  //     if (j < nblocks && mpi_rank(i) < leftover_nprocs) {
  //       int root_proc = translate_rank_comm_world(leftover_nprocs, mpi_rank(i), MPI_ROW_I_PROCESSES);
  //       MPI_Gather(&A.U(j, A.max_level),
  //                  A.U(j, A.max_level).numel(),
  //                  MPI_DOUBLE,
  //                  Utemp_buffer.data(),
  //                  opts.nleaf * opts.max_rank,
  //                  MPI_DOUBLE,
  //                  root_proc,
  //                  MPI_ROW_I_PROCESSES);

  //       if (mpi_rank(i) == MPIRANK) {
  //         for (int j = 0; j < leftover_nprocs; ++j) {
  //           int real_j = p_blocks * MPISIZE + j;
  //           if (i != real_j) {
  //             Matrix temp = matmul(A.U(i, A.max_level),
  //                                  generate_p2p_interactions(i, real_j, A.max_level, opts,
  //                                                            domain, A),
  //                                  true, false);

  //             Matrix Uj(opts.nleaf, opts.max_rank);
  //             array_copy(&Utemp_buffer[(opts.nleaf * opts.max_rank) * j], &Uj, Uj.numel());

  //             auto S_block = matmul(temp, Uj);

  //             A.S.insert(i, real_j, A.max_level, std::move(S_block));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
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

  // Multiply the bases from the remaining levels.
  // for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
  //   for (int64_t i = 0; i < nblocks; ++i) {
  //     int64_t index_i = i / MPISIZE;
  //     Matrix x_hat_i(opts.max_rank, 1);
  //     if (mpi_rank(i) == MPIRANK) {
  //       x_hat_i = x_hat[index_i];
  //     }

  //     MPI_Bcast();
  //   }
  // }

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

    for (int64_t i = MPIRANK; i < nblocks; i += MPIGRID[0]) {
      int64_t index_i = i / MPIGRID[0];
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        // std::cout << "i: " << i << " j: " << j << std::endl;
        // const Matrix& Sij = A.S(i, j, A.max_level);
        // matmul(Sij, x_hat_j, b_hat[index_i]);
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
      int j_index = j / MPIGRID[0];
      xj = x[j_index];
    }

    MPI_Bcast(&xj, xj.numel(), MPI_DOUBLE, mpi_rank(j), MPI_COMM_WORLD);
    for (int64_t i = MPIRANK; i < nblocks; i += MPIGRID[0]) {
      Matrix Aij = generate_p2p_interactions(i, j, A.max_level, opts,
                                             domain, A);
      int i_index = i / MPIGRID[0];
      matmul(Aij, xj, b[i_index]); //  b = 1 * b + 1 * A * x
    }
  }
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  int N = opts.N;

  assert(opts.N % opts.nleaf == 0);

  {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  MPIGRID[0] = MPISIZE; MPIGRID[1] = 1;

  Cblacs_get(-1, 0, &BLACS_CONTEXT );
  Cblacs_gridinit(&BLACS_CONTEXT, "Row", MPIGRID[0], MPIGRID[1]);
  Cblacs_pcoord(BLACS_CONTEXT, MPIRANK, &MYROW, &MYCOL);

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
    abort();
    const int64_t num_electrons_per_atom = 4;
    const int64_t num_atoms_per_molecule = 60;
    init_elses_state();
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

  auto start_construct = std::chrono::system_clock::now();

  int64_t construct_max_rank;

  A.num_blocks.resize(A.max_level+1);
  A.num_blocks[A.max_level] = opts.N/opts.nleaf;

  if (opts.admis_kind == GEOMETRY) {
    init_geometry_admis(A, domain, opts); // init admissiblity conditions with DTT
  }
  else if (opts.admis_kind == DIAGONAL) {
    init_diagonal_admis(A, domain, opts); // init admissiblity conditions with diagonal condition.
  }
  A.print_Csp(A.max_level);
  A.print_structure();

  construct_H2_matrix(A, domain, opts);

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
      double value = dist(gen);

      int l_row = indxg2l(g_row, opts.nleaf, MPIGRID[0]) - 1;
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

  Cblacs_gridexit(BLACS_CONTEXT);
  Cblacs_exit(1);
  MPI_Finalize();

  if (!MPIRANK) {
    std::cout << "Everything finished.\n";
  }

  return 0;
}
