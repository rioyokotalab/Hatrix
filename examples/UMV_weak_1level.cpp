#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <chrono>

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix { namespace UMV {
  class Vector {
  private:
    void copy_from_vector(const Hatrix::Matrix& vector) {
      int c_size = block_size - rank;

      for (int block = 0; block < nblocks; ++block) {
        Hatrix::Matrix c_vector(c_size, 1);
        Hatrix::Matrix o_vector(rank, 1);

        // copy c vector
        for (int i = 0; i < c_size; ++i) {
          c_vector(i, 0) = vector(block * block_size + i, 0);
        }

        // copy rank vector
        for (int i = 0; i < rank; ++i) {
          o_vector(i, 0) = vector(block * block_size + c_size + i, 0);
        }

        c.insert(block, std::move(c_vector));
        o.insert(block, std::move(o_vector));
      }
    }

  public:
    // Maps of the vector blocks. bc for upper part of the vector
    // and bo for lower part.
    int N, block_size, nblocks, rank;
    RowMap c, o;

    Vector(const Hatrix::Matrix& v, int _N, int _block_size, int _nblocks, int _rank) :
      N(_N), block_size(_block_size), nblocks(_nblocks), rank(_rank) {
      assert(v.cols == 1);
      copy_from_vector(v);
    }

    Vector(const Vector& V) : N(V.N), block_size(V.block_size),
                        nblocks(V.nblocks), rank(V.rank),
                        c(V.c), o(V.o) {}

    Vector(std::function<Matrix(int64_t, int64_t)> gen_fn, int _N, int _block_size, int _nblocks, int _rank) :
      N(_N), block_size(_block_size), nblocks(_nblocks), rank(_rank) {
      Hatrix::Matrix vector = gen_fn(N, 1);
      copy_from_vector(vector);
    }

    void print() {
      for (int block = 0; block < nblocks; ++block) {
        c[block].print();
        o[block].print();
      }
    }
  };

  class BLR2 {
  private:
    Matrix compose_dense(int i, int j) {
      Matrix dense(block_size, block_size);
      int c_size = block_size - rank;

      for (int irow = 0; irow < c_size; ++irow) {
        for (int jcol = 0; jcol < c_size; ++jcol) {
          dense(irow, jcol) = Dcc(i, j)(irow, jcol);
        }
      }

      for (int irow = 0; irow < rank; ++irow) {
        for (int jcol = 0; jcol < c_size; ++jcol) {
          dense(irow + c_size, jcol) = Doc(i, j)(irow, jcol);
        }
      }

      for (int irow = 0; irow < c_size; ++irow) {
        for (int jcol = 0; jcol < rank; ++jcol) {
          dense(irow, jcol + c_size) = Dco(i, j)(irow, jcol);
        }
      }

      for (int irow = 0; irow < rank; ++irow) {
        for (int jcol = 0; jcol < rank; ++jcol) {
          dense(irow + c_size, jcol + c_size) = Doo(i, j)(irow, jcol);
        }
      }

      return dense;
    }

  public:
    RowColMap Dcc, Dco, Doc, Doo;
    RowColMap S;
    RowMap U, Uc;
    ColMap V, Vc;
    int N, block_size, n_blocks, rank, admis;
    double construct_error;

    Hatrix::Matrix D(int row, int col) {
      return compose_dense(row, col);
    }

    void insert_D(int row, int col, Hatrix::Matrix& mat) {
      int c_size = block_size - rank;

      for (int i = 0; i < c_size; ++i) {
        for (int j = 0; j < c_size; ++j) {
          Dcc(row, col)(i, j) = mat(i, j);
        }
      }

      for (int i = 0; i < c_size; ++i) {
        for (int j = 0; j < rank; ++j) {
          Dco(row, col)(i, j) = mat(i, j + c_size);
        }
      }

      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < c_size; ++j) {
          Doc(row, col)(i, j) = mat(i + c_size, j);
        }
      }

      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          Doo(row, col)(i, j) = mat(i + c_size, j + c_size);
        }
      }
    }

    Hatrix::Matrix U_F(int row) {
      Hatrix::Matrix U_F(block_size, block_size);
      int c_size = block_size - rank;

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < c_size; ++j) {
          U_F(i, j) = Uc[row](i, j);
        }
      }

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < rank; ++j) {
          U_F(i, c_size + j) = U[row](i, j);
        }
      }

      return U_F;
    }

    Hatrix::Matrix V_F(int col) {
      Hatrix::Matrix V_F(block_size, block_size);
      int c_size = block_size - rank;
      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < c_size; ++j) {
          V_F(i, j) = Vc[col](i, j);
        }
      }

      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < rank; ++j) {
          V_F(i, c_size + j) = V[col](i, j);
        }
      }
      return V_F;
    }

    BLR2(randvec_t& randpts, int _N, int _block_size, int _n_blocks, int _rank, int _admis) :
      N(_N), block_size(_block_size), n_blocks(_n_blocks), rank(_rank), admis(_admis) {
      int c_size = block_size - rank;

      // Populate dense blocks.
      for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_blocks; ++j) {
          Dcc.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts, c_size, c_size,
                                                       i * block_size, j * block_size));
          Dco.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts, c_size, rank,
                                                       i * block_size,
                                                       j * block_size + c_size));

          Doc.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts, rank, c_size,
                                                       i * block_size + c_size,
                                                       j * block_size));
          Doo.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts, rank, rank,
                                                       i * block_size + c_size,
                                                       j * block_size + c_size));
        }
      }

      // Expected errors to check against later.
      std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;
      int64_t oversampling = 5;
      double error;
      std::vector<Hatrix::Matrix> Y;
      for (int64_t i = 0; i < n_blocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(block_size, rank + oversampling));
      }

      for (int64_t i = 0; i < n_blocks; ++i) {
        Hatrix::Matrix Ui, Si, Vi;
        Hatrix::Matrix AY(block_size, rank + oversampling);
        for (int64_t j = 0; j < n_blocks; ++j) {
          if (i == j) continue;
          Hatrix::matmul(compose_dense(i, j), Y[j], AY);
        }
        std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, std::move(Ui));
      }

      for (int64_t j = 0; j < n_blocks; ++j) {
        Hatrix::Matrix Ui, Si, Vi;
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        for (int64_t i = 0; i < n_blocks; ++i) {
          if (j == i) continue;
          Hatrix::matmul(Y[i], compose_dense(i, j), YtA, true);
        }
        std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, std::move(Vi.transpose()));
      }

      for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_blocks; ++j) {
          if (i != j) {
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U[i], compose_dense(i, j), true),
                                    V[j], false, false));
          }
        }
      }

      double diff = 0, norm = 0, fnorm, fdiff;
      for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j < n_blocks; ++j) {
          fnorm = Hatrix::norm(compose_dense(i, j));
          norm += fnorm * fnorm;
          if (i == j)
            continue;
          else {
            fdiff = Hatrix::norm(U[i] * S(i, j) * V[j].transpose() - compose_dense(i, j));
            diff += fdiff * fdiff;
          }
        }
      }
      construct_error = std::sqrt(diff/norm);
    };
  };

  Hatrix::Matrix full_qr(Hatrix::Matrix& A) {
    Hatrix::Matrix Q(A.rows, A.rows);
    std::vector<double> tau(std::max(A.rows, A.cols));
    for (int i = 0; i < Q.rows; ++i) {
      for (int j = 0; j < A.cols; ++j) {
        Q(i, j) = A(i, j);
      }
    }
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q.rows, A.cols, &Q, Q.stride, tau.data());
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.rows, Q.cols, &Q,
                   Q.stride, tau.data());
    return Q;
  }


  Hatrix::Matrix make_complement(const Hatrix::Matrix& Q) {
    Hatrix::Matrix Q_copy(Q);
    Hatrix::Matrix Q_F(Q.rows, Q.rows - Q.cols);
    Hatrix::Matrix Q_full = full_qr(Q_copy);

    for (int i = 0; i < Q_F.rows; ++i) {
      for (int j = 0; j < Q_F.cols; ++j) {
        Q_F(i, j) = Q_full(i, j + Q.cols);
      }
    }
    return Q_F;
  }

  void left_and_right_multiply_dense(BLR2& A, int block) {
    Hatrix::Matrix U_F = A.U_F(block);
    Hatrix::Matrix V_F = A.V_F(block);

    Hatrix::Matrix D = Hatrix::matmul(
                                      Hatrix::matmul(
                                                     U_F,
                                                     A.D(block, block),
                                                     true,
                                                     false),
                                      V_F);
    A.insert_D(block, block, D);
  }

  void partial_lu(BLR2& A, int block) {
    if (A.rank != A.block_size) {
      Hatrix::lu(A.Dcc(block, block));
      Hatrix::solve_triangular(A.Dcc(block, block), A.Dco(block, block), Hatrix::Left,
                               Hatrix::Lower, true, false, 1.0);
      Hatrix::solve_triangular(A.Dcc(block, block), A.Doc(block, block), Hatrix::Right,
                               Hatrix::Upper, false, false, 1.0);
      Hatrix::matmul(A.Doc(block, block), A.Dco(block, block), A.Doo(block, block),
                     false, false, -1.0, 1.0);
    }
  }

  Hatrix::Matrix factorize(BLR2& A) {
    for (int block = 0; block < A.n_blocks; ++block) {
      A.Uc.insert(block, std::move(make_complement(A.U[block])));
      A.Vc.insert(block, std::move(make_complement(A.V[block])));
      left_and_right_multiply_dense(A, block);
            Hatrix::Matrix c(A.D(block, block));
      partial_lu(A, block);
    }

    Hatrix::Matrix last(A.rank * A.n_blocks, A.rank * A.n_blocks);

    for (int i = 0; i < A.n_blocks; ++i) {
      for (int j = 0; j < A.n_blocks; ++j) {
        if (i == j) {
          for (int irow = 0; irow < A.rank; ++irow) {
            for (int jcol = 0; jcol < A.rank; ++jcol) {
              last(i * A.rank + irow, j * A.rank + jcol)  = A.Doo(i, j)(irow, jcol);
            }
          }
        }
        else {
          for (int irow = 0; irow < A.rank; ++irow) {
            for (int jcol = 0; jcol < A.rank; ++jcol) {
              last(i * A.rank + irow, j * A.rank + jcol) = A.S(i, j)(irow, jcol);
            }
          }
        }
      }
    }
    Hatrix::lu(last);

    return last;
  }

  void matrix_vector_multiply(Hatrix::Matrix& A, Hatrix::UMV::Vector& x, int block, bool transpose) {
    Hatrix::Matrix temp(A.rows, 1);
    int c_size = x.block_size - x.rank;

    for (int i = 0; i < c_size; ++i) {
      temp(i, 0) = x.c[block](i, 0);
    }
    for (int i = 0; i < x.rank; ++i) {
      temp(i + c_size, 0) = x.o[block](i, 0);
    }

    Hatrix::Matrix product = Hatrix::matmul(A, temp, transpose);

    for (int i = 0; i < c_size; ++i) {
      x.c[block](i, 0) = product(i, 0);
    }
    for (int i = 0; i < x.rank; ++i) {
      x.o[block](i, 0) = product(i + c_size, 0);
    }
  }

  Hatrix::Matrix gather_o_vectors(const Vector& x) {
    Hatrix::Matrix o_vectors(x.rank * x.nblocks, 1);
    for (int block = 0; block < x.nblocks; ++block) {
      for (int i = 0; i < x.rank; ++i) {
        o_vectors(block * x.rank + i, 0) = x.o[block](i, 0);
      }
    }

    return o_vectors;
  }

  void scatter_o_vectors(Vector& x, Hatrix::Matrix& o_vectors) {
    for (int block = 0; block < x.nblocks; ++block) {
      for (int i = 0; i < x.rank; ++i) {
        x.o[block](i, 0) = o_vectors(block * x.rank + i, 0);
      }
    }
  }

  Hatrix::UMV::Vector substitute(BLR2& A, Hatrix::Matrix& last, const Vector& b) {
    Hatrix::UMV::Vector x(b);

    // Forward substitute.
    for (int block = 0; block < A.n_blocks; ++block) {
      Hatrix::Matrix U_F = A.U_F(block);
      matrix_vector_multiply(U_F, x, block, true);

      if (A.rank != A.block_size) {
        Hatrix::solve_triangular(A.Dcc(block, block), x.c[block], Hatrix::Left,
                                 Hatrix::Lower, true);
        Hatrix::matmul(A.Doc(block, block), x.c[block], x.o[block], false, false,
                       -1.0, 1.0);
      }
    }
    Hatrix::Matrix o_vectors = gather_o_vectors(x);
    Hatrix::solve_triangular(last, o_vectors, Hatrix::Left, Hatrix::Lower, true);

    // Backward substitute
    Hatrix::solve_triangular(last, o_vectors, Hatrix::Left, Hatrix::Upper, false);
    scatter_o_vectors(x, o_vectors);

    for (int block = 0; block < A.n_blocks; ++block) {
      if (A.rank != A.block_size) {
        Hatrix::matmul(A.Dco(block, block), x.o[block], x.c[block], false,
                       false, -1.0, 1.0);
        Hatrix::solve_triangular(A.Dcc(block, block), x.c[block], Hatrix::Left,
                                 Hatrix::Upper, false);
      }

      Hatrix::Matrix V_F = A.V_F(block);
      matrix_vector_multiply(V_F, x, block, false);
    }

    return x;
  }
} // namespace UMV

  double norm(Hatrix::UMV::Vector& x) {
    double norm = 0;
    for (int b = 0; b < x.nblocks; ++b) {
      double c_norm = Hatrix::norm(x.c[b]);
      double o_norm = Hatrix::norm(x.o[b]);

      norm += c_norm * c_norm + o_norm * o_norm;
    }

    return std::sqrt(norm);
  }

} // namespace Hatrix

double rel_error(const double A_norm, const double B_norm) {
  double diff = A_norm - B_norm;
  return std::sqrt((diff * diff) / (B_norm * B_norm));
}

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int block_size = atoi(argv[3]);
  const char * fname = argv[4];
  int nblocks = N / block_size;

  if (rank > block_size || N % block_size != 0) {
    exit(1);
  }

  std::ofstream file;
  file.open(fname, std::ios::app | std::ios::out);

  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::Context::init();
  const Hatrix::Matrix _b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::UMV::Vector b(_b, N, block_size, nblocks, rank);


  auto start_construct = std::chrono::system_clock::now();
  Hatrix::UMV::BLR2 A(randpts, N, block_size, nblocks, rank, 0);
  auto stop_construct = std::chrono::system_clock::now();
  double construct_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_construct - start_construct).count();
  double construct_error = A.construct_error;

  auto start_factorize = std::chrono::system_clock::now();
  Hatrix::Matrix last = Hatrix::UMV::factorize(A);
  auto stop_factorize = std::chrono::system_clock::now();
  double factorize_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_factorize - start_factorize).count();

  auto start_subs = std::chrono::system_clock::now();
  Hatrix::UMV::Vector x = Hatrix::UMV::substitute(A, last, b);
  auto stop_subs = std::chrono::system_clock::now();
  double subs_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_subs - start_subs).count();

  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_dense = Hatrix::lu_solve(A_dense, _b);

  double substitute_error = rel_error(Hatrix::norm(x), Hatrix::norm(x_dense));
  std::cout << "err: " << substitute_error << std::endl;

  file << N << "," << rank << "," << block_size << "," << substitute_error << ","
       << construct_error << "," << construct_time << "," << factorize_time << ","
       << subs_time << std::endl;

  file.close();

  Hatrix::Context::finalize();

  return 0;
}
