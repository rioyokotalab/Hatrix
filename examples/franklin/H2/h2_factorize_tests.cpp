#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_operations.hpp"

using namespace Hatrix;

static void
make_dense(SymmetricSharedBasisMatrix& A, const int64_t level) {
  int64_t nblocks = pow(2, level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, level)) {
        if (A.is_admissible(i, j, level)) {
          A.D.insert(i, j, level,
                        matmul(matmul(A.U(i, level), A.S(i, j, level)),
                               A.U(j, level), false, true));
          A.is_admissible.erase(i, j, level);
          A.is_admissible.insert(i, j, level, false);
        }
        else {
          Matrix d = A.D(i, j, level);
          A.D.insert(i, j, level, std::move(d));
        }
      }

      // if (exists_and_inadmissible(A, i, j, level)) {
      //   std::cout << "norm<" <<  i << " " << j << " " << level <<  "> -> " << norm(A.D(i, j, level)) << std::endl;
      // }

      if (i == j) {             // make strict upper triangle zero.
        Matrix& d = A.D(i, j, level);
        for (int ii = 0; ii < d.rows; ++ii) {
          for (int jj = ii+1; jj < d.cols; ++jj) {
            d(ii, jj) = 0;
          }
        }
      }
    }
  }
}

static SymmetricSharedBasisMatrix
compute_product(SymmetricSharedBasisMatrix& A, int64_t level) {
  SymmetricSharedBasisMatrix actual(A);
  int64_t nblocks = pow(2, level);
  // init to zero.
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (exists_and_inadmissible(actual, i, j, level)) {
        Matrix& D = actual.D(i, j, level);
        for (int ii = 0; ii < D.rows; ++ii) {
          for (int jj = 0; jj < D.cols; ++jj) {
            D(ii, jj) = 0;
          }
        }
      }
    }
  }

  // compute cc blocks
  // cc = cc * cc.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k <= j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                        actualij.rows - actual.ranks(i, level),
                                        actualij.cols - actual.ranks(j, level));

          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          matmul(Dik_splits[0], Djk_splits[0], actualij_splits[0], false, true, 1, 1);
        }
      }
    }
  }

  // cc = co * co.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k < j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                        actualij.rows - actual.ranks(i, level),
                                        actualij.cols - actual.ranks(j, level));
          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          std::cout << "p -> i: " << i << " j: " << j
                    << " a -> i: " << i << " k: " << k
                    << " b -> j: " << j << " k: " << k << std::endl;

          matmul(Dik_splits[1], Djk_splits[1], actualij_splits[0], false, true, 1, 1);
        }
      }
    }
  }

  // oc blocks
  // oc = oc + oc * cc.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k <= j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                             actualij.rows - actual.ranks(i, level),
                                             actualij.cols - actual.ranks(j, level));
          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          matmul(Dik_splits[2], Djk_splits[0], actualij_splits[2], false, true, 1, 1);
        }
      }
    }
  }

  // oc = oc + oo * co.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k <= j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                             actualij.rows - actual.ranks(i, level),
                                             actualij.cols - actual.ranks(j, level));
          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          matmul(Dik_splits[3], Djk_splits[1], actualij_splits[2], false, true, 1, 1);
        }
      }
    }
  }

  // oo blocks
  // oo = oo + oc * oc.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k <= j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                             actualij.rows - actual.ranks(i, level),
                                             actualij.cols - actual.ranks(j, level));
          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          matmul(Dik_splits[2], Djk_splits[2], actualij_splits[3], false, true, 1, 1);
        }
      }
    }
  }

  // oo = oo + oo * oo.T
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      for (int k = 0; k <= j; ++k) {
        if (exists_and_inadmissible(actual, i, j, level) &&
            exists_and_inadmissible(A, i, k, level) &&
            exists_and_inadmissible(A, j, k, level)) {
          Matrix& actualij = actual.D(i, j, level);
          Matrix& Dik = A.D(i, k, level);
          Matrix& Djk = A.D(j, k, level);

          auto actualij_splits = split_dense(actualij,
                                             actualij.rows - actual.ranks(i, level),
                                             actualij.cols - actual.ranks(j, level));
          auto Dik_splits = split_dense(Dik,
                                        Dik.rows - A.ranks(i, level),
                                        Dik.cols - A.ranks(k, level));
          auto Djk_splits = split_dense(Djk,
                                        Djk.rows - A.ranks(j, level),
                                        Djk.cols - A.ranks(k, level));

          matmul(Dik_splits[3], Djk_splits[3], actualij_splits[3], false, true, 1, 1);
        }
      }
    }
  }


  return actual;
}

static double
check_error(SymmetricSharedBasisMatrix& actual, SymmetricSharedBasisMatrix& expected, int64_t level) {
  int64_t nblocks = pow(2, level);
  double actual_norm = 0, expected_norm = 0;

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (exists_and_inadmissible(actual, i, j, level)) {
        Matrix& actual_ij = actual.D(i, j, level);
        auto actual_ij_splits = split_dense(actual_ij,
                                            actual_ij.rows - actual.ranks(i, level),
                                            actual_ij.cols - actual.ranks(j, level));

        Matrix& expected_ij = expected.D(i, j, level);
        auto expected_ij_splits = split_dense(expected_ij,
                                            expected_ij.rows - expected.ranks(i, level),
                                            expected_ij.cols - expected.ranks(j, level));


        // cc
        // std::cout << "i: " << i << " j: " << j << std::endl;
        // (actual_ij_splits[0] - expected_ij_splits[0]).print();
        actual_norm += pow(norm(actual_ij_splits[0]), 2);
        expected_norm += pow(norm(expected_ij_splits[0]), 2);

        // oc
        // actual_norm += pow(norm(actual_ij_splits[2]), 2);
        // expected_norm += pow(norm(expected_ij_splits[2]), 2);

        // oo
        // actual_norm += pow(norm(actual_ij_splits[3]), 2);
        // expected_norm += pow(norm(expected_ij_splits[3]), 2);
      }

      actual_norm = sqrt(actual_norm);
      expected_norm = sqrt(expected_norm);
      double err = abs(actual_norm - expected_norm) / expected_norm;

      std::cout << "i : " << i << " j: " << j <<  " level: "
                << level << " err: " << err  << std::endl;
    }
  }

  actual_norm = sqrt(actual_norm);
  expected_norm = sqrt(expected_norm);

  return abs(actual_norm - expected_norm) / expected_norm;
}

// compute the full factorization of the trailing blocks. Include the S blocks.
static void
compute_trailing_cholesky(SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    Matrix& D = A.D(i, i, level);
    auto D_splits = split_dense(D,
                                D.rows - A.ranks(i, level),
                                D.cols - A.ranks(i, level));

    cholesky(D_splits[3], Hatrix::Lower);

    for (int64_t j = i+1; j < nblocks; ++j) {
      if (A.is_admissible.exists(j, i, level)) {
        if (!A.is_admissible(j, i, level)) {
          Matrix& D_ij = A.D(j, i, level);
          auto D_ij_splits = split_dense(D_ij,
                                         D_ij.rows - A.ranks(j, level),
                                         D_ij.cols - A.ranks(i, level));
          solve_triangular(D_splits[3], D_ij_splits[3], Hatrix::Right, Hatrix::Lower,
                           false, true, 1.0);
        }
        else {
          std::cout << "i: " << i << " j: " << j << " l: " << level << std::endl;
          abort();
        }
      }
    }

    for (int64_t j = i+1; j < nblocks; ++j) {
      for (int64_t k = i+1; k <= j; ++k) {
        if (exists_and_inadmissible(A, k, i, level) &&
            exists_and_inadmissible(A, j, i, level) &&
            exists_and_inadmissible(A, j, k, level)) {

          Matrix& D_jk = A.D(j, k, level);
          auto D_jk_splits = split_dense(D_jk,
                                         D_jk.rows - A.ranks(j, level),
                                         D_jk.cols - A.ranks(k, level));
          Matrix& D_ki = A.D(k, i, level);
          auto D_ki_splits = split_dense(D_ki,
                                         D_ki.rows - A.ranks(k, level),
                                         D_ki.cols - A.ranks(i, level));

          Matrix& D_ji = A.D(j, i, level);
          auto D_ji_splits = split_dense(D_ji,
                                         D_ji.rows - A.ranks(j, level),
                                         D_ji.cols - A.ranks(i, level));

          if (j == k) {
            syrk(D_ji_splits[3], D_jk_splits[3], Hatrix::Lower, false, -1, 1);
          }
          else {
            matmul(D_ji_splits[3], D_ki_splits[3], D_jk_splits[3], false, true, -1, 1);
          }
        }
      }
    }
  }
}

void
enforce_lower_triangle(SymmetricSharedBasisMatrix& A, int64_t level) {
  int nblocks = pow(2, level);

  for (int i = 0; i < nblocks; ++i) {
    Matrix& d = A.D(i, i, level);

    for (int ii = 0; ii < d.rows; ++ii) {
      for (int jj = ii+1; jj < d.cols; ++jj) {
        d(ii, jj) = 0;
      }
    }
  }
}

// Test function for partial factorization of each level. The matrix is first made dense
// and then partially factorized.
SymmetricSharedBasisMatrix
dense_cholesky_test(const SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts) {
  SymmetricSharedBasisMatrix A_test(A);
  SymmetricSharedBasisMatrix expected(A_test);

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);

    make_dense(A_test, level);
    expected.D.deep_copy(A_test.D);

    for (int64_t block = 0; block < nblocks; ++block) {
      factorize_diagonal(A_test, block, level);
      triangle_reduction(A_test, block, level);
      compute_schurs_complement(A_test, block, level);
    }

    compute_trailing_cholesky(A_test, level);
    enforce_lower_triangle(A_test, level);

    auto actual = compute_product(A_test, level);

    enforce_lower_triangle(actual, level);
    enforce_lower_triangle(expected, level);
    double rel_error = check_error(actual, expected, level);

    std::cout << "level: " << rel_error << std::endl;

    merge_unfactorized_blocks(A_test, level);
  }

  return A_test;
}

void
vector_permute_test(const Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Matrix& x) {
  Hatrix::Matrix x_copy(x);
  int64_t level, level_offset = 0;

  for (level = A.max_level; level >= A.min_level; --level) {
    level_offset = permute_forward(A, x_copy, level, level_offset);
  }

  ++level;
  for (; level <= A.max_level; ++level) {
    level_offset = permute_backward(A, x_copy, level, level_offset);
  }

  Hatrix::Matrix diff = x_copy - x;

  std::cout << "vector permutation diff norm: " << norm(diff) << std::endl;
}

Hatrix::Matrix
solve_dense_test(const Hatrix::SymmetricSharedBasisMatrix& A,
                 const Hatrix::Matrix& b, const Hatrix::Args& opts) {
  int64_t level, level_offset = 0;
  std::vector<Matrix> x_splits;
  Hatrix::Matrix x(b);

  for (level = A.max_level; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int64_t n = 0;              // total vector length due to variable ranks.
    for (int64_t i = 0; i < nblocks; ++i) { n += A.D(i, i, level).rows; }

    Matrix x_level(n, 1);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(level_offset + i, 0);
    }

    solve_forward_level(A, x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(level_offset + i, 0) = x_level(i, 0);
    }

    level_offset = permute_forward(A, x, level, level_offset);
  }

  x_splits = x.split(std::vector<int64_t>(1, level_offset), {});
  Matrix x_last(x_splits[1]);

  int64_t last_nodes = pow(2, level);
  std::vector<int64_t> vector_splits;
  int64_t nrows = 0;
  for (int64_t i = 0; i < last_nodes; ++i) {
    vector_splits.push_back(nrows + A.D(i, i, level).rows);
    nrows += A.D(i, i, level).rows;
  }
  auto x_last_splits = x_last.split(vector_splits, {});

  // forward for the last blocks
  for (int i = 0; i < last_nodes; ++i) {
    for (int j = 0; j < i; ++j) {
      matmul(A.D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
    }
    solve_triangular(A.D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Lower,
                     false, false, 1.0);
  }

  // backward for the last blocks.
  for (int j = last_nodes-1; j >= 0; --j) {
    for (int i = last_nodes-1; i > j; --i) {
      matmul(A.D(i, j, level), x_last_splits[i], x_last_splits[j], true, false, -1.0, 1.0);
    }
    solve_triangular(A.D(j, j, level), x_last_splits[j], Hatrix::Left, Hatrix::Lower,
                     false, true, 1.0);
  }

  x_splits[1] = x_last;
  ++level;

  // backward substitution.
  for (; level <= A.max_level; ++level) {
    int64_t nblocks = pow(2, level);

    int64_t n = 0;
    for (int64_t i = 0; i < nblocks; ++i) { n += A.D(i, i, level).cols; }
    Matrix x_level(n, 1);

    level_offset = permute_backward(A, x, level, level_offset);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(level_offset + i, 0);
    }

    solve_backward_level(A, x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(level_offset + i, 0) = x_level(i, 0);
    }
  }

  return x;
}

void
solve_dense_test_level(const Hatrix::SymmetricSharedBasisMatrix& A,
                       const Hatrix::Matrix& x, int64_t level,
                       const Hatrix::Args& opts) {

}

void
dense_factorize_and_solve_test(const Hatrix::SymmetricSharedBasisMatrix& A,
                               const Hatrix::Matrix& b, const Hatrix::Args& opts) {
  Hatrix::Matrix x(b);
  int64_t level_offset = 0;
  SymmetricSharedBasisMatrix A_copy(A);

  for (int64_t level = A.max_level; level >= A_copy.min_level; --level) {
    int64_t nblocks = pow(2, level);
    make_dense(A_copy, level);

    for (int64_t block = 0; block < nblocks; ++block) {
      multiply_complements(A_copy, block, level);
      factorize_diagonal(A_copy, block, level);
      triangle_reduction(A_copy, block, level);
      compute_schurs_complement(A_copy, block, level);
    }

    compute_trailing_cholesky(A_copy, level);
    enforce_lower_triangle(A_copy, level);

    // solve_dense_test_level(A_copy, x_level, level, opts);

    merge_unfactorized_blocks(A_copy, level);
  }

  // std::cout << "dense solve error: " << norm(product - b) / norm(b) << std::endl;
}

void
check_fill_ins(Hatrix::SymmetricSharedBasisMatrix& A_pre,
               Hatrix::SymmetricSharedBasisMatrix& A_post,
               int64_t block,
               int64_t level,
               RowColLevelMap<Matrix>& F) {
  int64_t nblocks = pow(2, level);
  // Check row for fill-ins
  for (int i = block; i < nblocks; ++i) {
    if (A_pre.is_admissible.exists(i, block, level) &&
        A_pre.is_admissible(i, block, level)) {
      Matrix actual = matmul(matmul(A_pre.U(i, level), A_pre.S(i, block, level)),
                          A_pre.U(block, level), false, true);
      Matrix expected = matmul(matmul(A_post.U(i, level), A_post.S(i, block, level)),
                          A_post.U(block, level), false, true);

      if (F.exists(i, block, level)) {
        actual -= F(i, block, level);
      }

      std::cout << "col fill in norm: " << norm(actual - expected) / norm(expected) << std::endl;
    }
  }

  for (int j = 0; j <= block; ++j) {
    if (A_pre.is_admissible.exists(block, j, level) &&
        A_pre.is_admissible(block, j, level)) {
      Matrix actual = matmul(matmul(A_pre.U(block, level), A_pre.S(block, j, level)),
                             A_pre.U(j, level), false, true);
      Matrix expected = matmul(matmul(A_post.U(block, level), A_post.S(block, j, level)),
                               A_post.U(j, level), false, true);

      if (F.exists(block, j, level)) {
        actual -= F(block, j, level);
      }

      std::cout << "row fill in norm: " << norm(actual - expected) / norm(expected) << std::endl;
    }
  }
}

void
cholesky_fill_in_recompress_check(const Hatrix::SymmetricSharedBasisMatrix& A,
                                  const Hatrix::Args& opts) {
  SymmetricSharedBasisMatrix A_copy(A);
  Hatrix::RowColLevelMap<Matrix> F;
  Hatrix::RowMap<Matrix> r, t;

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);

    for (int block = 0; block < nblocks; ++block) {
      SymmetricSharedBasisMatrix A_pre(A_copy);

      update_row_cluster_basis_and_S_blocks(A_copy, F, r, opts, block, level);
      update_col_cluster_basis_and_S_blocks(A_copy, F, t, opts, block, level);

      multiply_complements(A_copy, block, level);
      factorize_diagonal(A_copy, block, level);
      triangle_reduction(A_copy, block, level);
      compute_fill_ins(A_copy, block, level, F);

      check_fill_ins(A_pre, A_copy, block, level, F);

      merge_unfactorized_blocks(A_copy, level);
    }
  }
}
