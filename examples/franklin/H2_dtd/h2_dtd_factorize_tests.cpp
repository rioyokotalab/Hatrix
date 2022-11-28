#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_dtd_operations.hpp"

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
        actual_norm += pow(norm(actual_ij_splits[0]), 2);
        expected_norm += pow(norm(expected_ij_splits[0]), 2);

        // oc
        actual_norm += pow(norm(actual_ij_splits[2]), 2);
        expected_norm += pow(norm(expected_ij_splits[2]), 2);

        // oo
        actual_norm += pow(norm(actual_ij_splits[3]), 2);
        expected_norm += pow(norm(expected_ij_splits[3]), 2);
      }
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
dense_cholesky_test(const SymmetricSharedBasisMatrix& A, const Domain& domain, const Hatrix::Args& opts) {
  SymmetricSharedBasisMatrix A_test(A);
  SymmetricSharedBasisMatrix expected(A_test);

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);

    make_dense(A_test, level);
    expected.D.deep_copy(A_test.D);

    for (int64_t block = 0; block < nblocks; ++block) {
      factorize_diagonal(A_test, domain, block, level);
      triangle_reduction(A_test, domain, block, level);
      compute_schurs_complement(A_test, domain, block, level);
    }

    compute_trailing_cholesky(A_test, level);

    enforce_lower_triangle(A_test, level);

    auto actual = compute_product(A_test, level);

    enforce_lower_triangle(actual, level);
    enforce_lower_triangle(expected, level);
    double rel_error = check_error(actual, expected, level);

    std::cout << "level: " << level << " rel error: "<< rel_error << std::endl;

    merge_unfactorized_blocks(A_test, domain, level);
  }

  return A_test;
}
