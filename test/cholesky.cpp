#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.hpp"
#include "gtest/gtest.h"

class CholeskyTests : public testing::TestWithParam<std::tuple<int64_t>> {};

TEST_P(CholeskyTests, cholesky) {
  int64_t m;
  std::tie(m) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_spd_matrix(m, 10.0);
  // keep only the lower triangle of this matrix.
  for (int i = 0; i < m; ++i) {
    for (int j = i+1; j < m; ++j) {
      A(i, j) = 0;
    }
  }

  Hatrix::Matrix A_copy(A, true), A_actual(m, m);
  Hatrix::cholesky(A_copy, Hatrix::Lower);
  Hatrix::matmul(A_copy, A_copy, A_actual, false, true, 1, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = i+1; j < m; ++j) {
      A_actual(i, j) = 0;
    }
  }

  // Check result
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      EXPECT_FLOAT_EQ(A_actual(i, j), A(i, j));
    }
  }
}

TEST_P(CholeskyTests, block_cholesky) {
  int64_t m;
  std::tie(m) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_spd_matrix(m, 10.0);
  // keep only the lower triangle of this matrix.
  for (int i = 0; i < m; ++i) {
    for (int j = i+1; j < m; ++j) {
      A(i, j) = 0;
    }
  }

  Hatrix::Matrix expected(A, true);

  int block_size = 4;
  int nblocks = m / block_size;
  auto A_splits = A.split(nblocks, nblocks);

  for (int i = 0; i < nblocks; ++i) {
    Hatrix::cholesky(A_splits[i * nblocks + i], Hatrix::Lower);
    for (int j = i+1; j < nblocks; ++j) {
      solve_triangular(A_splits[i * nblocks + i], A_splits[j * nblocks + i],
                       Hatrix::Right, Hatrix::Lower,
                       false, true, 1);
    }
    for (int j = i+1; j < nblocks; ++j) {
      for (int k = i+1; k <= j; ++k) {
        if (j == k) {
          syrk(A_splits[j * nblocks + i], A_splits[j * nblocks + k],
               Hatrix::Lower, false, -1, 1);
        }
        else {
          matmul(A_splits[j * nblocks + i], A_splits[k * nblocks + i],
                 A_splits[j * nblocks + k], false, true, -1, 1);
        }
      }
    }
  }

  Hatrix::cholesky(expected, Hatrix::Lower);

  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j <= i; ++j) {
      EXPECT_FLOAT_EQ(expected(i, j), A(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, CholeskyTests,
    testing::Combine(testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<CholeskyTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)));
      return name;
    });
