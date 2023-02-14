#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class LdlTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {8, 16, 32};
};

// templated function to compare floats and doubles respectively
template <typename DT>
void inline expect_fp_eq(const DT a, const DT b, const std::basic_string<char>& err_msg) {
  if (std::is_same<DT, double>::value){
    EXPECT_DOUBLE_EQ(a, b) << err_msg;
  }     
  else {
    EXPECT_NEAR(a, b, 10e-3) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LdlTests, Types);

TYPED_TEST(LdlTests, Ldl) {
  for (auto const& m : this->sizes) {

    // Generate SPD Matrix
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, m);
    int64_t d = m * m;
    for (int64_t i = 0; i < m; ++i) {
      A(i, i) += d--;
      for(int64_t j = i+1; j < m; j++) {
        A(i, j) = A(j, i);
      }
    }

    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::Matrix<TypeParam> A_rebuilt(m, m);
    Hatrix::ldl(A);
  
    Hatrix::Matrix<TypeParam> L = lower_tri(A, true);
    Hatrix::Matrix<TypeParam> D(m, m);
    for(int64_t i = 0; i < m; i++) D(i, i) = A(i, i);  
    Hatrix::matmul(L*D, L, A_rebuilt, false, true, 1, 0);

    // Check result
    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        expect_fp_eq(A_rebuilt(i, j), A_copy(i, j),
        "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
      +"] (" + std::to_string(m) + "x" + std::to_string(m) + " matrix)");
      }
    }
  }
}

template <typename DT>
class SolveDiagTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {16, 32};
  std::vector<std::tuple<int64_t, int64_t>> dims;

  // Parameters used in the tests
  std::vector<Hatrix::Side> side = {Hatrix::Left, Hatrix::Right};
  std::vector<double> scalars = {-1, 0.5, 1};
  std::vector<std::tuple<Hatrix::Side, double>> params;

  void SetUp() override {
    for (size_t i = 0; i < side.size(); ++i) {
      for (size_t j = 0; j < scalars.size(); ++j) {
        params.push_back(
          std::make_tuple(side[i], scalars[j]));
      }
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < sizes.size(); ++j) {
          dims.push_back(std::make_tuple(sizes[i], sizes[j]));
      }
    }
  }
};

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SolveDiagTests, Types);

TYPED_TEST(SolveDiagTests, SolveDiagonal) {
  for (auto const& [m, n] : this->dims) {
    for (auto const& [side, alpha] : this->params) {
      Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, m);
      Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(
        side == Hatrix::Left ? m : n, side == Hatrix::Left ? n : m);
  
      // Set a large value on the diagonal
      int64_t d = m * m;
      for (int64_t i = 0; i < m; ++i) {
        A(i, i) += d--;
      }

      Hatrix::Matrix<TypeParam> B_copy(B);
      Hatrix::solve_diagonal(A, B, side, alpha);
  
      // Check result
      for (int64_t i = 0; i < B.rows; ++i) {
        for (int64_t j = 0; j < B.cols; ++j) {
          TypeParam val = B(i, j) * (side == Hatrix::Left ? A(i, i) : A(j, j)) / alpha;
          expect_fp_eq(B_copy(i, j), val,
            "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
            +"] (" + std::to_string(m) + "x" + std::to_string(m) + " matrix)");
        }
      }
    }
  }
}

class LDLTests
  : public testing::TestWithParam<std::tuple<int64_t>> {};
class SolveDiagonalTests :
  public testing::TestWithParam<std::tuple<int64_t, int64_t, Hatrix::Side, double>> {};

TEST_P(LDLTests, ldl) {
  Hatrix::Context::init();
  int64_t m;
  std::tie(m) = GetParam();

  // Generate SPD Matrix
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, m);
  int64_t d = m * m;
  for (int64_t i = 0; i < m; ++i) {
    A(i, i) += d--;
    for(int64_t j = i+1; j < m; j++) {
      A(i, j) = A(j, i);
    }
  }

  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix A_rebuilt(m, m);
  Hatrix::ldl(A);
  
  Hatrix::Matrix L = lower_tri(A, true);
  Hatrix::Matrix D(m, m);
  for(int64_t i = 0; i < m; i++) D(i, i) = A(i, i);  
  Hatrix::matmul(L*D, L, A_rebuilt, false, true, 1, 0);

  // Check result
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_FLOAT_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }

  Hatrix::Context::finalize();
}

TEST_P(SolveDiagonalTests, solve_diagonal) {
  Hatrix::Context::init();
  int64_t m, n;
  Hatrix::Side side;
  double alpha;
  std::tie(m, n, side, alpha) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, m);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(
    side == Hatrix::Left ? m : n, side == Hatrix::Left ? n : m);
  
  // Set a large value on the diagonal
  int64_t d = m * m;
  for (int64_t i = 0; i < m; ++i) {
    A(i, i) += d--;
  }

  Hatrix::Matrix B_copy(B);
  Hatrix::solve_diagonal(A, B, side, alpha);
  
  // Check result
  for (int64_t i = 0; i < B.rows; ++i) {
    for (int64_t j = 0; j < B.cols; ++j) {
      double val = B(i, j) * (side == Hatrix::Left ? A(i, i) : A(j, j)) / alpha;
      EXPECT_NEAR(B_copy(i, j), val, 1e-14);
    }
  }

  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, LDLTests,
    testing::Combine(testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<LDLTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)));
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    BLAS, SolveDiagonalTests,
    testing::Combine(testing::Values(16, 32), testing::Values(16, 32),
                     testing::Values(Hatrix::Left, Hatrix::Right),
                     testing::Values(-1., 0.5, 1.)));
