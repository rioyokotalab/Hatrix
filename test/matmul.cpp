#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class GemmTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {16, 32};
  std::vector<std::tuple<int64_t, int64_t, int64_t>> dims;

  // Parameters used in the tests
  std::vector<bool> trans = {true, false};
  std::vector<double> scalars = {-1, 0.5, 1};
  std::vector<std::tuple<bool, bool, double, double>> params;

  void SetUp() override {
    for (size_t i = 0; i < trans.size(); ++i) {
      for (size_t j = 0; j < trans.size(); ++j) {
        for (size_t k = 0; k < scalars.size(); ++k) {
          for (size_t l = 0; l < scalars.size(); ++l) {
            params.push_back(
              std::make_tuple(trans[i], trans[j], scalars[k], scalars[l]));
          }
        }
      }
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < sizes.size(); ++j) {
        for (size_t k = 0; k < sizes.size(); ++k) {
          dims.push_back(std::make_tuple(sizes[i], sizes[j], sizes[k]));
        }
      }
    }
  }
};

// templated function to compare floats and doubles respectively
template <typename DT>
void inline expect_fp_eq(const DT a, const DT b, const std::basic_string<char>& err_msg) {
  if (std::is_same<DT, double>::value){
    EXPECT_NEAR(a, b, 10e-14) << err_msg;
  }     
  else {
    EXPECT_NEAR(a, b, 10e-6) << err_msg;
  }                                        
}

// templated function to compare float and double matrices respectively
template <typename DT>
void inline matrix_fp_eq(const Hatrix::Matrix<DT> A, const Hatrix::Matrix<DT> B, const std::basic_string<char>& err_msg) {
  EXPECT_EQ(A.rows, B.rows);
  EXPECT_EQ(A.cols, B.cols);
  EXPECT_EQ(A.stride, B.stride);
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j < A.cols; ++j) {
      expect_fp_eq(A(i, j), B(i, j), 
      "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
      +"] " + err_msg);
    }
  }                                        
}

template <typename DT>
void naive_matmul(const Hatrix::Matrix<DT>& A, const Hatrix::Matrix<DT>& B, Hatrix::Matrix<DT>& C,
  const bool transA, const bool transB, const DT alpha, const DT beta) {
  const int64_t K = transA ? A.rows : A.cols;
  for (int64_t i = 0; i < C.rows; ++i) {
    for (int64_t j = 0; j < C.cols; ++j) {
      C(i, j) = (beta * C(i, j) +
        alpha * (transA ? A(0, i) : A(i, 0)) * (transB ? B(j, 0) : B(0, j)));
      for (int64_t k = 1; k < K; ++k) {
        C(i, j) += (alpha * (transA ? A(k, i) : A(i, k)) *
          (transB ? B(j, k) : B(k, j)));
      }
    }
  }
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GemmTests, Types);

TYPED_TEST(GemmTests, Gemm) {
  for (auto const& [M, N, K] : this->dims) {
    for (auto const& [transA, transB, alpha, beta] : this->params) {
      Hatrix::Matrix<TypeParam> A =
        Hatrix::generate_random_matrix<TypeParam>(transA ? K : M, transA ? M : K);
      Hatrix::Matrix<TypeParam> B =
        Hatrix::generate_random_matrix<TypeParam>(transB ? N : K, transB ? K : N);
      Hatrix::Matrix<TypeParam> C = 
        Hatrix::generate_random_matrix<TypeParam>(M, N);
      Hatrix::Matrix<TypeParam> C_check(C);
      Hatrix::matmul(A, B, C, transA, transB, alpha, beta);

      naive_matmul(A, B, C_check, transA, transB, static_cast<TypeParam>(alpha), static_cast<TypeParam>(beta));

      // Check result
      matrix_fp_eq(C_check, C,
        "Dims(m = "+std::to_string(M)+", k = "+std::to_string(K)+
        ", n = "+std::to_string(N)+") Params(transA = "
        +std::to_string(transA)+", transB = "+std::to_string(transB)
        +", alpha = "+std::to_string(alpha)+", beta = "
        +std::to_string(beta));
    }
  }
}

TYPED_TEST(GemmTests, GemmReturn) {
  for (auto const& [M, N, K] : this->dims) {
    for (auto const& [transA, transB, alpha, beta] : this->params) {
      Hatrix::Matrix<TypeParam> A =
        Hatrix::generate_random_matrix<TypeParam>(transA ? K : M, transA ? M : K);
      Hatrix::Matrix<TypeParam> B =
        Hatrix::generate_random_matrix<TypeParam>(transB ? N : K, transB ? K : N);
      Hatrix::Matrix<TypeParam> C = Hatrix::matmul(A, B, transA, transB, alpha);

      Hatrix::Matrix<TypeParam> C_check(C.rows, C.cols);
      naive_matmul(A, B, C_check, transA, transB, static_cast<TypeParam>(alpha), static_cast<TypeParam>(beta));

      // Check result
      matrix_fp_eq(C_check, C,
        "Dims(m = "+std::to_string(M)+", k = "+std::to_string(K)+
        ", n = "+std::to_string(N)+") Params(transA = "
        +std::to_string(transA)+", transB = "+std::to_string(transB)
        +", alpha = "+std::to_string(alpha)+", beta = "
        +std::to_string(beta));
    }
  }
}

//TODO is this really a unit test?
TEST(MatMulViewTests, matmulView) {
  Hatrix::Context::init();
  int64_t block = 100, sub_block = 25;
  int64_t splits = block / sub_block;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block, block);
  Hatrix::Matrix x = Hatrix::generate_random_matrix(block, 1);
  Hatrix::Matrix b(block, 1);

  auto A_splits = A.split(splits, splits);
  auto x_splits = x.split(splits, 1);
  auto b_splits = b.split(splits, 1);

  for (int64_t m = 0; m < splits; ++m) {
    for (int64_t n = 0; n < 1; ++n) {
      for (int64_t k = 0; k < splits; ++k) {
        matmul(A_splits[m * splits + k], x_splits[k], b_splits[m], false, false, 1, 1);
      }
    }
  }

  Hatrix::Matrix b_result = matmul(A, x);
  for (int64_t i = 0; i < block; ++i) {
    EXPECT_NEAR(b_result(i, 0), b(i, 0), 1e-13);
  }

  Hatrix::Context::finalize();
}
