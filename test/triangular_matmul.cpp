#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class TriMatMulTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {16, 32};
  std::vector<std::tuple<int64_t, int64_t>> dims;

  // Parameters used in the tests
  std::vector<bool> trans = {true, false};
  std::vector<bool> diag = {true, false};
  std::vector<Hatrix::Side> side = {Hatrix::Left, Hatrix::Right};
  std::vector<Hatrix::Mode> mode = {Hatrix::Lower, Hatrix::Upper};
  std::vector<double> scalars = {-1, -0.5, 0, 0.5, 1};
  std::vector<std::tuple<bool, bool, Hatrix::Side, Hatrix::Mode, double>> params;

  void SetUp() override {
    for (size_t i = 0; i < trans.size(); ++i) {
      for (size_t j = 0; j < diag.size(); ++j) {
        for (size_t k = 0; k < side.size(); ++k) {
          for (size_t l = 0; l < mode.size(); ++l) {
            for (size_t m = 0; m < scalars.size(); ++m)
            params.push_back(
              std::make_tuple(trans[i], diag[j], side[k], 
                mode[l], scalars[m]));
          }
        }
      }
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < sizes.size(); ++j) {
          dims.push_back(std::make_tuple(sizes[i], sizes[j]));
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

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TriMatMulTests, Types);

TYPED_TEST(TriMatMulTests, tri_matmul) {
  for (auto const& [M, N] : this->dims) {
    for (auto const& [transA, diag, side, mode, alpha] : this->params) {
      Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(M, N);
      Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(
        side == Hatrix::Left ? M : N, side == Hatrix::Left ? M : N);
      Hatrix::Matrix<TypeParam> B_copy(B);
      Hatrix::Matrix<TypeParam> A_tri(A);
      // Construct triangular A_tri
      for (int64_t j = 0; j < A_tri.cols; j++) {
        A_tri(j, j) = diag ? 1. : A(j, j);
        if (mode == Hatrix::Lower)
          for (int i = 0; i < j; i++) A_tri(i, j) = 0.;
        else
          for (int i = j + 1; i < A_tri.rows; i++) A_tri(i, j) = 0.;
      }

      Hatrix::triangular_matmul(A, B, side, mode, transA, diag, alpha);

      // Manual matmul
      // B_check = A_tri*B_copy or B_copy*A_tri
      Hatrix::Matrix<TypeParam> B_check(M, N);
      for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
          if (side == Hatrix::Left) {
            for (int64_t k = 0; k < M; k++) {
              if (transA)
                B_check(i, j) += alpha * A_tri(k, i) * B_copy(k, j);
              else
                B_check(i, j) += alpha * A_tri(i, k) * B_copy(k, j);
            }
          } else {
            for (int64_t k = 0; k < N; k++) {
              if (transA)
                B_check(i, j) += alpha * B_copy(i, k) * A_tri(j, k);
              else
                B_check(i, j) += alpha * B_copy(i, k) * A_tri(k, j);
            }
          }
        }
      }

      // Check result
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          expect_fp_eq(B_check(i, j), B(i, j),
            "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
            +"] "+"Dims(m = "+std::to_string(M)+", n = "+std::to_string(N)
            +") Params(transA = "+std::to_string(transA)+", diag = "
            +std::to_string(diag)+", side = "+std::to_string(side)
            +", mode = "+std::to_string(mode)+", alpha = "
            +std::to_string(alpha));
        }
      }
    }
  }
}
