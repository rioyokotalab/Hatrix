#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class ArithmeticTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<std::tuple<int64_t, int64_t>> params = {
    std::make_tuple(50, 50),
    std::make_tuple(23, 75),
    std::make_tuple(100, 66)
  };
};

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ArithmeticTests, Types);

TYPED_TEST(ArithmeticTests, PlusOperator) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> C = A + B;

    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_EQ(C(i, j), A(i, j) + B(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }

    // Check that A and B are unmodified.
    // TODO is this not guaranteed by the const declaration?
    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_NE(C(i, j), A(i, j)) << "A was modified at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
        EXPECT_NE(C(i, j), B(i, j)) << "B was modified at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, PlusEqualsOperator) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_check(A);
    Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(m, n);
    A += B;

    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_EQ(A_check(i, j) + B(i, j), A(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, MinusOperator) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> C = A - B;

    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_EQ(C(i, j), A(i, j) - B(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }

    // Check that A and B are unmodified.
    // TODO is this not guaranteed by the const declaration?
    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_NE(C(i, j), A(i, j)) << "A was modified at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
        EXPECT_NE(C(i, j), B(i, j)) << "B was modified at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, MinusEqualsOperator) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_check(A);
    Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(m, n);
    A -= B;

    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_EQ(A_check(i, j) - B(i, j), A(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, abs) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_check = abs(A);

    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        EXPECT_EQ(A_check(i, j), A(i, j) < 0 ? -A(i, j) : A(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, Transpose) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_trans = transpose(A);

    EXPECT_EQ(A_trans.rows, n) << "Wrong row-dimension ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_trans.cols, m) << "Wrong column-dimension ("<<m<<"x"<<n<<" matrix)";
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(A(i, j), A_trans(j, i)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, LowerTriangularPart) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_nounit_lower = lower_tri(A);
    Hatrix::Matrix<TypeParam> A_unit_lower = lower_tri(A, true);

    EXPECT_EQ(A_nounit_lower.rows, m) << "Wrong row-dimension (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_nounit_lower.cols, n) << "Wrong column-dimension (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_unit_lower.rows, m) << "Wrong row-dimension (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_unit_lower.cols, n) << "Wrong column-dimension (unit diagonal) ("<<m<<"x"<<n<<" matrix)";

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        if(i == j) {
          EXPECT_EQ(A(i, j), A_nounit_lower(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
          EXPECT_EQ(1., A_unit_lower(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
        }
        else if(i > j) {
          EXPECT_EQ(A(i, j), A_nounit_lower(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
          EXPECT_EQ(A(i, j), A_unit_lower(i, j))  << "Wrong value at index ["<<i<<", "<<j<<"] (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
        }
        else {
          EXPECT_EQ(0., A_nounit_lower(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
          EXPECT_EQ(0., A_unit_lower(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
        }
      }
    }
  }
}

TYPED_TEST(ArithmeticTests, UpperTriangularPart) {
  for (auto const& [m, n] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_nounit_upper = upper_tri(A);
    Hatrix::Matrix<TypeParam> A_unit_upper = upper_tri(A, true);

    EXPECT_EQ(A_nounit_upper.rows, m) << "Wrong row-dimension (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_nounit_upper.cols, n) << "Wrong column-dimension (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_unit_upper.rows, m) << "Wrong row-dimension (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    EXPECT_EQ(A_unit_upper.cols, n) << "Wrong column-dimension (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        if(i == j) {
	  EXPECT_EQ(A(i, j), A_nounit_upper(i, j)) << "Wrong value at index ["<<i<<","<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
	  EXPECT_EQ(1., A_unit_upper(i, j)) << "Wrong value at index ["<<i<<","<<j<<"] (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
        }
        else if(i > j) {
	  EXPECT_EQ(0., A_nounit_upper(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
	  EXPECT_EQ(0., A_unit_upper(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
        }
        else {
	  EXPECT_EQ(A(i, j), A_nounit_upper(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (non-unit diagonal) ("<<m<<"x"<<n<<" matrix)";
	  EXPECT_EQ(A(i, j), A_unit_upper(i, j)) << "Wrong value at index ["<<i<<", "<<j<<"] (unit diagonal) ("<<m<<"x"<<n<<" matrix)";
        }
      }
    }
  }
}

template <typename DT>
class ScalarArithmeticTests : public testing::Test {
  protected:
  // Matrix dimensions and scalar parameter used in the tests
  std::vector<std::tuple<int64_t, int64_t, DT>> params = {
    std::make_tuple(5, 5, 7.9834),
    std::make_tuple(11, 21, -4),
    std::make_tuple(18, 5, 1/8)
  };
};

// templated function to compare floats and doubles respectively
template <typename DT>
void inline expect_fp_eq(const DT a, const DT b, const std::basic_string<char>& err_msg) {
  if (std::is_same<DT, double>::value){
    EXPECT_DOUBLE_EQ(a, b) << err_msg;
  }     
  else {
    EXPECT_FLOAT_EQ(a, b) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScalarArithmeticTests, Types);

TYPED_TEST(ScalarArithmeticTests, ScalarMultiplicationOperator) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> B = A * alpha;
    Hatrix::Matrix<TypeParam> C = alpha * A;

    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        expect_fp_eq(A(i, j) * alpha, C(i, j),
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)+"] (scalar * matrix) ("
          +std::to_string(m)+"x"+std::to_string(n)+" matrix, alpha = "
          +std::to_string(alpha)+")");
        expect_fp_eq(A(i, j) * alpha, B(i, j),
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)+"] (matrix * scalar) ("
          +std::to_string(m)+"x"+std::to_string(n)+" matrix, alpha = "
          +std::to_string(alpha)+")");
      }
    }
  }
}

TYPED_TEST(ScalarArithmeticTests, ScalarMultiplicationEqualsOperator) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    A *= alpha;

    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        expect_fp_eq(A(i, j), A_copy(i, j) * alpha,
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)+"] ("
          +std::to_string(m)+"x"+std::to_string(n)+" matrix, alpha = "
          +std::to_string(alpha)+")");
      }
    }
  }
}

TYPED_TEST(ScalarArithmeticTests, ScalarDivisionOperator) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> B = A / alpha;

    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        expect_fp_eq(A(i, j) / alpha, B(i, j),
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)+"] ("
          +std::to_string(m)+"x"+std::to_string(n)+" matrix, alpha = "
          +std::to_string(alpha)+")");
      }
    }
  }
}

TYPED_TEST(ScalarArithmeticTests, ScalarDivisionEqualsOperator) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    A /= alpha;

    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        expect_fp_eq(A(i, j), A_copy(i, j) / alpha,
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)+"] ("
          +std::to_string(m)+"x"+std::to_string(n)+" matrix, alpha = "
          +std::to_string(alpha)+")");
      }
    }
  }
}

template <typename DT>
class MatMulOperatorTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {16, 32};
  std::vector<std::tuple<int64_t, int64_t, int64_t>> dims;
  
  void SetUp() override {
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < sizes.size(); ++j) {
        for (size_t k = 0; k < sizes.size(); ++k) {
          dims.push_back(std::make_tuple(sizes[i], sizes[j], sizes[k]));
        }
      }
    }
  }
};

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatMulOperatorTests, Types);

TYPED_TEST(MatMulOperatorTests, MatMulOperator) {
  for (auto const& [m, n, k] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, k);
    Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(k, n);
    Hatrix::Matrix<TypeParam> C(m, n);
    Hatrix::Matrix<TypeParam> C_check = A * B;
    Hatrix::matmul(A, B, C, false, false, 1, 0);

    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(C_check(i, j), C(i, j))  << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
      }
    }
  }
}

TYPED_TEST(MatMulOperatorTests, MatMulEqualsOperator) {
  for (auto const& [m, n, k] : this->dims) {
    if (n == k) {
      Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, k);
      Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(k, n);
      Hatrix::Matrix<TypeParam> C(m, n);
      Hatrix::matmul(A, B, C, false, false, 1, 0);
      A *= B;

      // Check result
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          EXPECT_EQ(A(i, j), C(i, j))  << "Wrong value at index ["<<i<<", "<<j<<"] ("<<m<<"x"<<n<<" matrix)";
        }
      }
    }
  }
}

