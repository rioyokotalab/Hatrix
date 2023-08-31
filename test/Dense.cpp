#include <typeinfo>
#include <complex>
#include <utility>
//#include <vector>
//#include <iostream>

#include "Hatrix/classes/Dense.hpp"
#include "gtest/gtest.h"


template <typename DT>
class TypedDenseTests : public testing::Test {};

// template types used in the following tests
// should cover all instantiated types
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(TypedDenseTests, Types);

TYPED_TEST(TypedDenseTests, DefaultConstructor) {
  Hatrix::Dense<TypeParam> A;
  EXPECT_EQ(A.rows, 0);
  EXPECT_EQ(A.cols, 0);
  EXPECT_EQ(A.is_view, false);
  EXPECT_EQ(&A, nullptr);
  EXPECT_EQ(typeid(&A), typeid(TypeParam*));

  // check default template type
  Hatrix::Dense B;
  EXPECT_EQ(typeid(&B), typeid(double*));
}

TYPED_TEST(TypedDenseTests, RowColConstructor) {
  const unsigned int M = 3;
  const unsigned int N = 5;

  Hatrix::Dense<TypeParam> A(M, N, false);
  EXPECT_EQ(A.rows, M);
  EXPECT_EQ(A.cols, N);
  EXPECT_EQ(A.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(TypeParam*));
  EXPECT_NE(&A, nullptr);

  Hatrix::Dense<TypeParam> B(N, M, false);
  EXPECT_EQ(B.rows, N);
  EXPECT_EQ(B.cols, M);
  EXPECT_EQ(B.is_view, false);
  EXPECT_EQ(typeid(&B), typeid(TypeParam*));
  EXPECT_NE(&B, nullptr);

  // this should be equal to default constructor
  Hatrix::Dense<TypeParam> C(0, 0, false);
  EXPECT_EQ(C.rows, 0);
  EXPECT_EQ(C.cols, 0);
  EXPECT_EQ(C.is_view, false);
  EXPECT_EQ(typeid(&C), typeid(TypeParam*));
  EXPECT_EQ(&C, nullptr);
}

TYPED_TEST(TypedDenseTests, RowColConstructorInit) {
  const unsigned int M = 3;
  const unsigned int N = 5;

  // check that default is 0-initialized
  Hatrix::Dense<TypeParam> A(M, N);
  EXPECT_EQ(A.rows, M);
  EXPECT_EQ(A.cols, N);
  EXPECT_EQ(A.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(TypeParam*));
  EXPECT_NE(&A, nullptr);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ((TypeParam) 0, *(&A + j * A.rows + i));
    }
  }

  Hatrix::Dense<TypeParam> B(N, M, true);
  EXPECT_EQ(B.rows, N);
  EXPECT_EQ(B.cols, M);
  EXPECT_EQ(B.is_view, false);
  EXPECT_EQ(typeid(&B), typeid(TypeParam*));
  EXPECT_NE(&B, nullptr);
  for (unsigned int j = 0; j < B.cols; ++j) {
    for (unsigned int i = 0; i < B.rows; ++i) {
      EXPECT_EQ((TypeParam) 0, *(&B + j * B.rows + i));
    }
  }

  // this should not allocate memory
  Hatrix::Dense<TypeParam> C(0, M, true);
  EXPECT_EQ(C.rows, 0);
  EXPECT_EQ(C.cols, M);
  EXPECT_EQ(C.is_view, false);
  EXPECT_EQ(typeid(&C), typeid(TypeParam*));
  EXPECT_EQ(&C, nullptr);
}

TYPED_TEST(TypedDenseTests, CopyConstructor) {
  const unsigned int M = 8;
  const unsigned int N = 5;

  Hatrix::Dense<TypeParam> A(M, N);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      *(&A + j * A.rows + i) = (TypeParam) (i * A.cols + j);
    }
  }

  // Deep copy
  Hatrix::Dense<TypeParam> A_copy = A;  // copy constructor
  EXPECT_EQ(A_copy.rows, A.rows);
  EXPECT_EQ(A_copy.cols, A.cols);
  EXPECT_EQ(A_copy.is_view, A.is_view);
  EXPECT_EQ(typeid(&A), typeid(&A_copy));
  EXPECT_NE(&A, &A_copy);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy + j * A.rows + i), *(&A + j * A.rows + i));
    }
  }
  // Verify that a deep copy is created
  *&A_copy = (TypeParam) 1;
  EXPECT_EQ(*&A, (TypeParam) 0);

  // View (i.e. shallow copy)
  // TODO overthink this
  A.is_view = true;
  Hatrix::Dense<TypeParam> A_view(A);
  A.is_view = false;
  EXPECT_EQ(A_view.rows, A.rows);
  EXPECT_EQ(A_view.cols, A.cols);
  EXPECT_EQ(A_view.is_view, true);
  EXPECT_EQ(&A, &A_view);
  // Verify that a shallow copy is created
  *&A_view = (TypeParam) 1;
  EXPECT_EQ(*&A, (TypeParam) 1);
}

TYPED_TEST(TypedDenseTests, ExplicitCopyConstructor) {
  const unsigned int M = 5;
  const unsigned int N = 8;

  Hatrix::Dense<TypeParam> A(M, N);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      *(&A + j * A.rows + i) = (TypeParam) (i * A.cols + j);
    }
  }

  // Explicit deep copy of non view
  Hatrix::Dense<TypeParam> A_copy(A, true);
  EXPECT_EQ(A_copy.rows, A.rows);
  EXPECT_EQ(A_copy.cols, A.cols);
  EXPECT_EQ(A_copy.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(&A_copy));
  EXPECT_NE(&A, &A_copy);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy + j * A.rows + i), *(&A + j * A.rows + i));
    }
  }
  // Verify that a deep copy is created
  *&A_copy = (TypeParam) 1;
  EXPECT_EQ(*&A, (TypeParam) 0);
  
  // Explicit deep copy of a view
  A.is_view = true;
  Hatrix::Dense<TypeParam> A_copy2(A, true);
  A.is_view = false;
  EXPECT_EQ(A_copy2.rows, A.rows);
  EXPECT_EQ(A_copy2.cols, A.cols);
  EXPECT_EQ(A_copy2.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(&A_copy2));
  EXPECT_NE(&A, &A_copy2);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy2+ j * A.rows + i), *(&A + j * A.rows + i));
    }
  }
  // Verify that a deep copy is created
  *&A_copy2 = (TypeParam) 1;
  EXPECT_EQ(*&A, (TypeParam) 0);

  // View (i.e. shallow copy) of a non view
  Hatrix::Dense<TypeParam> A_view(A, false);
  EXPECT_EQ(A_view.rows, A.rows);
  EXPECT_EQ(A_view.cols, A.cols);
  EXPECT_EQ(A_view.is_view, true);
  EXPECT_EQ(&A, &A_view);
  // Verify that a shallow copy is created
  *&A_view = (TypeParam) 1;
  EXPECT_EQ(*&A, (TypeParam) 1);

  // View (i.e. shallow copy) of a view
  Hatrix::Dense<TypeParam> A_view2(A_view, false);
  EXPECT_EQ(A_view2.rows, A_view.rows);
  EXPECT_EQ(A_view2.cols, A_view.cols);
  EXPECT_EQ(A_view2.is_view, true);
  EXPECT_EQ(&A_view, &A_view2);
  EXPECT_EQ(&A, &A_view2);
  // Verify that a shallow copy is created
  *&A_view2 = (TypeParam) 2;
  EXPECT_EQ(*&A_view, (TypeParam) 2);
  EXPECT_EQ(*&A, (TypeParam) 2);
}

TYPED_TEST(TypedDenseTests, MoveConstructor) {
  const unsigned int M = 7;
  const unsigned int N = 4;

  Hatrix::Dense<TypeParam> A(M, N);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      *(&A + j * A.rows + i) = (TypeParam) (i * A.cols + j);
    }
  }
  
  // Move a non-view
  Hatrix::Dense<TypeParam> B = std::move(A);  // move constructor
  EXPECT_EQ(B.rows, M);
  EXPECT_EQ(B.cols, N);
  EXPECT_EQ(B.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(TypeParam*));
  EXPECT_EQ(A.rows, 0);
  EXPECT_EQ(A.cols, 0);
  EXPECT_EQ(A.is_view, false);
  EXPECT_EQ(&A, nullptr);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&B + j * B.rows + i), (TypeParam) (i * B.cols + j));
    }
  }
  
  // Move a view
  Hatrix::Dense<TypeParam> B_view(B, false);
  Hatrix::Dense<TypeParam> C(std::move(B_view));
  EXPECT_EQ(B_view.rows, C.rows);
  EXPECT_EQ(B_view.cols, C.cols);
  EXPECT_EQ(C.is_view, B_view.is_view);
  EXPECT_EQ(C.is_view, true);
  EXPECT_EQ(typeid(&C), typeid(TypeParam*));
  EXPECT_EQ(&C, &B_view);
  // Verify that a shallow copy is created
  *&C = (TypeParam) 1;
  EXPECT_EQ(*&B, (TypeParam) 1);
}

/*
TYPED_TEST(TypedDenseTests, ConstructorViews) {
  const unsigned int m = 8;
  const unsigned int n = 5;

  Hatrix::Dense<TypeParam> A(m, n);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      *(&A + j * A.rows + i) = (TypeParam) (i * A.cols + j);
    }
  }

  // Deep copy
  Hatrix::Dense<TypeParam> A_copy = A;  // copy constructor
  EXPECT_EQ(A_copy.rows, m);
  EXPECT_EQ(A_copy.cols, n);
  EXPECT_EQ(A_copy.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(&A_copy));
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy + j * A.rows + i), (TypeParam) (i * A.cols + j));
    }
  }
  A_copy(0, 0) = (TypeParam) 1;
  EXPECT_EQ(A(0, 0), (TypeParam) 0);

  // Explicit deep copy
  Hatrix::Dense<TypeParam> A_copy_exp(A, true);
  EXPECT_EQ(A_copy_exp.rows, m);
  EXPECT_EQ(A_copy_exp.cols, n);
  EXPECT_EQ(A_copy_exp.is_view, false);
  EXPECT_EQ(typeid(&A), typeid(&A_copy_exp));
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy_exp + j * A.rows + i), (TypeParam) (i * A.cols + j));
    }
  }
  A_copy(0, 0) = (TypeParam) 1;
  EXPECT_EQ(A(0, 0), (TypeParam) 0);

  // View (i.e. shallow copy)
  Hatrix::Dense<TypeParam> A_view(A, false);
  EXPECT_EQ(A_view.rows, m);
  EXPECT_EQ(A_view.cols, n);
  EXPECT_EQ(A_view.is_view, true);
  EXPECT_TRUE(&A == &A_view);
  A_view(0, 0) = (TypeParam) 1;
  EXPECT_EQ(A(0, 0), (TypeParam) 1);

  // View of a View (i.e. shallow copy)
  Hatrix::Dense<TypeParam> A_view2 = A_view; // copy constructor
  EXPECT_EQ(A_view2.rows, m);
  EXPECT_EQ(A_view2.cols, n);
  EXPECT_EQ(A_view2.is_view, true);
  EXPECT_TRUE(&A == &A_view2);
  A_view2(0, 0) = (TypeParam) 2;
  EXPECT_EQ(A(0, 0), (TypeParam) 2);

  // Explicit View (i.e. shallow copy)
  Hatrix::Dense<TypeParam> A_view_exp(A_view, false);
  EXPECT_EQ(A_view_exp.rows, m);
  EXPECT_EQ(A_view_exp.cols, n);
  EXPECT_EQ(A_view_exp.is_view, true);
  EXPECT_TRUE(&A == &A_view_exp);
  A_view_exp(0, 0) = (TypeParam) 0;
  EXPECT_EQ(A(0, 0), (TypeParam) 0);

  // Copy of a View (i.e. deep copy)
  Hatrix::Dense<TypeParam> A_copy2(A_view, true);
  EXPECT_EQ(A_copy2.rows, m);
  EXPECT_EQ(A_copy2.cols, n);
  EXPECT_EQ(A_copy2.is_view, false);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&A_copy2+ j * A.rows + i), (TypeParam) (i * A.cols + j));
    }
  }
  A_copy2(0, 0) = (TypeParam) 1;
  EXPECT_EQ(A_view(0, 0), (TypeParam) 0);
}
*/
TYPED_TEST(TypedDenseTests, CopyAssignment) {
  const unsigned int m = 8;
  const unsigned int n = 5;

  Hatrix::Dense<TypeParam> A(m, n);
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      *(&A + j * A.rows + i) = (TypeParam) (i * A.cols + j);
    }
  }
  
  // TODO allow different size?
  Hatrix::Dense<TypeParam> B(m, n);
  B = A;
  for (unsigned int j = 0; j < A.cols; ++j) {
    for (unsigned int i = 0; i < A.rows; ++i) {
      EXPECT_EQ(*(&B+ j * A.rows + i), *(&A+ j * A.rows + i));
    }
  }
}

/*
TEST(DenseTests, CopyAssignment) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy(block_size, block_size);

  A_copy = A;
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A(i, j), A_copy(i, j));
    }
  }
}

TEST(DenseTests, MoveConstructor) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy = A;

  Hatrix::Matrix A_move(std::move(A));
  // Check result
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A_copy(i, j), A_move(i, j));
    }
  }
}

TEST(DenseTests, MoveAssignment) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy(A), A_move(A);

  A_move = std::move(A);
  // Check result
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A_copy(i, j), A_move(i, j));
    }
  }
}

TEST(DenseTests, ViewAssignment) {
  int64_t block_size = 40, sub_block = 10;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  auto A_splits = A.split(4, 4);

  Hatrix::Matrix& A1 = A_splits[1];

  for (int64_t i = 0; i < sub_block; ++i) {
    for (int64_t j = 0; j < sub_block; ++j) {
      A1(i, j) = 0;
    }
  }

  for (int64_t i = 0; i < sub_block; ++i) {
    for (int64_t j = 0; j < sub_block; ++j) {
      EXPECT_EQ(A_splits[1](i, j), 0.0);
    }
  }
}

TEST(DenseTests, ViewMoveAssignment) {
  int64_t block_size = 40, sub_block = 10;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  auto A_splits = A.split(4, 4);
  A_splits[1] = Hatrix::generate_identity_matrix(sub_block, sub_block);

  for (int64_t i = 0; i < sub_block; ++i) {
    for (int64_t j = 0; j < sub_block; ++j) {
      if (i == j)  {
        EXPECT_EQ(A(i, j + sub_block), 1);
      }
      else {
        EXPECT_EQ(A(i, j + sub_block), 0);
      }
    }
  }
}

TEST(DenseTests, shrink) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_shrunk(A);

  int64_t shrunk_size = 8;
  A_shrunk.shrink(shrunk_size, shrunk_size);

  // Check result
  for (int64_t i = 0; i < shrunk_size; ++i) {
    for (int64_t j = 0; j < shrunk_size; ++j) {
      EXPECT_EQ(A(i, j), A_shrunk(i, j));
    }
  }
}

class DenseTests : public testing::TestWithParam<
                        std::tuple<int64_t, int64_t, int64_t, int64_t, bool>> {
};

TEST_P(DenseTests, split) {
  int64_t M, N;
  int64_t m_splits, n_splits;
  bool copy;
  std::tie(M, N, m_splits, n_splits, copy) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, N);

  std::vector<Hatrix::Matrix> split = A.split(m_splits, n_splits, copy);

  // Check result
  int64_t m_start = 0, n_start = 0;
  for (int64_t i_c = 0; i_c < m_splits; ++i_c) {
    n_start = 0;
    for (int64_t j_c = 0; j_c < n_splits; ++j_c) {
      Hatrix::Matrix& block = split[i_c * n_splits + j_c];
      for (int64_t i = 0; i < block.rows; ++i) {
        for (int64_t j = 0; j < block.cols; ++j) {
          EXPECT_EQ(A(m_start + i, n_start + j), block(i, j));
        }
      }
      n_start += block.cols;
    }
    m_start += split[i_c * n_splits].rows;
  }
}

INSTANTIATE_TEST_SUITE_P(Sizes, DenseTests,
                         testing::Combine(testing::Values(32, 57),
                                          testing::Values(32, 57),
                                          testing::Values(1, 3, 8),
                                          testing::Values(1, 3, 8),
                                          testing::Values(true, false)));

TEST(DenseTests, shrinktest) {
  Hatrix::Matrix A = Hatrix::generate_random_matrix(64, 64);
  std::vector<int64_t> row_split_indices{6, 37, 55};
  std::vector<int64_t> col_split_indices{25, 41, 49};

  std::vector<Hatrix::Matrix> split =
      A.split(row_split_indices, col_split_indices, false);

  // Check result
  int64_t m_start = 0, n_start = 0;
  for (int64_t i_c = 0; i_c < row_split_indices.size() + 1; ++i_c) {
    n_start = 0;
    for (int64_t j_c = 0; j_c < col_split_indices.size() + 1; ++j_c) {
      Hatrix::Matrix& block = split[i_c * (col_split_indices.size()+1) + j_c];
      EXPECT_EQ(i_c == 0 ? 0 : row_split_indices[i_c-1], m_start);
      EXPECT_EQ(j_c == 0 ? 0 : col_split_indices[j_c-1], n_start);
      for (int64_t i = 0; i < block.rows; ++i) {
        for (int64_t j = 0; j < block.cols; ++j) {
          EXPECT_EQ(A(m_start + i, n_start + j), block(i, j));
        }
      }
      n_start += block.cols;
    }
    m_start += split[i_c * (col_split_indices.size()+1)].rows;
  }
}

TEST(DenseTests, uniform_split_copy) {
  int N = 100, Nslice = N / 2;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(N, N);
  std::vector<Hatrix::Matrix> A_splits = A.split(2, 2, false);
  Hatrix::Matrix B = Hatrix::generate_identity_matrix(Nslice, Nslice);

  A_splits[1] = B;

  // Verify upper right corner
  for (int i = 0; i < B.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      if (i == j) {
        EXPECT_EQ(A(i, j + Nslice) , 1.0);
      }
      else {
        EXPECT_EQ(A(i, j + Nslice) , 0.0);
      }
    }
  }

  A_splits[2] = B;

  // Verify lower right corner
  for (int i = 0; i < B.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      if (i == j) {
        EXPECT_EQ(A(i + Nslice, j), 1.0);
      }
      else {
        EXPECT_EQ(A(i + Nslice, j), 0.0);
      }
    }
  }
}

TEST(DenseTests, non_uniform_square_split_copy) {
  int N = 40; int Nslice = 10; int split_dim = 30;
  std::vector<int64_t> split_vector = {split_dim};
  Hatrix::Matrix A = Hatrix::generate_random_matrix(N, N);
  std::vector<Hatrix::Matrix> A_splits = A.split(split_vector, split_vector, false);
  Hatrix::Matrix B = Hatrix::generate_identity_matrix(Nslice, Nslice);

  A_splits[3] = B;

  for (int i = 0; i < B.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      if (i == j) {
        EXPECT_EQ(A(i + split_dim, j + split_dim), 1.0);
      }
      else {
        EXPECT_EQ(A(i + split_dim, j + split_dim), 0.0);
      }
    }
  }
}

TEST(DenseTests, non_uniform_rectangle_split_copy) {
  int64_t N = 40; int64_t Nslice = 10; int64_t split_dim = 30;
  std::vector<int64_t> y_split_vector = {split_dim};
  std::vector<int64_t> x_split_vector = {0};
  Hatrix::Matrix A = Hatrix::generate_random_matrix(N, N);
  std::vector<Hatrix::Matrix> A_splits = A.split(x_split_vector, y_split_vector, false);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(N, Nslice);

  A_splits[3] = B;

  for (int i = 0; i < B.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      EXPECT_EQ(A(i, j + split_dim), B(i, j));
    }
  }
}

TEST(DenseTests, split_no_split) {
  int64_t N = 40;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(N, N);
  std::vector<Hatrix::Matrix> A_splits = A.split(std::vector<int64_t>(1, 0),
                                                 std::vector<int64_t>(1, 0), false);
  // This should produce empty splits (A_splits[0], A_splits[1], A_splits[2])
  // And the full matrix in A_splits[3]
  for (int i = 0; i < A_splits[3].rows; ++i) {
    for (int j = 0; j < A_splits[3].cols; ++j) {
      EXPECT_EQ(A(i, j), A_splits[3](i, j));
    }
  }
}

TEST(DenseTests, split_null_at_the_end) {
  RecordProperty("description",
                 "Produce empty splits in location 1, 2 and 3. The full matrix is present in location 0.");
  int64_t N = 40;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(N, N);
  std::vector<Hatrix::Matrix> A_splits = A.split(std::vector<int64_t>(1, N),
                                                 std::vector<int64_t>(1, N), false);

  for (int i = 0; i < A_splits[0].rows; ++i) {
    for (int j = 0; j < A_splits[0].cols; ++j) {
      EXPECT_EQ(A(i, j), A_splits[0](i, j));
    }
  }
}

TEST(DenseTests, split_vector_row_end) {
  int64_t N = 40;
  Hatrix::Matrix V = Hatrix::generate_random_matrix(N, 1);
  std::vector<Hatrix::Matrix> V_splits = V.split({N/4, N/2, (3 * N) / 4}, {});

  EXPECT_EQ(V_splits.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(V_splits[i].rows, N/4);
    EXPECT_EQ(V_splits[i].cols, 1);
  }
}*/
