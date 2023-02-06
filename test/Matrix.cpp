#include <cstdint>
#include <utility>
#include <vector>
#include <iostream>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class MatriXTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {32, 57};
  std::vector<std::tuple<int64_t, int64_t>> dims;

  // Parameters used in the tests
  std::vector<bool> copy = {true, false};
  std::vector<double> block_size = {1, 3, 8};
  std::vector<std::tuple<int64_t, int64_t, bool>> params;

  void SetUp() override {
    for (size_t i = 0; i < block_size.size(); ++i) {
      for (size_t j = 0; j < block_size.size(); ++j) {
        for (size_t k = 0; k < copy.size(); ++k) {
          params.push_back(
            std::make_tuple(block_size[i], block_size[j], copy[k]));
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

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatriXTests, Types);

TYPED_TEST(MatriXTests, Constructor) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A(m, n);

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(0, A(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, CopyConstructor) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(A(i, j), A_copy(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, CopyAssignment) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(m, n);

    A_copy = A;
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(A(i, j), A_copy(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, MoveConstructor) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy = A;

    Hatrix::Matrix<TypeParam> A_move(std::move(A));
    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(A_copy(i, j), A_move(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, MoveAssignment) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A), A_move(A);

    A_move = std::move(A);
    // Check result
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        EXPECT_EQ(A_copy(i, j), A_move(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, ViewAssignment) {
  for (auto const& [m, n] : this->dims) {
    for (auto const& k : this->block_size) {
      Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
      auto A_splits = A.split(4, 4);

      //TODO verify
      // A1 = A_splits[1];
      auto A1 = A_splits[1];

      for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          A1(i, j) = 0;
        }
      }

      for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
          EXPECT_EQ(A_splits[1](i, j), 0.0);
        }
      }
    }
  }
}

// TODO template better
TYPED_TEST(MatriXTests, ViewMoveAssignment) {
  int64_t block_size = 40, sub_block = 10;
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(block_size, block_size);
  auto A_splits = A.split(4, 4);
  A_splits[1] = Hatrix::generate_identity_matrix<TypeParam>(sub_block, sub_block);

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

TYPED_TEST(MatriXTests, shrink) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_shrunk(A);

    int64_t shrunk_size = 8;
    A_shrunk.shrink(shrunk_size, shrunk_size);

    // Check result
    for (int64_t i = 0; i < shrunk_size; ++i) {
      for (int64_t j = 0; j < shrunk_size; ++j) {
        EXPECT_EQ(A(i, j), A_shrunk(i, j));
      }
    }
  }
}

TYPED_TEST(MatriXTests, split) {
  for (auto const& [m, n] : this->dims) {
    for (auto const& [m_splits, n_splits, copy] : this->params) {
      Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);

      //TODO
      //std::vector<Hatrix::Matrix> split = A.split(m_splits, n_splits, copy);
      auto split = A.split(m_splits, n_splits, copy);

      // Check result
      int64_t m_start = 0, n_start = 0;
      for (int64_t i_c = 0; i_c < m_splits; ++i_c) {
        n_start = 0;
        for (int64_t j_c = 0; j_c < n_splits; ++j_c) {
          //TODO
          //Hatrix::Matrix& block = split[i_c * n_splits + j_c];
          auto block = split[i_c * n_splits + j_c];
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
  }
}

// TODO template better
TYPED_TEST(MatriXTests, shrinktest) {
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(64, 64);
  std::vector<int64_t> row_split_indices{6, 37, 55};
  std::vector<int64_t> col_split_indices{25, 41, 49};

  //TODO
  //std::vector<Hatrix::Matrix> split =
  //    A.split(row_split_indices, col_split_indices, false);

  auto split = A.split(row_split_indices, col_split_indices, false);

  // Check result
  int64_t m_start = 0, n_start = 0;
  for (int64_t i_c = 0; i_c < row_split_indices.size() + 1; ++i_c) {
    n_start = 0;
    for (int64_t j_c = 0; j_c < col_split_indices.size() + 1; ++j_c) {
      //TODO
      //Hatrix::Matrix& block = split[i_c * (col_split_indices.size()+1) + j_c];
      auto block = split[i_c * (col_split_indices.size()+1) + j_c];
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

TYPED_TEST(MatriXTests, uniform_split_copy) {
  int N = 100, Nslice = N / 2;
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(N, N);
  //TODO
  //std::vector<Hatrix::Matrix> A_splits = A.split(2, 2, false);
  auto A_splits = A.split(2, 2, false);
  Hatrix::Matrix<TypeParam> B = Hatrix::generate_identity_matrix<TypeParam>(Nslice, Nslice);

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

// TODO template better
TYPED_TEST(MatriXTests, non_uniform_square_split_copy) {
  int N = 40; int Nslice = 10; int split_dim = 30;
  std::vector<int64_t> split_vector = {split_dim};
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(N, N);
  //TODO
  //std::vector<Hatrix::Matrix> A_splits = A.split(split_vector, split_vector, false);
  auto A_splits = A.split(split_vector, split_vector, false);
  Hatrix::Matrix<TypeParam> B = Hatrix::generate_identity_matrix<TypeParam>(Nslice, Nslice);

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

TYPED_TEST(MatriXTests, non_uniform_rectangle_split_copy) {
  int64_t N = 40; int64_t Nslice = 10; int64_t split_dim = 30;
  std::vector<int64_t> y_split_vector = {split_dim};
  std::vector<int64_t> x_split_vector = {0};
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(N, N);
  //TODO
  //std::vector<Hatrix::Matrix> A_splits = A.split(x_split_vector, y_split_vector, false);
  auto A_splits = A.split(x_split_vector, y_split_vector, false);
  Hatrix::Matrix<TypeParam> B = Hatrix::generate_random_matrix<TypeParam>(N, Nslice);

  A_splits[3] = B;

  for (int i = 0; i < B.rows; ++i) {
    for (int j = 0; j < B.cols; ++j) {
      EXPECT_EQ(A(i, j + split_dim), B(i, j));
    }
  }
}

TYPED_TEST(MatriXTests, split_no_split) {
  int64_t N = 40;
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(N, N);
  //TODO
  //std::vector<Hatrix::Matrix> A_splits = A.split(std::vector<int64_t>(1, 0),
  //                                               std::vector<int64_t>(1, 0), false);
  auto A_splits = A.split(std::vector<int64_t>(1, 0),
                                                 std::vector<int64_t>(1, 0), false);
  // This should produce empty splits (A_splits[0], A_splits[1], A_splits[2])
  // And the full matrix in A_splits[3]
  for (int i = 0; i < A_splits[3].rows; ++i) {
    for (int j = 0; j < A_splits[3].cols; ++j) {
      EXPECT_EQ(A(i, j), A_splits[3](i, j));
    }
  }
}

TYPED_TEST(MatriXTests, split_null_at_the_end) {
  //RecordProperty("description",
  //               "Produce empty splits in location 1, 2 and 3. The full matrix is present in location 0.");
  int64_t N = 40;
  Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(N, N);
  //TODO
  //std::vector<Hatrix::Matrix> A_splits = A.split(std::vector<int64_t>(1, N),
  //                                               std::vector<int64_t>(1, N), false);
  auto A_splits = A.split(std::vector<int64_t>(1, N),
                                                 std::vector<int64_t>(1, N), false);
  for (int i = 0; i < A_splits[0].rows; ++i) {
    for (int j = 0; j < A_splits[0].cols; ++j) {
      EXPECT_EQ(A(i, j), A_splits[0](i, j));
    }
  }
}

TYPED_TEST(MatriXTests, split_vector_row_end) {
  int64_t N = 40;
  Hatrix::Matrix<TypeParam> V = Hatrix::generate_random_matrix<TypeParam>(N, 1);
  //TODO
  //std::vector<Hatrix::Matrix> V_splits = V.split({N/4, N/2, (3 * N) / 4}, {});
  auto V_splits = V.split({N/4, N/2, (3 * N) / 4}, {});

  EXPECT_EQ(V_splits.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(V_splits[i].rows, N/4);
    EXPECT_EQ(V_splits[i].cols, 1);
  }
}
