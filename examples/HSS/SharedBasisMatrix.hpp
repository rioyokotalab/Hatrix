#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {

  class SharedBasisMatrix {
  private:
    void coarsen_blocks(int64_t level);
    void calc_diagonal_based_admissibility(int64_t level);
    Matrix get_Ubig(int64_t node, int64_t level);
    Matrix get_Vbig(int64_t node, int64_t level);
  public:
    int64_t N, nleaf, rank;
    double accuracy;
    double admis;
    ADMIS_KIND admis_kind;
    CONSTRUCT_ALGORITHM construct_algorithm;
    bool use_shared_basis;
    const Domain domain;
    const kernel_function kernel;

    int64_t height;
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    std::vector<int64_t> level_blocks;

    int64_t get_block_size(int64_t parent, int64_t level);
    SharedBasisMatrix(int64_t N, int64_t nleaf, int64_t rank, double accuracy,
                      double admis, ADMIS_KIND admis_kind,
                      CONSTRUCT_ALGORITHM construct_algorithm, bool use_shared_basis,
                      const Domain& domain, const kernel_function& kernel);

    // Obtain construction error w.r.t dense matrix with matvec.
    double construction_error();
    Matrix matvec(const Matrix& x);
  };

  // Strategy pattern class hierarchy for implementation of various
  // construction schemes.
  class ConstructAlgorithm {
  public:
    SharedBasisMatrix *context;
    ConstructAlgorithm(SharedBasisMatrix* context);
    virtual void construct() = 0;
  };

  // Construct HSS matrix using the algorithm written in the miro board.
  class ConstructMiro : public ConstructAlgorithm {
  private:
    Matrix generate_row_block(int64_t block, int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_row_bases(int64_t block, int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_column_bases(int64_t block, int64_t block_size, int64_t level);
    Matrix generate_column_block(int64_t block, int64_t block_size, int64_t level);
    void generate_leaf_nodes(const Domain& domain);

    bool row_has_admissible_blocks(int64_t row, int64_t level);
    bool col_has_admissible_blocks(int64_t col, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                              int64_t block_size, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                                              int64_t block_size, int64_t level);
    std::tuple<RowLevelMap, ColLevelMap>
    generate_transfer_matrices(int64_t level, RowLevelMap& Uchild, ColLevelMap& Vchild);

  public:
    ConstructMiro(SharedBasisMatrix* context);
    void construct();
  };

  class ConstructID_Random : public ConstructAlgorithm {
  public:
    ConstructID_Random(SharedBasisMatrix* context);
    void construct();
  };
}
