#pragma once

#include <cstdint>
#include <vector>

#include "Hatrix/classes/IndexedMap.hpp"
#include "Hatrix/classes/Matrix.hpp"

namespace Hatrix {
  enum class MatrixType { H2_MATRIX, BLR2_MATRIX };
  typedef struct SymmetricSharedBasisMatrix {
    int64_t min_level;                            // Min level of hierarchical subdivision (the level below root)
    int64_t max_level;                            // Max level of hierarchical subdivision (the leaf level)
    int64_t min_adm_level;                        // Min level that has admissible block(s)
    std::vector<int64_t> level_nblocks;           // Number of blocks within a row/column at each level
    ColLevelMap U;                                // Shared row/column bases and transfer matrices
    ColLevelMap Uc;                               // Complement of shared row/column bases and transfer matrices
    RowColLevelMap<Matrix> D;                     // Near coupling matrices (inadmissible dense blocks)
    RowColLevelMap<Matrix> S;                     // Far coupling matrices (of admissible blocks)
    RowColLevelMap<bool> is_admissible;           // Block-wise admissibility condition at each level
    RowColMap<std::vector<int64_t>> inadmissible_cols;  // Inadmissible column indices within a row at each level
    RowColMap<std::vector<int64_t>> admissible_cols;    // Admissible column indices within a row at each level
    RowLevelMap US_row;                           // Concatenation of all low-rank blocks in a row at each level
    RowColMap<std::vector<int64_t>> multipoles;   // Multipoles of each row/column at each level from ID
    RowLevelMap R_row;                            // Matrices produced from orthogonalization of ID bases

    int64_t max_rank();
    void print_structure();
    int64_t leaf_dense_blocks();
    double Csp(int64_t level);

    SymmetricSharedBasisMatrix();
    SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A); // deep copy

   private:
    void actually_print_structure(int64_t level);
  } SymmetricSharedBasisMatrix;
}
