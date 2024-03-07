#pragma once

// #include "Domain.hpp"

namespace Hatrix {
  // Algorithms that can be used for calculation of admissiblity.
  //
  // Use DUAL_TREE_TRAVERSAL for a geometry-based admissibilty condition and DIAGONAL_ADMIS
  // for a distance-from-the-diagonal admissibility condition.
  enum ADMIS_ALGORITHM {
    DUAL_TREE_TRAVERSAL,
    DIAGONAL_ADMIS
  };

  // Type for defining a symmetric matrix using shared basis. Can use nested bases.
  typedef struct SymmetricSharedBasisMatrix {
    // The minimmum level of the shared bases matrix. If using nested bases, this will
    // be the level that does not contain any admissible blocks. If not using nested bases,
    // this will be one less that the max_level.
    int64_t min_level;

    // The maximmum level of the shared bases matrix. The max level when using a binary
    // tree for classification of the data will be log2(N / nleaf). If using a non-binary
    // tree, this value can be set by the user.
    int64_t max_level;

    // A hashmap that represents the row bases of each level of the shared bases using a
    // 2-tuple of <row, level> to denote a single row basis 'U'. The value of each matrix
    // is of type Hatrix::Matrix.
    //
    // This type is defined in IndexedMap.hpp.
    ColLevelMap U;

    // A hashmap that represents the dense blocks at the leaf level of the shared bases
    // matrix using a 3-tuple of <row, col, level>. The value of each matrix is of type
    // Hatrix::Matrix.
    //
    // This type is defined in IndexedMap.hpp.
    RowColLevelMap<Matrix> D;

    // A hashmap that represents the magnitudes of the bases at each level of
    // the shared bases matrix using a 3-tuple of <row, col, level>. The value
    // of each matrix is of type Hatrix::Matrix. Each block 'S' denotes the
    // magnitudes of the bases at the row 'row', column 'col' and level 'level'.
    //
    // This type is defined in IndexedMap.hpp.
    RowColLevelMap<Matrix> S;

    // A hashmap that represents whether a given block in the symmetric shared
    // bases matrix is admissible or inadmissible. Each block is represented
    // as a 3-tuple of <row, col, level>. Admissible blocks are denoted as
    // 'true' values, and inadmissible values are 'false'. If a tuple at a
    // given level does not exist, it means that the block is part of a larger
    // block in the multi-level matrix structure.
    RowColLevelMap<bool> is_admissible;

    // Traverse all the bases matrices and return the max rank.
    int64_t max_rank();

    // Print the structure of the admissible and inadmissible blocks of the matrix.
    void print_structure();

    // Return the number of dense blocks at the leaf level of the symmetric shared
    // bases matrix.
    int64_t leaf_dense_blocks();

    // Return the ratio of the average number of dense blocks per row to the total
    // number of size of the longest row at the leaf level.
    double Csp(int64_t level);

    // Initialize an instance with default values.
    SymmetricSharedBasisMatrix();

    // Copy constructor for performing a deep copy.
    SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A);

    // Populate the is_admissible hashmap using the algorithm given by admis_algorithm.
    //
    // use_nested_basis: If true, generate a shared bases matrix using the nested bases,
    // which will result in the generation of an HSS or H2-matrix depending on the
    // condition of admissibility.
    // admis_algorithm: Decide the algorithm to use for generating the condition
    // of admissibility.
    // admis: Control which two blocks at each level are determined as admissible
    // or inadmissible. If the distance between the centers of two blocks is denoted
    // by 'distance' and their sizes by ci_size and cj_size, the well-separatedness
    // of the blocks is determined by distance >= ((ci_size + cj_size) * admis).
    void generate_admissibility(const Domain& domain,
                                const bool use_nested_basis,
                                const ADMIS_ALGORITHM admis_algorithm,
                                const double admis);

  private:
    void actually_print_structure(int64_t level);
    void dual_tree_traversal(const Domain& domain,
                             const int64_t Ci_index,
                             const int64_t Cj_index,
                             const bool use_nested_basis,
                             const double admis);
    void compute_matrix_structure(int64_t level,
                                  const double admis);

  } SymmetricSharedBasisMatrix;
}
