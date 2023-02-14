#include <vector>
#include <iostream>
#include <cmath>

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  template <typename DT = double>
  class HODLR {
    private:
      RowColLevelMap<Matrix<DT>> D;
      RowColLevelMap<LowRank<DT>> LR;
      RowColLevelMap<bool> is_admissible;
      RowColLevelMap<bool> is_leaf;
      int max_level = 2;

      set_pos_admis(int row = 0, int col = 0, int level = 0) {
        if (level < max_level) {
          is_admissible.insert(row, col, level, row != col);
          set_admis(row, col, level + 1);
          set_admis(row, col + 1, level + 1);
          set_admis(row + 1 , col, level + 1);
          set_admis(row + 1, col + 1, level + 1);
        }
      }

      void position_based_admis() {
        set_pos_admis();
      }

      void construct(int n, int row = 0, int col = 0, int level = 0) {
        n = (n + 1) / 2;
        if (is_admissible(i, j, l)) {
            Hatrix::generate_laplacend_matrix(randpts, n, n, i*n, j*n)
          return;
        }
        if (n < leaf_size) {
          
          return;
        }
        construct (n, row, col, level + 1);
        construct (n, row, col + 1, level + 1);
        construct (n, row + 1, col, level + 1);
        construct (n, row + 1, col + 1, level + 1);
    
      }

    public:
      void HODLR() {
        position_based_admis();
        for (int l = 1; l < max_level; ++l) {
          //(n+n_splits-1) / n_splits;
          int n_splits = l * 2;
          int num_rows = A.rows / (l*2);
          int num_cols = A.cols / (l*2);
          for (int i = 0; i < l*2 ; ++i) {
            for (int j = 0; j < l*2 ; ++j) {

              if is_admissible(i, j, l) {
                LR.insert(i, j, l, );
              }
              else{
                // is leaf?
                is_leaf.insert(i, j, l, true);
                D.insert(i, j, l, );
              }
    }
  };
}