#include <vector>
#include <iostream>
#include <cmath>

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  template <typename DT>
  RowColLevelMap<Matrix<DT>> create_map(const Matrix<DT>& A, int leaf_size, int num_blocks) {
    RowColLevelMap<Matrix<DT>> map;
    for (int i=0; i<num_blocks; ++i) {
      for (int j=0; j<num_blocks; ++j) {
        Matrix<DT> block_copy(leaf_size, leaf_size);
        for (int k=0; k<leaf_size; ++k) {
          for (int l=0; l<leaf_size; ++l) {
            block_copy(k,l) = A(i*leaf_size+k, j*leaf_size+l);
          }
        }
        map.insert(i, j, 0, std::move(block_copy));
      }
    }
    return map;
  }
}

int main() {
  int n = 10;
  int leaf_size = 5;
  int num_blocks = n / leaf_size;

  std::vector<std::vector<double>> randpts;
  randpts.push_back(Hatrix::equally_spaced_vector(2*n, 0.0, 1.0));

  Hatrix::Matrix D = Hatrix::generate_laplacend_matrix(randpts, n, n, 0, 0);
  Hatrix::Matrix D_copy(D, true);
  //D.print();
  Hatrix::RowColLevelMap<Hatrix::Matrix<double>> map = Hatrix::create_map(D, leaf_size, num_blocks);
  /*for (int i=0; i<num_blocks; i++) {
    for (int j=0; j<num_blocks; j++) {
      map(i,j,0).print();
    }
  }*/
  Hatrix::lu_nopiv(map(0,0,0));
  Hatrix::solve_triangular(map(0,0,0), map(0,1,0), Hatrix::Left, Hatrix::Lower, true, false);
  Hatrix::solve_triangular(map(0,0,0), map(1,0,0), Hatrix::Right, Hatrix::Upper, false, false);
  Hatrix::matmul(map(1,0,0), map(0,1,0), map(1,1,0), false, false, -1, 1);
  Hatrix::lu_nopiv(map(1,1,0));

  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix b_copy(b, true);

  std::vector<Hatrix::Matrix<double>> b_split = b.split(2, 1);
  
  std::cout<<b_split.size()<<std::endl;
  Hatrix::solve_triangular(map(1,1,0), b_split[1], Hatrix::Left, Hatrix::Upper, false, false);
  Hatrix::matmul(map(0,1,0), b_split[1], b_split[0], false, false, -1, 1);
  Hatrix::solve_triangular(map(0,0,0),b_split[0], Hatrix::Left, Hatrix::Upper, false, false);

  Hatrix::solve_triangular(map(0,0,0), b_split[0], Hatrix::Left, Hatrix::Lower, true, false);
  Hatrix::matmul(map(1,0,0), b_split[0], b_split[1], false, false, -1, 1);
  Hatrix::solve_triangular(map(1,1,0), b_split[1], Hatrix::Left, Hatrix::Lower, true, false);
  
  lu_nopiv(D_copy);
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Upper, false, false);
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Lower, true, false);

  double error = Hatrix::norm(b_copy - b);
  std::cout<<"Error: "<<error<<std::endl;

  b.print();
  std::cout<<std::endl;
  b_copy.print();
  return 0;
}