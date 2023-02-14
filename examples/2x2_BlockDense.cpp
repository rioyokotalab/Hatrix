#include <vector>
#include <iostream>
#include <cmath>

#include "Hatrix/Hatrix.h"

int main() {
    int n = 10;
    int block_size = 5;

    std::vector<std::vector<double>> randpts;
    randpts.push_back(Hatrix::equally_spaced_vector(2*n, 0.0, 1.0));

    Hatrix::Matrix D = Hatrix::generate_laplacend_matrix(randpts, n, n, 0, 0);

    for (int i=0; i<D.rows; ++i) {
      for(int j=0; j<D.cols; ++j) {
        std::cout<<D(i,j)<<" ";
      }
      std::cout<<std::endl;
    }
}