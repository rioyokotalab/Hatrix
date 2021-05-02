#include "Hatrix/classes/Matrix.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/util/norms.h"
#include "Hatrix/util/errors.h"

#include <cassert>
#include <cstdint>
using std::int64_t;
#include <vector>
#include <iostream>


namespace Hatrix {

void gesv_IR(Matrix &A, Matrix &b, int64_t max_iter){
  assert(A.rows == b.rows);

  int64_t mdim = A.min_dim();
  Matrix L(A.rows, mdim), U(mdim, A.cols), P(A.rows, A.rows);
  lup(A, L, U, P);
  Matrix x = P * b;
  solve_triangular(L, x, Left, Lower, true);
  solve_triangular(U, x, Right, Upper, false);
  
  Matrix res;
  std::vector<double> nbe;
  std::vector<double> cbe;

  for (int64_t i=0; i<max_iter; ++i){
    //compute residual
    res = b - (A * x);
    nbe.push_back(norm_bw_error(res, A, x, b));
    cbe.push_back(comp_bw_error(res, A, x, b));

    double res_norm = infinity_norm(res);
    for (int64_t i=0; i<res.rows; ++i)
      res(i, 0) = res(i, 0) / res_norm;

    //solve for correction term
    Matrix d = P * res;
    solve_triangular(L, d, Left, Lower, true);
    solve_triangular(U, d, Right, Upper, false);

    for (int64_t i=0; i<d.rows; ++i)
      d(i, 0) = d(i, 0) * res_norm;

    //update solution
    Matrix x_old(x);
    x = x + d;
    double dx = infinity_norm(x-x_old) / infinity_norm(x);

  }

  for (size_t i=0; i<nbe.size(); ++i)
    std::cout<<nbe.at(i)<<" ";
  std::cout<<std::endl;

  for (size_t i=0; i<cbe.size(); ++i)
    std::cout<<cbe.at(i)<<" ";
  std::cout<<std::endl;
}

} // namespace Hatrix
