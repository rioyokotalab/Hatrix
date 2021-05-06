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
#include <iomanip>

namespace Hatrix {

void gesv_IR(Matrix &A, Matrix &b, int64_t max_iter){
  assert(A.rows == b.rows);
  assert(b.cols == 1);
  assert(max_iter > 0);

  Matrix A_orig(A);
  int64_t mdim = A.min_dim();
  Matrix L(A.rows, mdim), U(mdim, A.cols), P(A.rows, A.rows);
  lup(A, L, U, P);
  Matrix x = P * b;
  solve_triangular(L, x, Left, Lower, true);
  solve_triangular(U, x, Left, Upper, false);

  Matrix res;
  std::vector<double> nbe;
  std::vector<double> cbe;
  Matrix d (x.rows, 1);
  //x has only about 5 accurate digits when compared to matlab
  //for (int64_t i=0; i<b.rows; ++i)
    //std::cout<<std::setprecision(17)<<x(i,0)<<std::endl;
  //  std::cout<<std::setprecision(17)<<b(i,0)<<" - "<<test(i,0)<<" = "<<res(i,0)<<std::endl;

  for (int64_t i=0; i<max_iter; ++i){
    //compute residual
    res = b - (A_orig * x);
    nbe.push_back(norm_bw_error(res, A, x, b));
    cbe.push_back(comp_bw_error(res, A, x, b));

    double res_norm = infinity_norm(res);

    //solve for correction term
    //matmul integrates scaling
    matmul(P, res, d, false, false, 1/res_norm, 0);
    solve_triangular(L, d, Left, Lower, true);
    solve_triangular(U, d, Left, Upper, false);

    //for (int64_t i=0; i<d.rows; ++i)
    //  d(i, 0) = d(i, 0) * res_norm;
    scale(d, res_norm);
      //std::cout<<std::setprecision(17)<<d(i,0)<<std::endl;}
    //update solution
    Matrix x_old(x);
    x = x + d;
    double dx = infinity_norm(x-x_old) / infinity_norm(x);
    std::cout<<"DX: "<<std::setprecision(17)<<dx<<std::endl;

  }

  for (size_t i=0; i<nbe.size(); ++i)
    std::cout<<nbe.at(i)<<" ";
  std::cout<<std::endl;

  for (size_t i=0; i<cbe.size(); ++i)
    std::cout<<cbe.at(i)<<" ";
  std::cout<<std::endl;
}

} // namespace Hatrix
