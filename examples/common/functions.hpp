#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <limits>

#include "Body.hpp"
#include "Cell.hpp"
#include "Domain.hpp"
#include "Hatrix/Hatrix.h"

namespace Hatrix {

using kernel_func_t =
    std::function<double(const Domain& domain,
                         const Body& source, const Body& target)>;

// Kernel constants
double PV = 1e-3;
double alpha = 1.0;

void set_kernel_constants(double _PV, double _alpha) {
  PV = _PV;
  alpha = _alpha;
}

kernel_func_t kernel_function;

void set_kernel_function(kernel_func_t _kernel_function) {
  kernel_function = _kernel_function;
}

double p2p_distance(const Body& source, const Body& target) {
  double r = 0;
  for (int64_t axis = 0; axis < 3; axis++) {
    r += (source.X[axis] - target.X[axis]) *
         (source.X[axis] - target.X[axis]);
  }
  return std::sqrt(r);
}

double laplace_kernel(const Domain& domain,
                      const Body& source, const Body& target) {
  const double r = p2p_distance(source, target) + PV;
  const double out = 1. / r;
  return out;
}

double yukawa_kernel(const Domain& domain,
                     const Body& source, const Body& target) {
  const double r = p2p_distance(source, target) + PV;
  const double out = std::exp(alpha * -r) / r;
  return out;
}

double ELSES_dense_input(const Domain& domain,
                         const Body& source, const Body& target) {
  const auto row = (int64_t)source.value;
  const auto col = (int64_t)target.value;
  return domain.p2p_matrix(row, col);
}

void ELSES_calc_db(const int64_t ja, double db[3]) {
  double dbx1 = 0, dby1 = 0, dbz1 = 0;
  switch (ja) {
    case 1: {
      dbx1 = 1.0;
      dby1 = 1.0;
      dbz1 = 1.0;
      break;
    }
    case 2: {
      dbx1 = -1.0;
      dby1 = -1.0;
      dbz1 = 1.0;
      break;
    }
    case 3: {
      dbx1 = 1.0;
      dby1 = -1.0;
      dbz1 = -1.0;
      break;
    }
    case 4: {
      dbx1 = -1.0;
      dby1 = 1.0;
      dbz1 = -1.0;
      break;
    }
  }
  const double ddd = dbx1*dbx1 + dby1*dby1 + dbz1*dbz1;
  const double dd = std::sqrt(ddd);
  db[0] = dbx1 / dd;
  db[1] = dby1 / dd;
  db[2] = dbz1 / dd;
}

double ELSES_carbon_kernel(const Domain& domain,
                           const Body& source, const Body& target) {
  double mat_val = 0;
  const auto ig = (int64_t)source.value + 1;  // Global row index within matrix
  const auto jg = (int64_t)target.value + 1;  // Global col index within matrix
  // Carbon model parameters
  const int64_t n_orb = 4;
  const auto js1 = (ig-1)/n_orb + 1;      // Atom number of source electron
  const auto ja1 = ((ig-1) % n_orb) + 1;  // Source electron number within atom (1-4)
  const auto js2 = (jg-1)/n_orb + 1;      // Atom number of target electron
  const auto ja2 = ((jg-1) % n_orb) + 1;  // Target electron number within atom (1-4)
  double ierr = 0;
  const double dnal0 = 2.38573234;  // n
  const double rnn0 = 1.42720722;   // r_0
  double dhal[5];  // V_{ss sigma} etc.
  dhal[1] = -5.0;
  dhal[2] = 4.7;
  dhal[3] = 5.5;
  dhal[4] = -1.55;
  double dnal[5];  // n_c (common)
  dnal[1] = 1.76325031;
  dnal[2] = 1.76325031;
  dnal[3] = 1.76325031;
  dnal[4] = 1.76325031;
  double rcal[5];  // r_c (common)
  rcal[1] = 8.04301633;
  rcal[2] = 8.04301633;
  rcal[3] = 8.04301633;
  rcal[4] = 8.04301633;
  const double ev4au = 2.0 * 13.6058;  // 1 au = (ev4au) eV
  const double angst = 0.529177;       // 1 au = 0.529177 A
  // No cut-off setting
  const double rcc = std::numeric_limits<double>::max() / 100.;  // r_m (cut-off distance) in a.u.
  const double es0 = -3.35 / ev4au;  // E_s
  const double ep0 =  3.35 / ev4au;  // E_p
  const double esp3a = 0.25 * es0 + 0.75 * ep0;
  const double esp3b = -0.25 * (ep0 - es0);
  const double qc0 = 1.6955223822 * 1e-2;   // c_0
  const double qc1 = -1.4135915717 * 1e-2;  // c_1
  const double qc2 = 7.2917027294 * 1e-6;   // c_2
  const double qc3 = 1.4516116860 * 1e-3;   // c_3
  // tail distance in A
  const double r_cut_tail = rcc * angst * 1.1; // r_1
  if (js1 == js2) {
    if (ig == jg) {
      mat_val = esp3a;
    }
    else {
      mat_val = esp3b;
    }
  }
  else {
    // Constants ax, ay, and az are taken from base carbon model
    // ELSES_mat_calc/make_supercell_C60_FCCs_w_noise/C60_fcc2x2x2_20220727.xyz
    // TODO Debug
    const double ax = 28.34;
    const double ay = 28.34;
    const double az = 28.34;
    double dxc = target.X[0] - source.X[0];
    double dyc = target.X[1] - source.X[1];
    double dzc = target.X[2] - source.X[2];
    dxc = dxc * ax;
    dyc = dyc * ay;
    dzc = dzc * az;
    double drr = std::sqrt(dxc*dxc + dyc*dyc + dzc*dzc);
    dxc = dxc / drr;
    dyc = dyc / drr;
    dzc = dzc / drr;
    double rnn = drr * 0.529177;  // Distance in [A]
    double dkwondd[5][3];  // Work array for TB parameters
    for (int64_t isym = 1; isym <= 4; isym++) {
      double dha = dhal[isym];
      double dna = dnal[isym];
      double rca = rcal[isym];

      double rat1 = rnn0 / rnn;
      double rat2 = rnn / rca;
      double rat3 = rnn0 / rca;
      double rat4 = dnal0 / rnn;

      double fac1 = std::pow(rat1, dnal0);
      double fac2 = std::pow(rat2, dna);
      double fac3 = std::pow(rat3, dna);
      double fac4 = 1.0 + dna * fac2;

      double dargexp = dnal0 * (-fac2 + fac3);
      double dexpon = std::exp(dargexp);
      dkwondd[isym][1] = dha * fac1 * dexpon;
      dkwondd[isym][2] = -dha * fac1 * dexpon * rat4 * fac4;
      if (rnn > r_cut_tail) {
        double dddx = rnn - r_cut_tail;
        double potij = qc0+qc1*dddx+qc2*dddx*dddx+qc3*dddx*dddx*dddx;
        double dphidr=qc1+2.0*qc2*dddx+3.0*qc3*dddx*dddx;
        dkwondd[isym][1] = dha * potij;
        dkwondd[isym][2] = dha * dphidr;
      }
    }
    // Slator-Koster parameters in au
    double dvss0 = dkwondd[1][1] / ev4au;
    double dvsp0 = dkwondd[2][1] / ev4au;
    double dvpp0 = dkwondd[3][1] / ev4au;
    double dvpp1 = dkwondd[4][1] / ev4au;

    // Calculate dbx, dby, and dbz
    double dbi[3], dbj[3];
    ELSES_calc_db(ja1, dbi);
    ELSES_calc_db(ja2, dbj);
    double dbx1 = dbi[0];
    double dby1 = dbi[1];
    double dbz1 = dbi[2];
    double dbx2 = dbj[0];
    double dby2 = dbj[1];
    double dbz2 = dbj[2];
    // Inner products
    double ad2 = dbx2*dxc + dby2*dyc + dbz2*dzc;  // inner product ( a_2 | d )
    double ad1 = dbx1*dxc + dby1*dyc + dbz1*dzc;  // inner product  ( a_1 | d )

    //  Vector :  a' = a - (ad) d
    double dbx1b = dbx1- ad1*dxc;
    double dby1b = dby1- ad1*dyc;
    double dbz1b = dbz1- ad1*dzc;

    double dbx2b = dbx2 - ad2*dxc;
    double dby2b = dby2 - ad2*dyc;
    double dbz2b = dbz2 - ad2*dzc;

    // double inner product : ( a_1 | d ) ( a_2 | d ) 
    double app0 = ad1*ad2;
    // inner product : ( a'_1 | a'_2 )
    double app1 = dbx1b*dbx2b + dby1b*dby2b + dbz1b*dbz2b;
    double aaa =
        dvss0+std::sqrt(3.0)*(ad2-ad1)*dvsp0+3.0*app0*dvpp0+3.0*app1*dvpp1;
    aaa = 0.25 * aaa;
    mat_val = aaa;
  }
  return mat_val;
}

Matrix generate_p2p_matrix(const Domain& domain,
                           const std::vector<int64_t>& source,
                           const std::vector<int64_t>& target,
                           const int64_t source_offset = 0,
                           const int64_t target_offset = 0) {
  const int64_t nrows = source.size();
  const int64_t ncols = target.size();
  Matrix out(nrows, ncols);
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = 0; j < ncols; j++) {
      out(i, j) = kernel_function(domain,
                                  domain.bodies[source_offset + source[i]],
                                  domain.bodies[target_offset + target[j]]);
    }
  }
  return out;
}

Matrix generate_p2p_matrix(const Domain& domain,
                           const int64_t row, const int64_t col,
                           const int64_t level) {
  const auto source_idx = domain.get_cell_idx(row, level);
  const auto target_idx = domain.get_cell_idx(col, level);
  const auto& source = domain.cells[source_idx];
  const auto& target = domain.cells[target_idx];

  return generate_p2p_matrix(domain, source.get_bodies(), target.get_bodies());
}

Matrix generate_p2p_matrix(const Domain& domain) {
  return generate_p2p_matrix(domain, 0, 0, 0);
}

// Source: https://github.com/scalable-matrix/H2Pack/blob/sample-pt-algo/src/H2Pack_build_with_sample_point.c
int64_t adaptive_anchor_grid_size(const Domain& domain,
                                  kernel_func_t kernel_function,
                                  const int64_t leaf_size,
                                  const double theta,
                                  const double ID_tol,
                                  const double stop_tol) {
  double L = 0;
  for (int64_t axis = 0; axis < domain.ndim; axis++) {
    const auto Xmin = Domain::get_Xmin(domain.bodies,
                                       domain.cells[0].get_bodies(), axis);
    const auto Xmax = Domain::get_Xmax(domain.bodies,
                                       domain.cells[0].get_bodies(), axis);
    L = std::max(L, Xmax - Xmin);
  }
  L /= 6.0; // Extra step taken from H2Pack MATLAB reference

  // Create two sets of points in unit boxes
  const auto box_nbodies = 2 * leaf_size;
  const auto shift = L * (theta == 0 ? 1 : theta);
  std::vector<Body> box1, box2;
  std::vector<int64_t> box_idx;
  box_idx.resize(box_nbodies);
  box1.resize(box_nbodies);
  box2.resize(box_nbodies);
  std::mt19937 gen(1234); // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(0, 1);
  for (int64_t i = 0; i < box_nbodies; i++) {
    box_idx[i] = i;
    for (int64_t axis = 0; axis < domain.ndim; axis++) {
      box1[i].X[axis] = L * dist(gen);
      box2[i].X[axis] = L * dist(gen) + shift;
    }
  }

  auto generate_matrix = [kernel_function, &domain]
                         (const std::vector<Body>& source,
                          const std::vector<Body>& target,
                          const std::vector<int64_t>& source_idx,
                          const std::vector<int64_t>& target_idx) {
    Matrix out(source_idx.size(), target_idx.size());
    for (int64_t i = 0; i < out.rows; i++) {
      for (int64_t j = 0; j < out.cols; j++) {
        out(i, j) = kernel_function(domain,
                                    source[source_idx[i]],
                                    target[target_idx[j]]);
      }
    }
    return out;
  };
  Matrix A = generate_matrix(box1, box2, box_idx, box_idx);
  // Find anchor grid size r by checking approximation error to A
  double box2_xmin[MAX_NDIM], box2_xmax[MAX_NDIM];
  for (int64_t i = 0; i < box_nbodies; i++) {
    for (int64_t axis = 0; axis < domain.ndim; axis++) {
      if (i == 0) {
        box2_xmin[axis] = box2[i].X[axis];
        box2_xmax[axis] = box2[i].X[axis];
      }
      else {
        box2_xmin[axis] = std::min(box2_xmin[axis], box2[i].X[axis]);
        box2_xmax[axis] = std::max(box2_xmax[axis], box2[i].X[axis]);
      }
    }
  }
  double box2_size[MAX_NDIM];
  for (int64_t axis = 0; axis < domain.ndim; axis++) {
    box2_size[axis] = box2_xmax[axis] - box2_xmin[axis];
  }
  int64_t r = 1;
  bool stop = false;
  while (!stop) {
    const auto box2_sample =
        Domain::select_sample_bodies(domain.ndim, box2, box_idx, r, 3, 0);
    // A1 = kernel(box1, box2_sample)
    Matrix A1 = generate_matrix(box1, box2, box_idx, box2_sample);
    Matrix U;
    std::vector<int64_t> ipiv_rows;
    std::tie(U, ipiv_rows) = error_id_row(A1, ID_tol, false);
    int64_t rank = U.cols;
    // A2 = A(ipiv_rows[:rank], :)
    Matrix A2(rank, A.cols);
    for (int64_t i = 0; i < rank; i++) {
      for (int64_t j = 0; j < A.cols; j++) {
        A2(i, j) = A(ipiv_rows[i], j);
      }
    }
    Matrix UxA2 = matmul(U, A2);
    const double error = norm(A - UxA2);
    if ((error < stop_tol) ||
        ((box_nbodies - box2_sample.size()) < (box_nbodies / 10))) {
      stop = true;
    }
    else {
      r++;
    }
  }
  return r;
}

Matrix prepend_complement_basis(const Matrix &Q) {
  Matrix Q_F(Q.rows, Q.rows);
  Matrix Q_full, R;
  std::tie(Q_full, R) =
      qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

  for (int64_t i = 0; i < Q_F.rows; i++) {
    for (int64_t j = 0; j < Q_F.cols - Q.cols; j++) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }
  for (int64_t i = 0; i < Q_F.rows; i++) {
    for (int64_t j = 0; j < Q.cols; j++) {
      Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
    }
  }
  return Q_F;
}

} // namespace Hatrix

