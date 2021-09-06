
#include "h2mv.h"
#include "kernel.h"

#include <cblas.h>

using namespace nbd;

void nbd::upwardPass(const Cell* jcell, const Matrix* base, const real_t* x, Matrix* m) {

  if (jcell->NCHILD == 0) {
    if (base->N > 0) {
      *m = Matrix(base->N, 1, base->N);
      cblas_dgemv(CblasColMajor, CblasTrans, base->M, base->N, 1., base->A.data(), base->LDA, x, 1, 0., m->A.data(), 1);
    }
    return;
  }

  int x_off = 0;
  for (auto c = jcell->CHILD; c != jcell->CHILD + jcell->NCHILD; c++) {
    auto i = c - jcell;
    upwardPass(c, base + i, x + x_off, m + i);
    x_off += c->NBODY;
  }

  if (base->N > 0) {
    *m = Matrix(base->N, 1, base->N);
    double beta = 0.;
    x_off = 0;
    for (auto c = jcell->CHILD; c != jcell->CHILD + jcell->NCHILD; c++) {
      auto i = c - jcell;
      cblas_dgemv(CblasColMajor, CblasTrans, m[i].M, base->N, 1., base->A.data() + x_off, base->LDA, m[i].A.data(), 1, beta, m->A.data(), 1);
      beta = 1.;
      x_off += m[i].M;
    }
  }

}


void nbd::horizontalPass(const Cells& icells, const Cells& jcells, const Matrices& d, const Matrices& m, Matrices& l) {
  int ld = (int)icells.size();

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    double beta = 0.;
    for (auto& j : i.listFar) {
      auto x = j - &jcells[0];
      const Matrix& s = d[y + x * ld];
      if (l[y].M == 0)
        l[y] = Matrix(s.M, 1, s.M);
      cblas_dgemv(CblasColMajor, CblasNoTrans, s.M, s.N, 1., s.A.data(), s.LDA, m[x].A.data(), 1, beta, l[y].A.data(), 1);
      beta = 1.;
    }
  }
}


void nbd::downwardPass(const Cell* icell, const Matrix* base, Matrix* l, real_t* b) {

  if (icell->NCHILD == 0) {
    if (base->N > 0)
      cblas_dgemv(CblasColMajor, CblasNoTrans, base->M, base->N, 1., base->A.data(), base->LDA, l->A.data(), 1, 1., b, 1);
    return;
  }

  int b_off = 0, u_off = 0;
  for (auto c = icell->CHILD; c != icell->CHILD + icell->NCHILD; c++) {
    auto i = c - icell;
    double beta = 1.;
    if (base->N > 0 && l->M > 0) {
      if (l[i].M == 0) {
        l[i] = Matrix(base[i].N, 1, base[i].N);
        beta = 0.;
      }
      cblas_dgemv(CblasColMajor, CblasNoTrans, l[i].M, base->N, 1., base->A.data() + u_off, base->LDA, l->A.data(), 1, beta, l[i].A.data(), 1);
    }
    downwardPass(c, base + i, l + i, b + b_off);
    u_off += l[i].M;
    b_off += c->NBODY;
  }
}

void nbd::closeQuarter(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& d, const real_t* x, real_t* b) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      mvec_kernel(ef, &icells[y], &jcells[_x], dim, 1., x + xi, 1, 1, b + yi, 1);
    }
  }
}


void nbd::h2mv_complete(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, const Matrices& ibase, const Matrices& jbase, const Matrices& d, const real_t* x, real_t* b) {

  Matrices m(jcells.size()), l(icells.size());

  std::fill(b, b + icells[0].NBODY, 0.);
  upwardPass(&jcells[0], &jbase[0], x, &m[0]);
  horizontalPass(icells, jcells, d, m, l);
  downwardPass(&icells[0], &ibase[0], &l[0], b);
  closeQuarter(ef, icells, jcells, dim, d, x, b);
}

