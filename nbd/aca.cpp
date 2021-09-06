
#include "aca.h"
#include "kernel.h"

#include <lapacke.h>
#include <cblas.h>
#include <cmath>
#include <random>

using namespace nbd;

struct InputMat {
  int m;
  int n;
  const double* a;
  int lda;

  EvalFunc ef;
  const Cell* ci;
  const Cell* cj; 
  int dim;
};

inline void _eval(const InputMat* M, int y, int x, real_t* out) {
  if (M->a != NULL)
    *out = M->a[y + (size_t)M->lda * x];
  else
    eval(M->ef, M->ci->BODY + y, M->cj->BODY + x, M->dim, out);
}

void _daca(const InputMat* M, int max_iters, double* u, int ldu, double* v, int ldv, int* info) {

  int y = 0, x, iter = 1, m = M->m, n = M->n, piv_i = 0;
  double piv = 0;
  std::vector<int> p_u, p_v;

  max_iters = std::min(max_iters, m);
  max_iters = std::min(max_iters, n);
  
  p_u.reserve(max_iters);
  p_v.reserve(max_iters);

  for (int i = 0; i < n; i++) {
    real_t s;
    _eval(M, y, i, &s);
    v[i] = s;

    if (std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  x = piv_i;
  cblas_dscal(n, 1. / piv, v, 1);

  piv = 0.;
  for (int i = 0; i < m; i++) {
    real_t s;
    _eval(M, i, x, &s);
    u[i] = s;

    if (i > 0 && std::abs(s) > std::abs(piv))
    { piv_i = i; piv = s; }
  }

  p_u.push_back(0);
  p_v.push_back(x);

  double my_norm_u = cblas_ddot(m, u, 1, u, 1);
  double my_norm_v = cblas_ddot(n, v, 1, v, 1);
  double norm = my_norm_u * my_norm_v;
  double epi2 = ACA_EPI * ACA_EPI;
  double n2 = norm;

  if (ACA_USE_NORM && n2 <= epi2 * n2)
    piv_i = -1;

  while (piv_i != -1 && iter < max_iters) {
    double* curr_u = u + (size_t)iter * ldu;
    double* curr_v = v + (size_t)iter * ldv;
    y = piv_i;

    piv = 0.;
    for (int i = 0; i < n; i++) {
      real_t vi;
      _eval(M, y, i, &vi);

      double s = cblas_ddot(iter, u + y, ldu, v + i, ldv);
      vi -= s;
      curr_v[i] = vi;

      auto p = std::find(p_v.begin(), p_v.end(), i);

      if (p == p_v.end() && std::abs(vi) > std::abs(piv))
      { piv_i = i; piv = vi; }
    }

    x = piv_i;
    cblas_dscal(n, 1. / piv, curr_v, 1);

    p_u.push_back(y);
    p_v.push_back(x);

    piv = 0.;
    for (int i = 0; i < m; i++) {
      real_t ui;
      _eval(M, i, x, &ui);

      double s = cblas_ddot(iter, v + x, ldv, u + i, ldu);
      ui -= s;
      curr_u[i] = ui;

      auto p = std::find(p_u.begin(), p_u.end(), i);

      if (p == p_u.end() && std::abs(ui) > std::abs(piv))
      { piv_i = i; piv = ui; }
    }

    if (ACA_USE_NORM) {
      for (int j = 0; j < iter; j++) {
        my_norm_u = cblas_ddot(m, u + (size_t)j * ldu, 1, curr_u, 1);
        my_norm_v = cblas_ddot(n, v + (size_t)j * ldv, 1, curr_v, 1);

        norm += 2. * my_norm_u * my_norm_v;
      }
      my_norm_u = cblas_ddot(m, curr_u, 1, curr_u, 1);
      my_norm_v = cblas_ddot(n, curr_v, 1, curr_v, 1);

      n2 = my_norm_u * my_norm_v;
      if (n2 <= epi2 * (norm += n2))
        piv_i = -1;
    }
    iter++;
  }

  if (info)
    *info = (int)iter;
}

void nbd::daca_cells(EvalFunc ef, const Cell* ci, const Cell* cj, int dim, int max_iters, double* u, int ldu, double* v, int ldv, int* info) {
  InputMat M{ ci->NBODY, cj->NBODY, NULL, 0, ef, ci, cj, dim };
  _daca(&M, max_iters, u, ldu, v, ldv, info);
}

void nbd::daca(int m, int n, int max_iters, const double* a, int lda, double* u, int ldu, double* v, int ldv, int* info) {
  EvalFunc ef{ NULL, 0., 0. };
  InputMat M{ m, n, a, lda, ef, NULL, NULL, 0 };
  _daca(&M, max_iters, u, ldu, v, ldv, info);
}


inline real_t rand(real_t min, real_t max) {
  return min + (max - min) * ((double)std::rand() / RAND_MAX);
}

void nbd::ddspl(int m, int na, const double* a, int lda, int ns, double* s, int lds) {
  std::vector<double> rnd((size_t)na * ns);

  for (auto& i : rnd)
    i = rand(-1, 1);
    
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, ns, na, 1., a, lda, &rnd[0], na, 1., s, lds);
}

void nbd::drspl(int m, int na, int r, const double* ua, int ldu, const double* va, int ldv, int ns, double* s, int lds) {
  std::vector<double> rnd((size_t)ns * na);
  std::vector<double> work((size_t)ns * r);

  for (auto& i : rnd)
    i = rand(-1, 1);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ns, r, na, 1., &rnd[0], ns, va, ldv, 0., &work[0], ns);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, ns, r, 1., ua, ldu, &work[0], ns, 1., s, lds);
}

void nbd::dorth(int m, int n, double* a, int lda, double* r, int ldr) {
  std::vector<double> tau(std::min(m, n));
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, &tau[0]);
  if (r != nullptr)
    for (int i = 0; i < n; i++)
      cblas_dcopy(std::min(i + 1, m), a + i * lda, 1, r + i * ldr, 1);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, a, lda, &tau[0]);
}

void nbd::dmul_ut(int m, int n, int k, const double* u, int ldu, const double* a, int lda, double* c, int ldc) {
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k, 1., u, ldu, a, lda, 0., c, ldc);
}

void nbd::dmul_s(int m, int n, int k, const double* u, int ldu, const double* v, int ldv, double* s, int lds) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, k, 1., u, ldu, v, ldv, 0., s, lds);
}
