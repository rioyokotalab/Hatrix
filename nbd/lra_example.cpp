
#include "aca.h"
#include "test_util.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

using namespace nbd;

void compress_using_aca(int m, int n, int r, double* a, int lda);

int main(int argc, char* argv[]) {

  int r = argc > 1 ? atoi(argv[1]) : 16;
  int m = argc > 2 ? atoi(argv[2]) : 32;
  int n = argc > 3 ? atoi(argv[3]) : m;

  r = std::min(r, m);
  r = std::min(r, n);

  std::srand(199);
  std::vector<double> left(m * r), right(n * r);

  for(auto& i : left)
    i = ((double)std::rand() / RAND_MAX) * 100;
  for(auto& i : right)
    i = ((double)std::rand() / RAND_MAX) * 100;

  std::vector<double> a(m * n);

  for(int j = 0; j < n; j++) {
    for(int i = 0; i < m; i++) {
      double e = 0.;
      for(int k = 0; k < r; k++)
        e += left[i + k * m] * right[j + k * n];
      a[i + j * m] = e;
    }
  }

  compress_using_aca(m, n, r, a.data(), m);

  return 0;
}


void compress_using_aca(int m, int n, int r, double* a, int lda) {
  int rp = r + 8;
  std::vector<double> u(m * rp), v(n * rp), b(m * n);

  int iters;
  daca(m, n, rp, a, lda, u.data(), m, v.data(), n, &iters);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double e = 0.;
      for (int k = 0; k < iters; k++)
        e += u[i + k * m] * v[j + k * n];
      b[i + j * lda] = e;
    }
  }

  printf("aca rel err: %e, aca iters %d\n", rel2err(b.data(), a, m, n, m, m), iters);
}

