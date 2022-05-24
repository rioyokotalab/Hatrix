#pragma once

extern "C" {
  /* Cblacs declarations */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int*, const char*, int, int);
  void Cblacs_gridinfo(int, int*, int*, int*, int*);
  void Cblacs_pcoord(int, int, int*, int*);
  void Cblacs_gridexit(int);
  void Cblacs_barrier(int, const char*);

  int numroc_(int*, int*, int*, int*, int*);

  void descinit_(int *desc, const int *m,  const int *n, const int *mb,
    const int *nb, const int *irsrc, const int *icsrc, const int *ictxt,
    const int *lld, int *info);
  void pdgetrf_(
                int *m, int *n, double *a, int *ia, int *ja, int *desca,
                int *ipiv,int *info);
  int indxg2l_(int* , int*, int*, int*, int*);
  int indxg2p_(int*, int*, int*, int*, int*);
}
