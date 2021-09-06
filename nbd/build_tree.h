
#pragma once
#include "nbd.h"

namespace nbd {

#define PART_EQ_SIZE

  Cells buildTree(Bodies& bodies, int ncrit, int dim);

  void getList(Cell * Ci, Cell * Cj, int dim, real_t theta, bool symm);

  void evaluate(EvalFunc ef, Cells& cells, const Cell* jcell_start, int dim, Matrices& d, int rank);

  void traverse(EvalFunc ef, Cells& icells, Cells& jcells, int dim, Matrices& d, real_t theta, int rank);

  void sample_base_i(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p);

  void sample_base_j(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p);

  void sample_base_recur(Cell* cell, Matrix* base);

  void shared_base_i(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, bool symm);

  void shared_base_j(Cells& icells, Cells& jcells, Matrices& d, Matrices& base);

  void nest_base(Cell* icell, Matrix* base);

  void traverse_i(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p);

  void traverse_j(Cells& icells, Cells& jcells, Matrices& d, Matrices& base, int p);

  void shared_epilogue(Matrices& d);

  int lvls(const Cell* cell, int* lvl);

  Cells getLeaves(const Cells& cells);

}

