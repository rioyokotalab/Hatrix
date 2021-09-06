

H\H2-matrix Headers:
nbd.h \ kernel.h \ aca.h \ build_tree.h \ h2mv.h

Testing Headers:
test_util.h \ timer.h

Namespace: nbd

Example usage:

Step 1, Initialize Bodies of Charge:

Bodies b;
for(auto &i : b)
  b.X = ...;

Step2, Build a tree of cells from bodies of charge:

Cells c = build_tree(b, LEAF_SIZE, B_DIM);
// building tree permutes points using Z-curve and binary partition,
// so that geometrically closer points are placed nearer in Bodies vector.

Step3, build an H-matrix from one tree(symmetric), or two trees(non-symmetric) of cells,
from dual-tree traversal.

eval_func_t fun = l3d(); // using our predefined laplace-3d kernel for evaluation.
Matrices d; // stores interactions
traverse(fun, c1, c2, BDIM, d, THETA, RANK); // traverse(fun, c1, c1, BDIM, d, THETA, RANK) for symmetric

// d has size c1.size * c2.size. For any interaction between c1_i and c2_j,
// interaction is stored in d_(i + c1.size * j).
// interaction matrix can be either dense (m x n full), or a Low-rank approximated
// U (m x r) *VT (n x r) matrix.

// THETA (above 0.) determines admissibility, smaller THETA makes admissibility stronger.
// RANK determines max iteration for ACA Low-rank approximation, larger rank makes it more accurate,
// as well as make ACA run slower.

Step4, if an H2-matrix representation is wanted, continue to sample from,
the constructed H-matrix and convert it into a H2-matrix, by traversing the individual trees.

Matrices bi, bj; // basis matrices.
p = 20; // oversampling parameter: basis rank = max(LR.rank) + p.

traverse_i(c1, c2, BDIM, bi, p); // sampling on the i_side.
// [traverse_j(c1, c2, BDIM, bj, p);] for non_symmetric j.
shared_epilogue(d); // convert the low-rank represented matrices to store a single S matrix.

// Basis for c1_i is stored inside bi_i. Same for c2_j and bj_j.
// Increasing oversampling makes shared basis more accurately reflecting the contents,
// but with the cost of more computation.
// The combination of Bodies, Cells c1\c2, Basis Matrices bi\bj, Direct Interaction d makes up the H2-matrix.

Step5, from an H2-matrix representation, do H2-vector multiplication Ax = b.

h2mv_complete(c1, c2, bi, bj, d, &x[0], &b[0]); // Link to Bodies are stored inside cells.

// A * x, Answer is stored in b.
