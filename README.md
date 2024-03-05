# Hatrix

Hatrix is meant to be a library for fast algorithms on structured dense matrices using the shared basis.

Hatrix lets you build fast matrix routines from a pre-defined set of easy-to-use routines with a focus on customizatibility. The minimal building blocks provided by Hatrix make sure that the user is in full control of their algorithms.

# Basic Usage

Use the `Domain` class to generate a square with uniformly spaced points along the boundary.
Then sort
``` cpp
int N = 1024;
int ndim = 2;
Hatrix::Domain domain(N, ndim);
domain.generate_grid_particles();
domain.cardinal_sort_and_cell_generation(leaf_size);
```

Define a 2D laplace green's function that will work with the grid in the domain as a
solution for the boundary value integral.
``` cpp
double diagonal_constant = 1e-6;
Hatrix::greens_functions::kernel_function_t kernel;
kernel = [&](const std::vector<double>& c_row,
             const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, diagonal_constant);
};
```

Generate a dense matrix using a 2D laplace function provided under the `greens_functions` namespace.
``` cpp
Hatrix::Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
```

Generate a random matrix vector for verificiation using a matrix generator and multiply it with
the dense matrix for verification.
``` cpp
Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);
Hatrix::Matrix b_dense = Hatrix::matmul(A_dense, x);
```

Use the `SymmetricSharedBasisMatrix` type for representing a symmetric shared basis matrix.
Initilialize the number of levels, and setup conditions of admissibility using the dual
tree traversal algorithm. Passing `true` into `generate_admissibility()` will use the nested
bases.
``` cpp
const double admis = 0.5;
const int leaf_size = 64;
Hatrix::SymmetricSharedBasisMatrix A;
A.max_level = log2(N / leaf_size);
A.generate_admissibility(domain, true,
    Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);
```

Generate an H2-matrix with strong admissibility as shown in the `H2_strong_CON.cpp` file
using the `construct_H2_strong` function, and perform a matrix-vector multiplication with
the previously generated random vector `x`.
``` cpp
const int max_rank = 30;
construct_H2_strong(A, domain, N, leaf_size, max_rank, 0.0);
Hatrix::Matrix b_lowrank = matmul(A, x, N, max_rank);
```

Verify the construction by the comparing the matvec by subtracting the vectors and
calculating the relative error between the norm of the difference and the original
vector.
``` cpp
double rel_error = Hatrix::norm(b_dense - b_lowrank) / Hatrix::norm(b_dense);
```

# Compiling and building

We use cmake for compiling.

# Example files

Usage of the library can be learnt from various
