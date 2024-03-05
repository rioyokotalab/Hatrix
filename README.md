# Hatrix

Hatrix is meant to be a library for fast algorithms on structured dense matrices using the shared basis.

Hatrix lets you build fast matrix routines from a pre-defined set of easy-to-use routines with a focus on customizatibility. The minimal building blocks provided by Hatrix make sure that the user is in full control of their algorithms.

# Basic Usage

Use the `Domain` class to generate a square with uniformly spaced points along the boundary.
Then sort
``` cpp
Hatrix::Domain domain(N, ndim);
domain.generate_grid_particles();
domain.cardinal_sort_and_cell_generation(leaf_size);
```

Generate a dense matrix using a 2D laplace function provided under the `greens_functions` namespace.
``` cpp
double diagonal_constant = 1e-6;
Hatrix::greens_functions::kernel_function_t kernel;
kernel = [&](const std::vector<double>& c_row,
    const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, diagonal_constant);
};
```

Generate a random matrix vector for verificiation using a matrix generator and multiply it with
the dense matrix for verification.
``` cpp
Hatrix::Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
```

Use the `SymmetricSharedBasisMatrix` type for representing a symmetric shared basis matrix.
Initilialize the number of levels, and setup conditions of admissibility using the dual
tree traversal algorithm.
```
```

Generate an H2-matrix with strong admissibility as shown in the `H2_strong_CON.cpp` file
using the `construct_H2_strong` function, and perform a matrix-vector multiplication with
the previously generated random vector `x`.
```
```


# Example files
