# Hatrix

Hatrix is meant to be a library for fast algorithms on structured dense matrices using the shared basis.

Hatrix lets you build your matrix factorization routines from a pre-defined set of easy-to-use routines that make minimum assumptions about how they are used. It makes sure that you, as the scientist, are fully in control of your programs.

# Basic Usage

Lets assume you want to generate and factorize a dense co-efficient matrix arising out of a Boundary Element Method for a 2D laplace Green's function on a uniform grid. The basic workflow for doing this is as follows:

1. Define a geometry using the `Domain` class.
2. Define a Green's function using the definition of `kernel_function`.
3. Write a construction routine for generation of symmetric H2-matrix using the `SymmetricSharedBasisMatrix` class.
4. Write a factorization routine.

Hatrix is useful for a large variety of programs as a result of its small and nimble interface. You can see various examples making use of the above workflow in some of the example files below:

| File name                      | Description                                                                                                          |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------|
| examples/BLR2\_strong\_CON.cpp | Construct a strongly admissible BLR2 matrix and apply a matrix vector product to check the accuracy of construction. |
|                                |                                                                                                                      |
|                                |                                                                                                                      |
|                                |                                                                                                                      |
|                                |                                                                                                                      |

# Important classes
