#pragma once
#include <string>

#include "internal_types.hpp"

namespace Hatrix {
  class Args {
  private:
    std::string geometry_to_string(KIND_OF_GEOMETRY geometry);
    std::string admis_kind_to_string(ADMIS_KIND admis_kind);
    std::string construct_algorithm_to_string(CONSTRUCT_ALGORITHM construct_algorithm);
  public:
    int64_t N;                      // problem size
    int64_t nleaf;                  // leaf size
    kernel_function kernel;         // kernel function for matrix generation
    std::string kernel_verbose;     // name of the kernel function
    KIND_OF_GEOMETRY kind_of_geometry;
    std::string geometry_file;      // if the geometry is to be read from a file.
    int64_t ndim;
    int64_t max_rank;
    double admis;                   // admissibility value
    double perturbation;            // constant added to admissibility
    double accuracy;                // desired construction accuracy
    double qr_accuracy;             // cut off accuracy for QR factorization
    // 0 - accuracy truncated, 1 - lapack constant rank QR, 2 - constant rank no pivot, 3 - fixed rank svd
    int kind_of_recompression;
    double add_diag;
    ADMIS_KIND admis_kind;      // whether this is diagonal or geometry based admis
    CONSTRUCT_ALGORITHM construct_algorithm;
    bool use_nested_basis;      // whether to use nested basis in construction
    bool verbose;             // whether execution is verbose.
    bool is_symmetric;
    int num_args;               // number of arguments actually passed by the user.

    Args(int argc=0, char **argv=NULL);
    void usage(const char *name);
    void show();
  };
}
