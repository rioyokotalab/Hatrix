#include "internal_types.hpp"

#include <string>

namespace Hatrix {
  class Args {
  private:
    std::string geometry_to_string(KIND_OF_GEOMETRY geometry);
    std::string admis_kind_to_string(ADMIS_KIND admis_kind);
    std::string construct_algorithm_to_string(CONSTRUCT_ALGORITHM construct_algorithm);
  public:
    int64_t N;                  // problem size
    int64_t nleaf;              // leaf size
    kernel_function kernel;    // kernel function for matrix generation
    std::string kernel_verbose; // name of the kernel function
    KIND_OF_GEOMETRY kind_of_geometry;
    int64_t ndim;
    int64_t max_rank;
    double admis;               // admissibility value
    double accuracy;            // desired construction accuracy
    double add_diag;
    ADMIS_KIND admis_kind;      // whether this is diagonal or geometry based admis
    CONSTRUCT_ALGORITHM construct_algorithm;
    bool use_nested_basis;      // whether to use nested basis in construction
    bool verbose;             // whether execution is verbose.
    bool is_symmetric;

    Args(int argc=0, char **argv=NULL);
    void usage(const char *name);
    void show();
  };
}
