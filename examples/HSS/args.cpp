#include "args.hpp"

#include <string>
#include <getopt.h>

namespace Hatrix {
  static struct option long_options[] = {
    {"N", required_argument, 0, 'n'},
    {"nleaf", required_argument, 0, 'l'},
    {"kernel_func", required_argument, 0, 'k'},
    {"kind_of_geometry", required_argument, 0, 'g'},
    {"ndim", required_argument, 0, 'd' },
    {"max_rank", required_argument, 0, 'r'},
    {"accuracy", required_argument, 0, 'e'},
    {"admis", required_argument, 0, 'a'},
    {"admis_kind", required_argument, 0, 'm'},
    {"construct_algorithm", required_argument, 0, 'c'},
    {"add_diag", required_argument, 0, 'z'},
    {"nested_basis", required_argument, 0, 'b'},
    {"verbose", required_argument, 0, 'v'}
  };

  std::string
  Args::kernel_func_to_string(KERNEL_FUNC kernel_func) {
    switch(kernel_func) {
    default:
      return std::string("laplace");
    }
  }

  std::string
  Args::geometry_to_string(KIND_OF_GEOMETRY geometry) {
    switch(geometry) {
    case CIRCULAR:
      return std::string("CIRCULAR");
    default:
      return std::string("GRID");
    }
  }

  std::string
  Args::admis_kind_to_string(ADMIS_KIND admis_kind) {
    switch(admis_kind) {
    case GEOMETRY:
      return std::string("GEOMETRY");
    default:
      return std::string("DIAGONAL");
    }
  }

  std::string
  Args::construct_algorithm_to_string(CONSTRUCT_ALGORITHM construct_algorithm) {
    switch(construct_algorithm) {
    case ID_RANDOM:
      return std::string("ID_RANDOM");
    default:
      return std::string("MIRO");
    }
  }

  Args::Args(int argc, char** argv)
    : N(1000),
      nleaf(10),
      kernel_func(LAPLACE),
      kind_of_geometry(GRID),
      ndim(1),
      max_rank(2),
      admis(0),
      accuracy(1),
      add_diag(1-4),
      admis_kind(DIAGONAL),
      construct_algorithm(MIRO),
      use_nested_basis(true)
  {
    while(1) {
      int option_index;
      int c = getopt_long(argc, argv, "n:l:k:g:d:r:e:a:m:c:z:b:v:",
                          long_options, &option_index);

      if (c == -1) break;
      switch(c) {
      case 'n':
        break;
      case 'l':
        break;
      case 'k':
        break;
      case 'g':
        break;
      case 'd':
        break;
      case 'r':
        break;
      case 'e':
        break;
      case 'a':
        break;
      case 'm':
        break;
      case 'c':
        break;
      case 'z':
        break;
      case 'b':
        break;
      case 'v':
        break;
      default:
        usage(argv[0]);
        abort();
      }
    }
  }

  void
  Args::usage(const char *name) {
    if (verbose) {
      fprintf(stderr,
              "Usage: %s [options]\n"
              "Long option (short option)              : Description (Default value)\n"
              "--N (-n)                                : Number of points to consider (%lld).\n"
              "--nleaf (-l)                            : Max. number of points in a leaf node (%lld).\n"
              "--kernel_func (-k) [laplace]            : Kernel function to use (%s).\n"
              "--kind_of_geometry (-g) [sphere|grid]   : Kind of geometry of the points  (%s).\n"
              "--ndim (-d)                             : Number of dimensions of the geometry (%lld).\n"
              "--max_rank (-r)                             : Maximum rank (%lld).\n"
              "--accuracy (-e)                         : Desired accuracy for construction. Leave out for a constant rank construction (%lf).\n"
              "--admis (-a)                            : Admissibility constant (%lf).\n"
              "--admis_kind (-m) [diagonal|geometry]   : Whether geometry-based or diagonal-based admis (%s).\n"
              "--construct_algorithm (-c) [miro|id_random] : Construction algorithm to use (%s).\n"
              "--add_diag (-z)                             : Value to add to the diagonal (%lf).\n"
              "--nested_basis (-b)                         : Whether to use nested basis (%d).\n"
              "--verbose (-v)                              : Verbose mode (%d).\n",
              name,
              N,
              nleaf,
              kernel_func_to_string(kernel_func).c_str(),
              geometry_to_string(kind_of_geometry).c_str(),
              ndim,
              max_rank,
              accuracy,
              admis,
              admis_kind_to_string(admis_kind).c_str(),
              construct_algorithm_to_string(construct_algorithm).c_str(),
              add_diag,
              use_nested_basis,
              verbose);
    }
  }

  void
  Args::show() {

  }
}
