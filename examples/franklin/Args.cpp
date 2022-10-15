#include "franklin/franklin.hpp"

#include <cstring>
#include <string>
#include <getopt.h>

namespace Hatrix {
  static struct option long_options[] = {
    {"N",                   required_argument, 0, 'n'},
    {"nleaf",               required_argument, 0, 'l'},
    {"kernel_func",         required_argument, 0, 'k'},
    {"kind_of_geometry",    required_argument, 0, 'g'},
    {"geometry_file",       required_argument, 0, 'f'},
    {"ndim",                required_argument, 0, 'd'},
    {"max_rank",            required_argument, 0, 'r'},
    {"accuracy",            required_argument, 0, 'e'},
    {"admis",               required_argument, 0, 'a'},
    {"admis_kind",          required_argument, 0, 'm'},
    {"construct_algorithm", required_argument, 0, 'c'},
    {"add_diag",            required_argument, 0, 'z'},
    {"use_nested_basis",    no_argument,       0, 'b'},
    {"verbose",             no_argument,       0, 'v'},
    {"help",                no_argument,       0, 'h'},
    {0,                     0,                 0,  0},
  };

  std::string
  Args::geometry_to_string(KIND_OF_GEOMETRY geometry) {
    switch(geometry) {
    case CIRCULAR:
      return std::string("circular");
    default:
      return std::string("grid");
    }
  }

  std::string
  Args::admis_kind_to_string(ADMIS_KIND admis_kind) {
    switch(admis_kind) {
    case GEOMETRY:
      return std::string("geometry");
    default:
      return std::string("diagonal");
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
      kind_of_geometry(GRID),
      ndim(1),
      max_rank(2),
      admis(0),
      accuracy(1),
      add_diag(1e-4),
      admis_kind(DIAGONAL),
      construct_algorithm(MIRO),
      use_nested_basis(true),
      verbose(false),
      is_symmetric(false)
  {
    KERNEL_FUNC kfunc;
    while(1) {
      int option_index;
      int c = getopt_long(argc, argv, "n:l:k:f:g:d:r:e:a:m:c:z:bvh",
                          long_options, &option_index);

      if (c == -1) break;
      num_args++;
      switch(c) {
      case 'n':
        N = std::stol(optarg);
        break;
      case 'l':
        nleaf = std::stol(optarg);
        break;
      case 'k':
        if (!strcmp(optarg, "laplace")) {
          kfunc = LAPLACE;
          kernel_verbose = std::string(optarg);
          is_symmetric = true;
        }
        else {
          throw std::invalid_argument("Cannot support "
                                      + std::string(optarg)
                                      + " for --kind_of_kernel (-k).");
        }
        break;
      case 'g':
        if (!strcmp(optarg, "grid")) {
          kind_of_geometry = GRID;
        }
        else if (!strcmp(optarg, "circular")) {
          kind_of_geometry = CIRCULAR;
        }
        else if (!strcmp(optarg, "col_file_3d")) {
          kind_of_geometry = COL_FILE_3D;
        }
        else {
          throw std::invalid_argument("Cannot support " +
                                      std::string(optarg) +
                                      " for --kind_of_geometry (-g)");
        }
        break;
      case 'd':
        ndim = std::stol(optarg);
        break;
      case 'r':
        max_rank = std::stol(optarg);
        break;
      case 'e':
        accuracy = std::stod(optarg);
        break;
      case 'a':
        admis = std::stod(optarg);
        break;
      case 'm':
        if (!strcmp(optarg, "geometry")) {
          admis_kind = GEOMETRY;
        }
        else if (!strcmp(optarg, "diagonal")) {
          admis_kind = DIAGONAL;
        }
        else {
          throw std::invalid_argument("Cannot support " +
                                      std::string(optarg) +
                                      " for --admis_kind (-m).");
        }
        break;
      case 'f':
        geometry_file = std::string(optarg);
        break;
      case 'c':
        if (!strcmp(optarg, "miro")) {
          construct_algorithm = MIRO;
        }
        else if (!strcmp(optarg, "id_random")) {
          construct_algorithm = ID_RANDOM;
        }
        else {
          throw std::invalid_argument("Cannot support " +
                                      std::string(optarg) +
                                      " for --construct-algorithm (-c).");
        }
        break;
      case 'z':
        add_diag = std::stod(optarg);
        break;
      case 'b':
        use_nested_basis = true;
        break;
      case 'v':
        verbose = true;
        break;
      case 'h':
        usage(argv[0]);
        exit(0);
      default:
        fprintf(stderr, "Please supply appropriate cmd options as below. %s is not supported\n",
                optarg);
        usage(argv[0]);
        exit(2);
      }
    }

    if (kfunc == LAPLACE) {
      kernel = [&](const std::vector<double>& c_row,
                   const std::vector<double>& c_col) {
        return laplace_kernel(c_row, c_col, add_diag);
      };
    }
  }

  void
  Args::usage(const char *name) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Long option (short option)                  : Description (Default value)\n"
            "--N (-n)                                    : Number of points to consider (%lld).\n"
            "--nleaf (-l)                                : Max. number of points in a leaf node (%lld).\n"
            "--kernel_func (-k) [laplace]                : Kernel function to use (%s).\n"
            "--kind_of_geometry (-g) [sphere|grid|       \n"
            " 3d_col_fild]                               : Kind of geometry of the points (%s). \n"
            "                                              If specifying 3d_col_file you must specify a geometry \n"
            "                                              file with fields <x y z cluster_num> using. \n"
            "                                              geometry_file or -f. \n"
            "--geometry_file (-f)                        : Geometry file. Reader format determined by --kind_of_geometry (%s). \n"
            "--ndim (-d)                                 : Number of dimensions of the geometry (%lld).\n"
            "--max_rank (-r)                             : Maximum rank (%lld).\n"
            "--accuracy (-e)                             : Desired accuracy for construction. > 0 for constant rank construction. (%lf).\n"
            "--admis (-a)                                : Admissibility constant (%lf).\n"
            "--admis_kind (-m) [diagonal|geometry]       : Whether geometry-based or diagonal-based admis (%s).\n"
            "--construct_algorithm (-c) [miro|id_random] : Construction algorithm to use (%s).\n"
            "--add_diag (-z)                             : Value to add to the diagonal (%lf).\n"
            "--nested_basis (-b)                         : Whether to use nested basis (%d).\n"
            "--verbose (-v)                              : Verbose mode (%d).\n"
            "--help (-h)                                 : Show this help message.\n",
            name,
            N,
            nleaf,
            kernel_verbose.c_str(),
            geometry_to_string(kind_of_geometry).c_str(),
            "",
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

  void
  Args::show() {

  }
}
