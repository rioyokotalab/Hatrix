#include "distributed/distributed.hpp"

#include <cstring>
#include <string>
#include <getopt.h>
#include <cassert>

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
    {"perturbation",        required_argument, 0, 'p'},
    {"admis_kind",          required_argument, 0, 'm'},
    {"construct_algorithm", required_argument, 0, 'c'},
    {"param_1",             required_argument, 0, 'x'},
    {"param_2",             required_argument, 0, 'y'},
    {"param_3",             required_argument, 0, 'z'},
    {"qr_accuracy",         required_argument, 0, 'q'},
    {"parsec_cores",        required_argument, 0, 'i'},
    {"kind_of_recompression", required_argument, 0, 's'},
    {"use_nested_basis",    required_argument,       0, 'b'},
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

  void
  Args::check_args() {
    if (kernel_verbose == "elses_c60") {
      assert(kind_of_geometry == ELSES_C60_GEOMETRY);
    }
  }

  Args::Args(int argc, char** argv)
    : N(1000),
      nleaf(10),
      kind_of_geometry(GRID),
      ndim(1),
      max_rank(2),
      admis(0),
      perturbation(0),
      accuracy(1),
      qr_accuracy(1e-2),
      kind_of_recompression(1),
      param_1(1e-4),
      param_2(0),
      param_3(0),
      admis_kind(DIAGONAL),
      construct_algorithm(MIRO),
      use_nested_basis(false),
      verbose(false),
      is_symmetric(false),
      parsec_cores(-1)
  {
    KERNEL_FUNC kfunc = LAPLACE;
    while(1) {
      int option_index;
      int c = getopt_long(argc, argv, "n:l:k:g:f:d:r:e:a:p:m:c:x:y:z:q:i:s:b:vh",
                          long_options, &option_index);

      if (c == -1) break;
      num_args++;
      switch(c) {
      case 'i':
        parsec_cores = std::stol(optarg);
        break;
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
        else if (!strcmp(optarg, "gsl_matern")) {
          kfunc = GSL_MATERN;
          kernel_verbose = std::string(optarg);
          is_symmetric = true;
        }
        else if (!strcmp(optarg, "yukawa")) {
          kfunc = YUKAWA;
          kernel_verbose = std::string(optarg);
          is_symmetric = true;
        }
        else if (!strcmp(optarg, "elses_c60")) {
          kfunc = ELSES_C60;
          kernel_verbose = std::string(optarg);
          is_symmetric = true;
        }
        else {
          throw std::invalid_argument("Cannot support "
                                      + std::string(optarg)
                                      + " for --kernel_func (-k).");
        }
        break;
      case 'g':
        if (!strcmp(optarg, "grid")) {
          kind_of_geometry = GRID;
        }
        else if (!strcmp(optarg, "circular")) {
          kind_of_geometry = CIRCULAR;
        }
        else if (!strcmp(optarg, "col_file")) {
          kind_of_geometry = COL_FILE;
        }
        else if (!strcmp(optarg, "elses_c60_geometry")) {
          kind_of_geometry = ELSES_C60_GEOMETRY;
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
      case 'p':
        perturbation = std::stod(optarg);
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
      case 's':
        kind_of_recompression = std::stol(optarg);
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
      case 'q':
        qr_accuracy = std::stod(optarg);
        break;
      case 'x':
        param_1 = std::stod(optarg);
        break;
      case 'y':
        param_2 = std::stod(optarg);
        break;
      case 'z':
        param_3 = std::stod(optarg);
        break;
      case 'b':
        use_nested_basis = std::stoi(optarg) == 1 ? true : false;
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
      switch (ndim) {
      case 2:
        kernel = [&](const std::vector<double>& c_row,
                     const std::vector<double>& c_col) {
          return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, param_1); // add_diag
        };
        break;
      default:
        kernel = [&](const std::vector<double>& c_row,
                     const std::vector<double>& c_col) {
          return Hatrix::greens_functions::laplace_3d_kernel(c_row, c_col, param_1); // add_diag
        };
      }
    }
    else if (kfunc == GSL_MATERN) {
      kernel = [&](const std::vector<double>& c_row,
                   const std::vector<double>& c_col) {
        return Hatrix::greens_functions::matern_kernel(c_row, c_col, param_1, param_2,
                             param_3); // sigma, nu, smoothness
      };
    }
    else if (kfunc == YUKAWA) {
      kernel = [&](const std::vector<double>& c_row,
                   const std::vector<double>& c_col) {
        return Hatrix::greens_functions::yukawa_kernel(c_row, c_col, param_1, param_2); // alpha, singularity
      };
    }

    check_args();
  }

  void
  Args::usage(const char *name) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Long option (short option)                  : Description (Default value)\n"
            "--N (-n)                                    : Number of points to consider (%ld).\n"
            "--nleaf (-l)                                : Max. number of points in a leaf node (%ld).\n"
            "--kernel_func (-k) [laplace]                : Kernel function to use (%s).\n"
            "--kind_of_geometry (-g) [circular|grid|       \n"
            " col_file]                                  : Kind of geometry of the points (%s). \n"
            "                                              If specifying col_file you must specify a geometry \n"
            "                                              file with fields <x y z> using --geometry_file or -f. \n"
            "--geometry_file (-f)                        : Geometry file. Reader format determined by --kind_of_geometry (%s). \n"
            "--ndim (-d)                                 : Number of dimensions of the geometry (%ld).\n"
            "--max_rank (-r)                             : Maximum rank (%ld).\n"
            "--accuracy (-e)                             : Desired accuracy for construction. > 0 for constant rank construction. (%lf).\n"
            "--qr_accuracy (-q)                          : Desired accuracy for QR. (%lf).\n"
            "--kind_of_recompression (-s)                : Recompression scheme (0,1,2,3) (%d). \n"
            "--admis (-a)                                : Admissibility constant (%lf).\n"
            "--pertubation (-p)                          : Parameter to add to the admissibility (%lf).\n"
            "--parsec_cores (-i)                         : Parameter to control the number of physical cores used by a single process of PaRSEC. (%d) \n"
            "--admis_kind (-m) [diagonal|geometry]       : Whether geometry-based or diagonal-based admis (%s).\n"
            "--construct_algorithm (-c) [miro|id_random] : Construction algorithm to use (%s).\n"
            "--param_1 (-x)                              : First parameter to the kernel. (%lf).\n"
            "--param_2 (-y)                              : Second parameter to the kernel. (%lf).\n"
            "--param_3 (-z)                              : Third parameter to the kernel. (%lf).\n"
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
            qr_accuracy,
            kind_of_recompression,
            admis,
            perturbation,
            parsec_cores,
            admis_kind_to_string(admis_kind).c_str(),
            construct_algorithm_to_string(construct_algorithm).c_str(),
            param_1,
            param_2,
            param_3,
            use_nested_basis,
            verbose);
  }

  void
  Args::show() {

  }
}
