#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cassert>

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  void
  Domain::search_tree_for_nodes(const Cell& tree, const int64_t level_index, const int64_t level,
                                int64_t &pstart, int64_t &pend) const {
    if (tree.level == level && tree.level_index == level_index) {
      pstart = tree.start_index;
      pend = tree.end_index;
      return;
    }

    if (tree.cells.size() > 0) {
      search_tree_for_nodes(tree.cells[0], level_index, level, pstart, pend);
      search_tree_for_nodes(tree.cells[1], level_index, level, pstart, pend);
    }
  }

  int64_t
  Domain::cell_size(int64_t level_index, int64_t level) const {
    int64_t pstart, pend;

    search_tree_for_nodes(tree, level_index, level, pstart, pend);

    return pend - pstart;
  }

  void
  Domain::print_file(std::string file_name) {
    std::vector<char> coords{'x', 'y', 'z'};

    std::ofstream file;
    file.open(file_name, std::ios::app | std::ios::out);
    for (int64_t k = 0; k < ndim; ++k) {
      file << coords[k] << ",";
    }
    file << std::endl;

    for (int64_t i = 0; i < N; ++i) {
      if (ndim == 1) {
        file << particles[i].coords[0] <<  std::endl;
      }
      else if (ndim == 2) {
        file << particles[i].coords[0] << "," << particles[i].coords[1] << std::endl;
      }
      else if (ndim == 3) {
        file << particles[i].coords[0] << "," << particles[i].coords[1]
             << "," << particles[i].coords[2] << std::endl;
      }
    }

    file.close();
  }

  Domain::Domain(int64_t N, int64_t ndim) : N(N), ndim(ndim) {
    if (ndim <= 0) {
      std::cout << "invalid ndim : " << ndim << std::endl;
      abort();
    }
    Xmin.resize(ndim);
    Xmax.resize(ndim);
  }

  void Domain::generate_grid_particles() {
    std::vector<int64_t> sides(ndim, 0);
    sides[0] = ceil(pow(N, 1.0 / ndim));
    int64_t total = sides[0];
    int64_t temp_N = N;
    for (int k = 1; k < ndim; ++k) {
      sides[k] = temp_N / sides[k-1];
      temp_N = sides[k];
    }
    for (int k = 1; k < ndim; ++k) { total += sides[k]; }
    int64_t extra = N - total;
    particles.resize(N, Particle(std::vector<double>(ndim), 0));

    if (ndim == 1) {
      double space_0 = 1.0 / N;
      for (int64_t i = 0; i < sides[0]; ++i) {
        std::vector<double> point(ndim);
        point[0] = i * space_0;
        particles[i] = Hatrix::Particle(point, i);
      }
    }
    else if (ndim == 2) {
      const double a = 0.5;
      const int64_t top = N / 4;
      const int64_t left = N / 2;
      const int64_t bottom = 3 * N / 4;
      int64_t i = 0;
      for (i = 0; i < top; i++) {
        const double x = a - 2.0 * a * i / top;
        const double y = a;
        const double value = (double)i;
        particles[i] = Hatrix::Particle(x, y, value);
      }
      for (; i < left; i++) {
        const double x = -a;
        const double y = a - 2.0 * a * (i - top) / (left - top);
        const double value = (double)i;
        particles[i] = Hatrix::Particle(x, y, value);
      }
      for (; i < bottom; i++) {
        const double x = -a + 2.0 * a * (i - left) / (bottom - left);
        const double y = -a;
        const double value = (double)i;
        particles[i] = Hatrix::Particle(x, y, value);
      }
      for (; i < N; i++) {
        const double x = a;
        const double y = -a + 2.0 * a * (i - bottom) / (N - bottom);
        const double value = (double)i;
        particles[i] = Hatrix::Particle(x, y, value);
      }
    }
    else if (ndim == 3) {
      abort();
    }
  }

  void Domain::generate_circular_particles(double min_val, double max_val) {
    double range = max_val - min_val;

    if (ndim == 1) {
      auto vec = equally_spaced_vector(N, min_val, max_val);
      for (int64_t i = 0; i < N; ++i) {
        particles.push_back(Hatrix::Particle(vec[i], min_val + (double(i) / double(range))));
      }
    }
    else if (ndim == 2) {
      // Generate a unit circle with N points on the circumference.
      // std::random_device rd;  // Will be used to obtain a seed for the random number engine
      std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
      double radius = 1.0;

      for (int64_t i = 0; i < N; ++i) {
        double theta = (i * 2.0 * M_PI) / N ;
        Particle p(ndim);
        p.coords[0] = radius * cos(theta);
        p.coords[1] = radius * sin(theta);
        p.value = i;

        particles.emplace_back(p);
      }
    }
    else if (ndim == 3) {
      // Generate a unit sphere mesh with N uniformly spaced points on the surface
      // https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
      const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
      for (int64_t i = 0; i < N; i++) {
        const double y = 1. - ((double)i / ((double)N - 1)) * 2.;  // y goes from 1 to -1

        // Note: setting constant radius = 1 will produce a cylindrical shape
        const double radius = std::sqrt(1. - y * y);  // radius at y
        const double theta = (double)i * phi;

        const double x = radius * std::cos(theta);
        const double z = radius * std::sin(theta);
        const double value = (double)i / (double)N;
        particles.emplace_back(Particle(x, y, z, value));
      }
    }
  }

  void
  Domain::read_col_file_3d(const std::string& geometry_file) {
    std::ifstream file;
    file.open(geometry_file, std::ios::in);
    double x, y, z, kmeans_index;

    for (int64_t line = 0; line < N; ++line) {
      file >> x >> y >> z >> kmeans_index;
      particles.push_back(Hatrix::Particle(x, y, z, kmeans_index));
    }

    file.close();
  }

  void Domain::read_col_file_2d(const std::string& geometry_file) {
    std::ifstream file;
    file.open(geometry_file, std::ios::in);

    double x, y;

    for (int64_t line = 0; line < N; ++line) {
      file >> x >> y;
      particles.push_back(Hatrix::Particle(x, y));
    }

    file.close();
  }

  double Domain::get_axis_min(int64_t start_index, int64_t end_index, int64_t axis) {
    double min_coord = particles[start_index].coords[axis];
    for (int64_t i = start_index; i < end_index; ++i) {
      min_coord = std::min(min_coord, particles[i].coords[axis]);
    }

    return min_coord;
  }

  double Domain::get_axis_max(int64_t start_index, int64_t end_index, int64_t axis) {
    double min_coord = particles[start_index].coords[axis];
    for (int64_t i = start_index; i < end_index; ++i) {
      min_coord = std::max(min_coord, particles[i].coords[axis]);
    }

    return min_coord;
  }

  // Get integer indices for the given index X.
  std::vector<int64_t>
  Domain::int_index_3d(const std::vector<double>& X,
                       const std::vector<Cell>& temp_cell_list,
                       const int64_t level) {
    std::vector<int64_t> iX(ndim);
    const std::vector<double>& R0 = temp_cell_list[0].radii;
    for (int axis = 0; axis < ndim; ++axis) {
      const double dx = 2 * R0[axis] / pow(2, level);
      iX[axis] = floor((X[axis] - Xmin[axis]) / dx);
    }
    return iX;
  }

  int64_t
  Domain::hilbert_index(std::vector<int64_t>& iX,
                        const int64_t level,
                        const bool offset) {
    int64_t level_offset = (((int64_t)1 << 3 * level) - 1) / 7; // level-wise offset for hilbert key.
    // Preprocess for Hilbert
    // int64_t M = 1 << (level - 1);
    // for (int64_t Q=M; Q>1; Q>>=1) {
    //   int64_t R = Q - 1;
    //   for (int64_t d = 0; d < ndim; d++) {
    //     if (iX[d] & Q) iX[0] ^= R;
    //     else {
    //       int64_t t = (iX[0] ^ iX[d]) & R;
    //       iX[0] ^= t;
    //       iX[d] ^= t;
    //     }
    //   }
    // }
    // for (int64_t d=1; d < ndim; d++) iX[d] ^= iX[d-1];
    // int64_t t = 0;
    // for (int64_t Q=M; Q>1; Q>>=1)
    //   if (iX[2] & Q) t ^= Q - 1;
    // for (int64_t d=0; d < ndim; d++) iX[d] ^= t;

    // Just generate the morton index.
    int64_t i = 0;
    for (int64_t l = 0; l < level; l++) {
      for (int axis = 0; axis < ndim; ++axis) {
        i |= (iX[ndim - axis - 1] & (int64_t)1 << l) << (2*l + axis);
      }
    }
    if (offset) i += level_offset;

    return i;
  }

  // Code derived from https://github.com/exafmm/minimal/blob/master/3d/build_tree.h#L24
  void
  Domain::sort_particles_and_build_tree(Particle *buffer, Particle* bodies,
                                        const int64_t start_index, const int64_t end_index,
                                        int64_t cell_list_index,
                                        std::vector<Cell>& cell_list,
                                        int64_t nleaf, int64_t level,
                                        bool direction) {
    Cell& cell = cell_list[cell_list_index];
    cell.start_index = start_index;
    cell.end_index = end_index;
    cell.level = level;
    cell.nchild = 0;
    std::vector<int64_t> index_3d = int_index_3d(cell.center, cell_list, level);
    cell.key = hilbert_index(index_3d, level, false);


    if (end_index - start_index <= nleaf) {
      if (direction) {                      // copy data into the original array.
        for (int64_t i = start_index; i < end_index; ++i) {
          buffer[i].coords = bodies[i].coords;
          buffer[i].value = bodies[i].value;
        }
      }

      // std::cout << "st: " << cell.start_index << " end: " << cell.end_index
      //           << " lvl: " << cell.level << " +++ " << cell.key << " --- ";

      // for (int axis = 0; axis < ndim; ++axis) {
      //   std::cout << cell.center[axis] << " $$$ " << index_3d[axis];
      // }
      return;
    }

    int domain_divs = pow(2, ndim); // number of divisions of the domain.
    std::vector<int64_t>size(domain_divs, 0);
    std::vector<double> x(ndim);

    for (int64_t i = start_index; i < end_index; ++i) {
      x = bodies[i].coords;
      int64_t octant = 0;
      for (int axis = 0; axis < ndim; ++axis) {
        octant += (x[axis] > cell.center[axis]) << axis;
      }
      size[octant]++;
    }

    // Obtain the offsets for each octant from the number of particles in 'size'.
    int64_t offset = start_index;
    std::vector<int64_t> offsets(domain_divs, 0), counter(domain_divs, 0);
    for (int i = 0; i < domain_divs; ++i) {
      offsets[i] = offset;
      offset += size[i];
      if (size[i]) { cell.nchild++; }
    }

    // Sort bodies by the octant/quadrant that they belong to.
    counter = offsets;
    for (int64_t i = start_index; i < end_index; ++i) {
      x = bodies[i].coords;
      int64_t octant = 0;
      for (int axis = 0; axis < ndim; ++axis) {
        octant += (x[axis] > cell.center[axis]) << axis;
      }
      // Copy this particle to its correct location in the buffer.
      buffer[counter[octant]].coords = bodies[i].coords;
      buffer[counter[octant]].value = bodies[i].value;
      counter[octant]++;        // move to the next location in the sorted octant/quadrant.
    }

    std::cout << "level: " << level << " size: ";
    for (int ax = 0; ax < domain_divs; ++ax) {
      std::cout << size[ax] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < cell.nchild; ++i) { cell_list.push_back(Cell(ndim));}
    Cell& first_child = cell_list[cell_list.size() - cell.nchild];
    const Cell& parent = cell_list[cell_list_index];

    cell.child = &first_child;

    int c = 0;
    int64_t child_index = cell_list.size() - cell.nchild;
    for (int i = 0; i < domain_divs; ++i) {
      if (size[i]) {
        Cell& child = cell_list[child_index];
        std::vector<double> child_center(ndim);
        double child_radius = parent.radius / 2;

        for (int axis = 0; axis < ndim; ++axis) {
          child_center[axis] = parent.center[axis];
          child_center[axis] += child_radius * (((i & 1 << axis) >> axis) * 2 - 1);
        }
        child.center = child_center;
        child.radius = child_radius;

        sort_particles_and_build_tree(bodies, buffer,
                                      offsets[i], offsets[i] + size[i],
                                      child_index, cell_list,
                                      nleaf, level+1, !direction);
        child_index++;
        c++;
      }
    }
  }

  int64_t
  Domain::level_offset(int64_t level) {
    return (((int64_t)1 << 3 * level) - 1) / 7;
  }

  int64_t
  Domain::get_level(int64_t hilbert_key) {
    int64_t level = -1;
    uint64_t offset = 0;

    while (hilbert_key >= offset) {
      level++;
      offset += (int64_t)1 << 3 * level;
    }

    return level;
  }

  // Get the hilbert key of the first child node of this node.
  int64_t
  Domain::get_hilbert_id_child(int64_t hilbert_key) {
    int64_t level = get_level(hilbert_key);
    return (hilbert_key - level_offset(level)) * 8 + level_offset(level+1);
  }

  void
  Domain::sort_elses_bodies(const int64_t molecule_size) {
    // Subdivide the particles (atoms) into molecules. Each molecule contains
    // molecule_size atoms.
    assert(N % molecule_size == 0);
    const int64_t nmolecules = N / molecule_size;

    std::vector<Cell> temp_cell_list;

    temp_cell_list.reserve(nmolecules * log(nmolecules));

    // Bounding box of the root cell.
    Cell root(ndim);
    root.start_index = 0;
    root.end_index = nmolecules;
    root.radius = 0;
    for (int64_t axis = 0; axis < ndim; ++axis) {
      auto axis_min = get_axis_min(0, N, axis);
      auto axis_max = get_axis_max(0, N, axis);

      Xmin[axis] = axis_min;
      Xmax[axis] = axis_max;

      root.radii[axis] = std::abs(axis_max - axis_min) / 2;
      root.center[axis] = (axis_min + axis_max) / 2;
      root.radius = std::max(std::abs(root.radii[axis] - axis_min), root.radius);
      root.radius = std::max(std::abs(axis_max - root.radii[axis]), root.radius);
    }
    root.level = 0;
    temp_cell_list.push_back(root);


    // Each vector within this vector contains co-ordinates that correspond to
    // the center of each molecule.
    std::vector<Particle> molecule_centers;

    for (int64_t mol = 0; mol < nmolecules; ++mol) {
      Particle mol_center(ndim);
      mol_center.value = mol;

      auto start_index = mol * molecule_size;
      auto end_index = mol * molecule_size + molecule_size;
      for (int axis = 0; axis < ndim; ++axis) {
        auto axis_min = get_axis_min(start_index, end_index, axis);
        auto axis_max = get_axis_max(start_index, end_index, axis);
        mol_center.coords[axis] = (axis_min + axis_max) / 2;
      }

      molecule_centers.push_back(mol_center);
    }

    // Sort the particles according to octants using an out-of-place sort
    // (something like a merge sort). Along with sorting, also generate
    // temp_cell_list that correspond to ranges of sorted particles and assign various
    // parameters such as start and stop index, hilbert index and level.
    const int64_t nmolecules_per_box = 1;
    std::vector<Particle> buffer = molecule_centers;
    sort_particles_and_build_tree(buffer.data(), molecule_centers.data(),
                                  0, nmolecules,
                                  0, temp_cell_list,
                                  nmolecules_per_box, 0, false);
    // Partition the molecules using Hilbert indexing and sort them.

    int64_t max_level = 0;
    for (auto cell : temp_cell_list) { max_level = std::max(max_level, cell.level); }

    // Permutation vector for storing the sorted order of the molecules.
    // <array_index, hilbert_index>

    std::vector<std::tuple<int64_t, int64_t> > hilbert_permute_vector;
    hilbert_permute_vector.reserve(nmolecules);
    for (auto cell : temp_cell_list) {
      if (cell.nchild == 0) { // leaf level
        auto hilbert_level = cell.level;
        auto hilbert_id = cell.key;

        while (hilbert_level < max_level) {
          hilbert_id = get_hilbert_id_child(hilbert_id);
          hilbert_level++;
        }

        for (int64_t b = cell.start_index; b < cell.end_index; ++b) {
          std::tuple<int64_t, int64_t> t = std::make_tuple(b, hilbert_id);
          hilbert_permute_vector.push_back(t);
        }
      }
    }

    // Sort the molecules based on the hilbert index generated during the tree generation.
    std::sort(hilbert_permute_vector.begin(), hilbert_permute_vector.end(),
              [](const std::tuple<int64_t, int64_t>& a,
                 const std::tuple<int64_t, int64_t>& b) {
                return std::get<1>(a) <= std::get<1>(b); // sort based on hilbert index.
              });
    std::vector<Particle> hilbert_sorted_molecule_centers(nmolecules);
    for (int64_t i = 0; i < nmolecules; ++i) {
      int64_t sorted_index = std::get<0>(hilbert_permute_vector[i]);
      hilbert_sorted_molecule_centers[sorted_index] = molecule_centers[i];
    }

    // Sort the electrons (actual Particles in the Domain) based on the sorted molecules.
    std::vector<Particle> temp = particles;
    int64_t count = 0;
    for (int64_t i = 0; i < nmolecules; ++i) {
      const auto& molecule = hilbert_sorted_molecule_centers[i];
      const auto src_begin = molecule.value * molecule_size;
      const auto dst_begin = count;
      for (int64_t k = 0; k < molecule_size; ++k) {
        particles[dst_begin+k] = temp[src_begin+k];
      }
      count += molecule_size;
    }

    auto get_sort_axis = [this](const int64_t start_index, const int64_t end_index) {
      double max_radius = 0;
      int64_t sort_axis = 0;
      for (int axis = 0; axis < ndim; ++axis) {
        const auto Xmin = get_axis_min(start_index, end_index, axis);
        const auto Xmax = get_axis_max(start_index, end_index, axis);
        const auto radius = (Xmax - Xmin) / 2.0;
        if (radius > max_radius) {
          max_radius = radius;
          sort_axis = axis;
        }
      }

      return sort_axis;
    };

    // Sort the electrons within the molecules
    for (int64_t mol = 0; mol < nmolecules; ++mol) {
      // Sort along the longest axis.
      const int64_t start_index = mol * molecule_size;
      const int64_t end_index = start_index + molecule_size;
      const int64_t sort_axis = get_sort_axis(start_index, end_index);
      std::sort(particles.begin() + start_index, particles.begin() + end_index,
                [sort_axis](const Particle& a, const Particle& b) {
                  return a.coords[sort_axis] < b.coords[sort_axis];
                });

      // Cut the molecules in half, find the longest axis for each half and
      // then sort again.
    }
  }

  int64_t Domain::build_bottom_up_binary_tree(const int64_t nleaf) {
    int64_t height = (int64_t)std::log2((double)N / (double)nleaf);
    const int64_t leaf_ncells = pow(2, height);
    tree_list.resize(leaf_ncells * 2 - 1, Cell(ndim));

    for (int64_t level = height; level >= 0; --level) {
      const int64_t level_ncells = pow(2, level);
      const int64_t level_offset = level_ncells - 1;
      for (int64_t node = 0; node < level_ncells; ++node) {
        Cell& cell = tree_list[level_offset + node];
        cell.level = level;
        cell.key = node;

        if (level == height) {  // leaf node directly from particles.
          cell.nchild = 0;
          cell.start_index = node * nleaf;
          cell.end_index = cell.start_index + nleaf;
        }
        else {                  // non-leaf node from adjacent lower nodes.
          cell.nchild = 2;
          const int64_t child1_id = (pow(2, level+1) - 1) + node * 2;
          const int64_t child2_id = (pow(2, level+1) - 1) + node * 2 + 1;
          Cell& child1 = tree_list[child1_id];
          Cell& child2 = tree_list[child2_id];
          cell.child = &child1;
          cell.start_index = child1.start_index;
          cell.end_index = child2.end_index;
        }

        // Calculate cell center and radius.
        cell.radius = 0;
        for (int axis = 0; axis < ndim ; ++axis) {
          const double X_min = get_axis_min(cell.start_index, cell.end_index, axis);
          const double X_max = get_axis_max(cell.start_index, cell.end_index, axis);

          cell.center[axis] = (X_min + X_max) / 2;
          cell.radii[axis] = std::abs(X_max - X_min) / 2;
          cell.radius = std::max(cell.radius, std::abs(cell.radius - X_min));
          cell.radius = std::max(cell.radius, std::abs(X_max - cell.radius));
        }
      }
    }

    // std::cout << "distances:\n";
    // for (int i = 16; i < tree_list.size(); ++i) {
    //   for (int j = 16; j < tree_list.size(); ++j) {
    //     const auto& Ci = tree_list[i];
    //     const auto& Cj = tree_list[j];

    //     if (Ci.nchild == 0 && Cj.nchild == 0) {
    //       double dist = 0;
    //       for (int axis = 0; axis < ndim; ++axis) {
    //         dist += pow(Ci.center[axis] - Cj.center[axis], 2);
    //       }
    //       dist = sqrt(dist);

    //       std::cout << std::setw(6) << std::setprecision(3)
    //                 << dist << " ";

    //                 // << "(" << Ci.center[0] << ","
    //                 // << Cj.center[0] << ")";
    //     }
    //   }
    //   std::cout << std::endl;
    // }

    return height;
  }

  int64_t
  Domain::build_elses_tree(const int64_t molecule_size) {
    sort_elses_bodies(molecule_size);
    int64_t tree_height = build_bottom_up_binary_tree(molecule_size);

    return tree_height;
  }

  int64_t
  Domain::sort_generic_geometry_particles(const int64_t nleaf) {
    assert(N % nleaf == 0);
    const int64_t nblocks = N / nleaf;

    std::vector<Cell> temp_cell_list;

    temp_cell_list.reserve(nblocks * log(nblocks));
    // Bounding box of the root cell.
    Cell root(ndim);
    root.start_index = 0;
    root.end_index = N;
    root.radius = 0;
    for (int64_t axis = 0; axis < ndim; ++axis) {
      auto axis_min = get_axis_min(0, N, axis);
      auto axis_max = get_axis_max(0, N, axis);

      Xmin[axis] = axis_min;
      Xmax[axis] = axis_max;

      root.radii[axis] = std::abs(axis_max - axis_min) / 2;
      root.center[axis] = (axis_min + axis_max) / 2;
      root.radius = std::max(std::abs(root.radii[axis] - axis_min), root.radius);
      root.radius = std::max(std::abs(axis_max - root.radii[axis]), root.radius);
    }
    root.level = 0;
    temp_cell_list.push_back(root);

    std::vector<Particle> buffer = particles;
    sort_particles_and_build_tree(buffer.data(), particles.data(),
                                  0, N,
                                  0, temp_cell_list,
                                  nleaf, 0, false);

    int64_t max_level = 0;
    for (auto cell : temp_cell_list) { max_level = std::max(max_level, cell.level); }

    std::vector<std::tuple<Cell, int64_t> > hilbert_permute_vector;
    hilbert_permute_vector.reserve(nblocks);
    for (auto cell : temp_cell_list) {
      if (cell.nchild == 0) { // leaf level
        auto hilbert_level = cell.level;
        auto hilbert_id = cell.key;

        while (hilbert_level < max_level) {
          hilbert_id = get_hilbert_id_child(hilbert_id);
          hilbert_level++;
        }

        std::tuple<Cell, int64_t> t = std::make_tuple(cell, hilbert_id);
        hilbert_permute_vector.push_back(t);
      }
    }

    // std::cout << "circle centers: ";
    // for (auto c : hilbert_permute_vector) {
    //   auto cell = std::get<0>(c);
    //   auto h_id = std::get<1>(c);
    //   for (int axis = 0; axis < ndim; ++axis) {
    //     std::cout << cell.center[axis] << ",";
    //   }
    //   std::cout << " -- " << h_id;
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // Sort the molecules based on the hilbert index generated during the tree generation.
    std::sort(hilbert_permute_vector.begin(), hilbert_permute_vector.end(),
              [](const std::tuple<Cell, int64_t>& a,
                 const std::tuple<Cell, int64_t>& b) {
                return std::get<1>(a) <= std::get<1>(b); // sort based on hilbert index.
              });

    // std::cout << "sorted hilbert permute vector: ";
    // for (auto c : hilbert_permute_vector) {
    //   const auto a = std::get<1>(c);
    //   std::cout << a << " ";
    // }
    // std::cout << std::endl;

    // Sort the electrons (actual Particles in the Domain) based on the sorted molecules.
    std::vector<Particle> temp = particles;
    int64_t count = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
      const auto& t = hilbert_permute_vector[i];
      const Cell& cell = std::get<0>(t);
      const int64_t src_begin = cell.start_index;
      const int64_t dst_begin = count;

      for (int64_t k = 0; k < nleaf; ++k) {
        particles[dst_begin + k] = temp[src_begin + k];
      }
      count += nleaf;
    }

    // for (const auto& p : hilbert_permute_vector) {
    //   auto h = std::get<1>(p);
    //   std::cout << h << " ";
    // }
    // std::cout << std::endl;

    auto get_sort_axis = [this](const int64_t start_index, const int64_t end_index) {
      double max_radius = 0;
      int64_t sort_axis = 0;
      for (int axis = 0; axis < ndim; ++axis) {
        const auto Xmin = get_axis_min(start_index, end_index, axis);
        const auto Xmax = get_axis_max(start_index, end_index, axis);
        const auto radius = (Xmax - Xmin) / 2.0;
        if (radius > max_radius) {
          max_radius = radius;
          sort_axis = axis;
        }
      }

      return sort_axis;
    };

    // Sort the electrons within the molecules
    for (int64_t mol = 0; mol < nblocks; ++mol) {
      // Sort along the longest axis.
      const int64_t start_index = mol * nleaf;
      const int64_t end_index = start_index + nleaf;
      const int64_t sort_axis = get_sort_axis(start_index, end_index);
      std::sort(particles.begin() + start_index, particles.begin() + end_index,
                [sort_axis](const Particle& a, const Particle& b) {
                  return a.coords[sort_axis] < b.coords[sort_axis];
                });

      // Cut the molecules in half, find the longest axis for each half and
      // then sort again.
    }

    return max_level;
  }

  void Domain::read_xyz_chemical_file(const std::string& geometry_file,
                                      const int64_t num_electrons_per_atom) {
    std::ifstream file;
    file.open(geometry_file);
    int64_t num_atoms;
    file >> num_atoms;
    ndim = 3;
    N = num_atoms * num_electrons_per_atom;

    // Ignore the rest of line after num_particles
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    // Ignore line before atom positions
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    int64_t body_idx = 0;
    for(int64_t i = 0; i < num_atoms; i++) {
      std::string pref;
      double x, y, z;
      file >> pref >> x >> y >> z;
      file.ignore(1, '\n'); //Ignore newline

      for (int64_t k = 0; k < num_electrons_per_atom; k++) {
        // The body_idx stores a number that corresponds to the index at which this particle exists
        // in the geometry. This value is used as a placeholder for tracing the initial position
        // of this particle in the domain. This is useful when sorting the bodies and generating
        // the admissibility because the sorting can destroy the original indexing of the particles.
        // The body_idx is used for tracing back the initial position of this particle.
        particles.emplace_back(Hatrix::Particle(x, y, z, (double)body_idx));
        body_idx++;
      }
    }
    file.close();
  }
}
