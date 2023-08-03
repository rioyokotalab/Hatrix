#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
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
        particles[i] = Hatrix::Particle(point, 0);
      }
    }
    else if (ndim == 2) {
      double space_0 = 1.0 / sides[0], space_1 = 1.0 / sides[1];
      for (int64_t i = 0; i < sides[0]; ++i) {
        for (int64_t j = 0; j < sides[1]; ++j) {
          std::vector<double> point(ndim);
          point[0] = i * space_0;
          point[1] = j * space_1;
          particles[i + j * sides[0]] = Hatrix::Particle(point, 0);
        }
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
        double x = radius * cos(theta);
        double y = radius * sin(theta);

        particles.emplace_back(Hatrix::Particle(x, y, min_val + (double(i) / double(range))));
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
                       const int64_t level) {
    std::vector<int64_t> iX(ndim);
    const std::vector<double>& R0 = tree_list[0].radii;
    for (int axis = 0; axis < ndim; ++axis) {
      const double dx = 2 * R0[axis] / pow(2, level);
      iX[axis] = floor((X[axis] - Xmin[axis]) / dx);
    }
    return iX;
  }

  int64_t Domain::hilbert_index(std::vector<int64_t>& iX, const int64_t level,
                                const bool offset) {
    int64_t level_offset = (((int64_t)1 << 3 * level) - 1) / 7; // level-wise offset for hilbert key.
    // Preprocess for Hilbert
    int64_t M = 1 << (level - 1);
    for (int64_t Q=M; Q>1; Q>>=1) {
      int64_t R = Q - 1;
      for (int64_t d = 0; d < ndim; d++) {
        if (iX[d] & Q) iX[0] ^= R;
        else {
          int64_t t = (iX[0] ^ iX[d]) & R;
          iX[0] ^= t;
          iX[d] ^= t;
        }
      }
    }
    for (int64_t d=1; d < ndim; d++) iX[d] ^= iX[d-1];
    int64_t t = 0;
    for (int64_t Q=M; Q>1; Q>>=1)
      if (iX[2] & Q) t ^= Q - 1;
    for (int64_t d=0; d < ndim; d++) iX[d] ^= t;

    int64_t i = 0;
    for (int64_t l = 0; l < level; l++) {
      i |= (iX[2] & (int64_t)1 << l) << 2*l;
      i |= (iX[1] & (int64_t)1 << l) << (2*l + 1);
      i |= (iX[0] & (int64_t)1 << l) << (2*l + 2);
    }
    if (offset) i += level_offset;

    return i;
  }

  // Code derived from https://github.com/exafmm/minimal/blob/master/3d/build_tree.h#L24
  void Domain::sort_particles_and_build_tree(Particle *buffer, Particle* bodies,
                                             const int64_t start_index, const int64_t end_index,
                                             int64_t cell_list_index, std::vector<Cell>& cell_list,
                                             int64_t nleaf, int64_t level, bool direction) {
    Cell& cell = cell_list[cell_list_index];
    cell.start_index = start_index;
    cell.end_index = end_index;
    cell.level = level;
    cell.nchild = 0;
    std::vector<int64_t> index_3d = int_index_3d(cell.center, level);
    cell.key = hilbert_index(index_3d, level, false);

    if (end_index - start_index <= nleaf) {
      if (direction) {                      // copy data into the original array.
        for (int64_t i = start_index; i < end_index; ++i) {
          buffer[i].coords = bodies[i].coords;
          buffer[i].value = bodies[i].value;
        }
      }
      return;
    }

    int domain_divs = pow(2, ndim); // number of divisions of the domain.
    std::vector<int64_t>size(domain_divs, 0);
    std::vector<double> x(ndim);

    for (int64_t i = start_index; i < end_index; ++i) {
      x = bodies[i].coords;
      int64_t octant = (x[0] > cell.center[0]) +
        ((x[1] > cell.center[1]) << 1) +
        ((x[2] > cell.center[2]) << 2);
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
      int64_t octant = (x[0] > cell.center[0]) +
        ((x[1] > cell.center[1]) << 1) +
        ((x[2] > cell.center[2]) << 2);
      // Copy this particle to its correct location in the buffer.
      buffer[counter[octant]].coords = bodies[i].coords;
      buffer[counter[octant]].value = bodies[i].value;
      counter[octant]++;        // move to the next location in the sorted octant/quadrant.
    }

    // std::cout << "nchild: " << cell.nchild << std::endl;
    for (int i = 0; i < cell.nchild; ++i) { cell_list.push_back(Cell(ndim));}
    Cell& first_child = cell_list[cell_list.size() - cell.nchild];
    cell.child = &first_child;

    int c = 0;
    int64_t child_index = cell_list.size() - cell.nchild;
    for (int i = 0; i < domain_divs; ++i) {
      if (size[i]) {
        Cell& child = cell_list[child_index];
        std::vector<double> child_center(ndim);
        double child_radius = cell.radius / 2;

        for (int axis = 0; axis < ndim; ++axis) {
          child_center[axis] = cell.center[axis];
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

  void Domain::sort_elses_bodies(const int64_t molecule_size) {
    // Subdivide the particles (atoms) into molecules. Each molecule contains
    // molecule_size atoms.
    assert(N % molecule_size == 0);
    const int64_t nmolecules = N / molecule_size;

    tree_list.reserve(nmolecules * log(nmolecules));

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
      root.radius = std::max(root.radii[axis] - axis_min, root.radius);
      root.radius = std::max(axis_max - root.radii[axis], root.radius);
    }

    root.level = 0;
    tree_list.push_back(root);


    // Each vector within this vector contains co-ordinates that correspond to
    // the center of each molecule.
    std::vector<Particle> molecule_centers;

    for (int64_t mol = 0; mol < nmolecules; ++mol) {
      Particle mol_center(ndim);
      auto start_index = mol * molecule_size;
      auto end_index = mol * molecule_size + molecule_size;

      for (int axis = 0; axis < ndim; ++axis) {
        auto axis_min = get_axis_min(start_index, end_index, axis);
        auto axis_max = get_axis_max(start_index, end_index, axis);
        mol_center.coords[axis] = (axis_min + axis_max) / 2;
      }
      molecule_centers.push_back(mol_center);
    }

    // Sort the particles according to octants using an out-of-place sort (something like a merge sort).
    // Along with sorting, also generate cells that correspond to ranges of sorted particles and assign
    // various parameters such as start and stop index, hilbert index and level.
    const int64_t nmolecules_per_box = 1;
    std::vector<Particle> buffer = molecule_centers;
    sort_particles_and_build_tree(buffer.data(), molecule_centers.data(),
                                  0, nmolecules,
                                  0, tree_list,
                                  nmolecules_per_box, 0, false);

    // Partition the molecules using Hilbert indexing and sort them.
    int64_t max_level = 0;
    for (auto cell : tree_list) { max_level = std::max(max_level, cell.level); }

    // Sort the molecules based on the hilbert index generated during the tree generation.

    // Sort the electrons (actual Particles in the Domain) based on the sorted molecules.
  }

  void Domain::build_elses_tree(const int64_t molecule_size) {
    sort_elses_bodies(molecule_size);

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

    std::cout << "particles: " << particles.size() << std::endl;
    file.close();
  }

  void Domain::orb_split(Cell& cell,
                         const int64_t pstart,
                         const int64_t pend,
                         const int64_t max_nleaf,
                         const int64_t dim,
                         const int64_t level,
                         uint32_t level_index) {
    std::vector<double> Xmin(ndim, std::numeric_limits<double>::max()),
      Xmax(ndim, std::numeric_limits<double>::min()),
      cell_center(ndim, 0);

    // calculate the min and max points for this cell
    for (int64_t i = pstart; i < pend; ++i) {
      for (int k = 0; k < ndim; ++k) {
        if (Xmax[k] < particles[i].coords[k]) {
          Xmax[k] = particles[i].coords[k];
        }
        if (Xmin[k] > particles[i].coords[k]) {
          Xmin[k] = particles[i].coords[k];
        }
      }
    }

    // set the center point and radius
    cell.radius = 0;
    for (int64_t k = 0; k < ndim; ++k) {
      cell_center[k] = (double)(Xmax[k] + Xmin[k]) / 2.0;
      // set radius to the max distance along any dimension.
      cell.radius = std::max((Xmax[k] - Xmin[k]) / 2, cell.radius);
    }
    cell.center = cell_center;
    cell.start_index = pstart;
    cell.end_index = pend;
    cell.level = level;
    cell.level_index = level_index;

    // we have hit the leaf level. return without sorting.
    if (pend - pstart <= max_nleaf) {
      return;
    }

    std::sort(particles.begin() + pstart,
              particles.begin() + pend,
              [&](const Particle& lhs, const Particle& rhs) {
                return lhs.coords[dim] < rhs.coords[dim];
              });

    int64_t mid = ceil(double(pstart + pend) / 2.0);
    cell.cells.resize(2);
    orb_split(cell.cells[0], pstart, mid, max_nleaf, (dim+1) % ndim, level+1, level_index << 1);
    orb_split(cell.cells[1], mid, pend, max_nleaf, (dim+1) % ndim, level+1, (level_index << 1) + 1);
  }

  void
  Domain::build_tree(const int64_t max_nleaf) {
    orb_split(tree, 0, N, max_nleaf, 0, 0, 0);
  }
}
