#include <cmath>

#include "Hatrix/classes/HSS.h"

namespace Hatrix {
  void HSS::generate_leaf_nodes() {
    int level = height - 1;
    int nleaf_nodes = pow(2, level);
    for (int node = 0; node < nleaf_nodes; ++node) {
      // Generate diagonal dense blocks

    }
  }

  HSS::HSS(int _N, int _rank, int _height) :
    N(_N), rank(_rank), height(_height) {
    generate_leaf_nodes();
  }
}
