#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

#include "Hatrix/Hatrix.h"

using vec = std::vector<int64_t>;
using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class HSS {
  public:
    RowLevelMap U, Uc;
    ColLevelMap V;
    RowColLevelMap D, S;
    int64_t N, rank, leaf_size, height;

    HSS(const randvec_t& randpts, int64_t N, int64_t rank, int64_t height):
      N(N), rank(rank), height(height), leaf_size(N/pow(2, height))
    {
      generate_leaf_nodes(randpts);
      RowLevelMap Uchild = U;
      ColLevelMap Vchild = V;
      for(int64_t level = height-1; level > 0; level--) {
	std::tie(Uchild, Vchild) = generate_non_leaf_nodes(randpts, level, Uchild, Vchild);
      }
    }

    Matrix Ubig(int64_t row, int64_t level) {
      if(level == height) return U(row, level);

      int64_t child1 = 2*row;
      Matrix Ubig_child1 = Ubig(child1, level+1);
      int64_t child2 = 2*row + 1;
      Matrix Ubig_child2 = Ubig(child2, level+1);

      int64_t node_size = Ubig_child1.rows + Ubig_child2.rows;
      Matrix U_big(node_size, rank);
      auto U_big_splits = U_big.split(vec{Ubig_child1.rows}, vec());
      auto transfer_matrix_splits = U(row, level).split(2, 1);
      matmul(Ubig_child1, transfer_matrix_splits[0], U_big_splits[0]);
      matmul(Ubig_child2, transfer_matrix_splits[1], U_big_splits[1]);
      return U_big;
    }

    Matrix Vbig(int64_t col, int64_t level) {
      if(level == height) return V(col, level);

      int64_t child1 = 2*col;
      Matrix Vbig_child1 = Vbig(child1, level+1);
      int64_t child2 = 2*col + 1;
      Matrix Vbig_child2 = Vbig(child2, level+1);

      int64_t node_size = Vbig_child1.rows + Vbig_child2.rows;
      Matrix V_big(node_size, rank);
      auto V_big_splits = V_big.split(vec{Vbig_child1.rows}, vec());
      auto transfer_matrix_splits = V(col, level).split(2, 1);
      matmul(Vbig_child1, transfer_matrix_splits[0], V_big_splits[0]);
      matmul(Vbig_child2, transfer_matrix_splits[1], V_big_splits[1]);
      return V_big;
    }

    Hatrix::Matrix Uf(int64_t row, int64_t level) {
      Hatrix::Matrix Uf(U(row, level).rows, U(row, level).rows);
      int64_t c_size = U(row, level).rows - rank;
      if(c_size == 0) return U(row, level);
      
      vec col_split_indices = vec{c_size};
      auto Uf_splits = Uf.split(vec(), col_split_indices);
      Uf_splits[0] = Uc(row, level);
      Uf_splits[1] = U(row, level);
      return Uf;
    }

    double construction_rel_error(const randvec_t& randpts) {
      double diff = 0, norm = 0, fnorm, fdiff;
      for(int64_t level = height; level > 0; level--) {
	int64_t num_nodes = pow(2, level);
	int64_t node_size = N / num_nodes;
	for(int64_t row = 0; row < num_nodes; row++) {
	  //Leaf diagonals
	  if(level == height) {
	    Matrix Dii = generate_laplacend_matrix(randpts, node_size, node_size,
						   row*node_size, row*node_size);
	    fnorm = Hatrix::norm(Dii);
	    norm += fnorm * fnorm;
	  }
	  //Off-diagonals
	  int64_t col = row % 2 == 0 ? row + 1 : row - 1;
	  Matrix Dij = generate_laplacend_matrix(randpts, node_size, node_size,
						 row*node_size, col*node_size);
	  fnorm = Hatrix::norm(Dij);
	  norm += fnorm * fnorm;
	  Matrix Urow = Ubig(row, level);
	  Matrix Vcol = Vbig(col, level);
	  Matrix Aij = matmul(matmul(Urow, S(row, col, level)), Vcol, false, true);
	  fdiff = Hatrix::norm(Aij - Dij);
	  diff += fdiff * fdiff;
	}
      }
      return std::sqrt(diff/norm);
    }

  private:
    Matrix admissible_row_slice(int64_t row, int64_t level,
				const randvec_t& randpts) {
      int64_t num_nodes = pow(2, level);
      int64_t node_size = N / num_nodes;	
      Matrix row_slice(node_size, (num_nodes - 1) * node_size);
      auto row_slice_splits = row_slice.split(1, num_nodes - 1);
      
      int64_t j = 0;
      for(int64_t col = 0; col < num_nodes; col++) {
	if(row == col) continue;
	Matrix block = generate_laplacend_matrix(randpts, node_size, node_size,
						 row*node_size, col*node_size);
	row_slice_splits[j++] = block;
      }
      return row_slice;
    }

    Matrix admissible_column_slice(int64_t col, int64_t level,
				   const randvec_t& randpts) {
      int64_t num_nodes = pow(2, level);
      int64_t node_size = N / num_nodes;	
      Matrix col_slice((num_nodes - 1) * node_size, node_size);
      auto col_slice_splits = col_slice.split(num_nodes - 1, 1);
      
      int64_t i = 0;
      for(int64_t row = 0; row < num_nodes; row++) {
	if(row == col) continue;
	Matrix block = generate_laplacend_matrix(randpts, node_size, node_size,
						 row*node_size, col*node_size);
	col_slice_splits[i++] = block;
      }
      return col_slice;
    }

    Matrix generate_leaf_column_bases(int64_t row, const randvec_t& randpts) {
      Matrix row_slice = admissible_row_slice(row, height, randpts);
      Matrix U;
      std::tie(U, std::ignore, std::ignore, std::ignore) = truncated_svd(row_slice, rank);
      return U;
    }

    Matrix generate_leaf_row_bases(int64_t col, const randvec_t& randpts) {
      Matrix col_slice = admissible_column_slice(col, height, randpts);
      Matrix V;
      std::tie(std::ignore, std::ignore, V, std::ignore) = truncated_svd(col_slice, rank);
      return transpose(V);
    }

    void generate_leaf_nodes(const randvec_t& randpts) {
      int64_t num_nodes = pow(2, height);
      //Generate dense diagonal
      for(int64_t node = 0; node < num_nodes; node++) {
	D.insert(node, node, height,
		 generate_laplacend_matrix(randpts, leaf_size, leaf_size,
					   node*leaf_size, node*leaf_size));
      }
      //Generate U and V bases
      for(int64_t node = 0; node < num_nodes; node++) {
	U.insert(node, height,
		 generate_leaf_column_bases(node, randpts));
	V.insert(node, height,
		 generate_leaf_row_bases(node, randpts));
      }
      //Generate coupling matrices
      for(int64_t row = 0; row < num_nodes; row++) {
	int64_t col = row % 2 == 0 ? row + 1 : row - 1;
	Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
					     row*leaf_size, col*leaf_size);
	S.insert(row, col, height,
		 matmul(matmul(U(row, height), D, true), V(col, height)));
      }
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_non_leaf_nodes(const randvec_t& randpts,
								 int64_t level,
								 RowLevelMap& Ubig_child,
								 ColLevelMap& Vbig_child) {
      int64_t num_nodes = pow(2, level);
      int64_t node_size = N / num_nodes;
      int64_t child_level = level + 1;
      int64_t child_num_nodes = pow(2, child_level);
      
      //Save the generated bases of the current level and pass them to this
      //function again for generating upper level nodes
      RowLevelMap Ubig_node;
      ColLevelMap Vbig_node;
      
      //Generate transfer matrices
      for(int64_t node = 0; node < num_nodes; node++) {
	int64_t child1 = 2*node;
	int64_t child2 = 2*node + 1;
	
	//U transfer matrix
	Matrix& Ubig_child1 = Ubig_child(child1, child_level);
	Matrix& Ubig_child2 = Ubig_child(child2, child_level);
	Matrix row_slice = admissible_row_slice(node, level, randpts);
	auto row_slice_splits = row_slice.split(2, 1);
	Matrix projected_row_slice(Ubig_child1.cols + Ubig_child2.cols,
				   row_slice.cols);
	auto projected_row_slice_splits = projected_row_slice.split(2, 1);
	matmul(Ubig_child1, row_slice_splits[0],
	       projected_row_slice_splits[0], true);
	matmul(Ubig_child2, row_slice_splits[1],
	       projected_row_slice_splits[1], true);
	
	Matrix Utransfer;
	std::tie(Utransfer, std::ignore, std::ignore, std::ignore) =
	  truncated_svd(projected_row_slice, rank);
	U.insert(node, level, std::move(Utransfer));
	
	//V transfer matrix
	Matrix& Vbig_child1 = Vbig_child(child1, child_level);
	Matrix& Vbig_child2 = Vbig_child(child2, child_level);
	Matrix col_slice = admissible_column_slice(node, level, randpts);
	auto col_slice_splits = col_slice.split(1, 2);
	Matrix projected_col_slice(col_slice.rows,
				   Vbig_child1.cols + Vbig_child2.cols);
	auto projected_col_slice_splits = projected_col_slice.split(1, 2);
	matmul(col_slice_splits[0], Vbig_child1, projected_col_slice_splits[0]);
	matmul(col_slice_splits[1], Vbig_child2, projected_col_slice_splits[1]);
	
	Matrix Vtransfer;
	std::tie(std::ignore, std::ignore, Vtransfer, std::ignore) =
	  truncated_svd(projected_col_slice, rank);
	V.insert(node, level, transpose(Vtransfer));
	
	//Generate bases to passed for generating upper level transfer matrices
	auto Utransfer_splits = U(node, level).split(2, 1);
	Matrix U_big(node_size, rank);
	auto U_big_splits = U_big.split(2, 1);
	matmul(Ubig_child1, Utransfer_splits[0], U_big_splits[0]);
	matmul(Ubig_child2, Utransfer_splits[1], U_big_splits[1]);
	Ubig_node.insert(node, level, std::move(U_big));
	
	auto Vtransfer_splits = V(node, level).split(2, 1);
	Matrix V_big(node_size, rank);
	auto V_big_splits = V_big.split(2, 1);
	matmul(Vbig_child1, Vtransfer_splits[0], V_big_splits[0]);
	matmul(Vbig_child2, Vtransfer_splits[1], V_big_splits[1]);
	Vbig_node.insert(node, level, std::move(V_big));
      }
      
      //Generate coupling matrices
      for(int64_t row = 0; row < num_nodes; row++) {
	int64_t col = row % 2 == 0 ? row + 1 : row - 1;
	Matrix D = generate_laplacend_matrix(randpts, node_size, node_size,
					     row*node_size, col*node_size);
	S.insert(row, col, level,
		 matmul(matmul(Ubig_node(row, level), D, true),
			Vbig_node(col, level)));
      }

      return {Ubig_node, Vbig_node};
    }
  };
}

std::vector<double> equally_spaced_vector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

Hatrix::Matrix make_complement(const Hatrix::Matrix& _U) {
  Hatrix::Matrix U(_U);
  int c_size = U.rows - U.cols;
  if(c_size == 0) return Hatrix::Matrix(0, 0);
  
  Hatrix::Matrix Q(U.rows, U.rows);
  Hatrix::Matrix R(U.rows, U.cols);
  Hatrix::qr(U, Q, R);

  auto Q_splits = Q.split(vec(), vec{U.cols});
  Hatrix::Matrix Uc(U.rows, U.rows - U.cols);
  Uc = Q_splits[1];
  return Uc;
}

Hatrix::Matrix& get_oo_part(Hatrix::HSS& A,
			   int64_t row, int64_t col, int64_t level) {
  if(row == col) { //Dense diagonal
    int64_t c_size = A.D(row, col, level).rows - A.rank;
    auto D_splits = A.D(row, col, level).split(vec{c_size}, vec{c_size});
    return D_splits[3];
  }
  else { //Low-rank off-diagonal
    return A.S(row, col, level);
  }
}

void partial_ldl_diag(Hatrix::HSS& A, int64_t node, int64_t level) {
  int64_t c_size = A.D(node, node, level).rows - A.rank;
  auto D_splits = A.D(node, node, level).split(vec{c_size}, vec{c_size});
  Hatrix::ldl(D_splits[0]);
  Hatrix::solve_triangular(D_splits[0], D_splits[2], Hatrix::Right,
			   Hatrix::Lower, true, true, 1.);
  Hatrix::solve_diagonal(D_splits[0], D_splits[2], Hatrix::Right, 1.);
  //Compute Schur's complement
  Hatrix::Matrix L_oc(D_splits[2].rows, D_splits[2].cols);
  L_oc = D_splits[2]; // Hatrix::Matrix L_oc(D_splits[2]) doesn't work due to current copy-constructor
  column_scale(L_oc, D_splits[0]); //L*D
  Hatrix::matmul(L_oc, D_splits[2], D_splits[3], false, true, -1., 1.);
}

void factorize(Hatrix::HSS& A) {
  //Bottom up from leaf to root
  for(int64_t level = A.height; level > 0; level--) {
    int64_t num_nodes = pow(2, level);
    for(int64_t node = 0; node < num_nodes; node++) {
      A.Uc.insert(node, level, make_complement(A.U(node, level)));
      Hatrix::Matrix &D = A.D(node, node, level);
      D = Hatrix::matmul(Hatrix::matmul(A.Uf(node, level), D, true),
			 A.Uf(node, level));
      int64_t c_size = D.rows - A.rank;
      if(c_size > 0) partial_ldl_diag(A, node, level);
    }
    //Merge child oo parts into parent
    int64_t parent_level = level - 1;
    int64_t num_parent_nodes = pow(2, parent_level);
    for(int64_t parent_node = 0; parent_node < num_parent_nodes; parent_node++) {
      int64_t child1 = 2 * parent_node;
      int64_t child2 = 2 * parent_node + 1;
      Hatrix::Matrix child_oo(2 * A.rank, 2 * A.rank);
      auto child_oo_splits = child_oo.split(2, 2);
      child_oo_splits[0] = get_oo_part(A, child1, child1, level);
      child_oo_splits[1] = get_oo_part(A, child1, child2, level);
      child_oo_splits[2] = get_oo_part(A, child2, child1, level);
      child_oo_splits[3] = get_oo_part(A, child2, child2, level);
      A.D.insert(parent_node, parent_node, parent_level, std::move(child_oo));
    }
  }
  Hatrix::ldl(A.D(0, 0, 0));
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 8;
  int64_t height = 2;

  if (N % int(pow(2, height)) != 0 || rank > int(N / pow(2, height))) {
    std::cout << N << " % " << pow(2, height) << " != 0 || rank > leaf(" << int(N / pow(2, height))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS A(randpts, N, rank, height);
  factorize(A);
  
  std::cout <<"N=" <<N <<", rank=" <<rank <<", height=" <<height;
  std::cout <<", construction error=" <<A.construction_rel_error(randpts) <<"\n";

  Hatrix::Context::finalize();
  return 0;
}
