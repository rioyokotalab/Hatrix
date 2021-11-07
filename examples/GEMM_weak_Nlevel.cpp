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
    RowLevelMap U;
    ColLevelMap V;
    RowColLevelMap D, S;
    int64_t N, rank, leaf_size, height;
    double pv;

    HSS(const randvec_t& randpts, double pv, int64_t N, int64_t rank, int64_t height):
      pv(pv), N(N), rank(rank), height(height), leaf_size(N/pow(2, height))
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

    double construction_rel_error(const randvec_t& randpts) {
      double diff = 0, norm = 0, fnorm, fdiff;
      for(int64_t level = height; level > 0; level--) {
	int64_t num_nodes = pow(2, level);
	int64_t node_size = N / num_nodes;
	for(int64_t row = 0; row < num_nodes; row++) {
	  //Leaf diagonals
	  if(level == height) {
	    Matrix Dii = generate_laplacend_matrix(randpts, node_size, node_size,
						   row*node_size, row*node_size, pv);
	    fnorm = Hatrix::norm(Dii);
	    norm += fnorm * fnorm;
	  }
	  //Off-diagonals
	  int64_t col = row % 2 == 0 ? row + 1 : row - 1;
	  Matrix Dij = generate_laplacend_matrix(randpts, node_size, node_size,
						 row*node_size, col*node_size, pv);
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

    double rel_error(Matrix dense) {
      double diff = 0, norm = 0, fnorm, fdiff;
      for(int64_t level = height; level > 0; level--) {
	int64_t num_nodes = pow(2, level);
	int64_t node_size = N / num_nodes;
	auto dense_splits = dense.split(num_nodes, num_nodes);
	for(int64_t row = 0; row < num_nodes; row++) {
	  //Leaf diagonals
	  if(level == height) {
	    Matrix& Dii = dense_splits[row*num_nodes + row];
	    fnorm = Hatrix::norm(Dii);
	    norm += fnorm * fnorm;
	    fdiff = Hatrix::norm(D(row, row, level) - Dii);
	    diff += fdiff * fdiff;
	  }
	  
	  //Off-diagonals
	  int64_t col = row % 2 == 0 ? row + 1 : row - 1;
	  Matrix& Dij = dense_splits[row*num_nodes + col];
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
						 row*node_size, col*node_size, pv);
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
						 row*node_size, col*node_size, pv);
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
					   node*leaf_size, node*leaf_size, pv));
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
					     row*leaf_size, col*leaf_size, pv);
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
					     row*node_size, col*node_size, pv);
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

//Compute product of cluster bases W1^T x W2
Hatrix::RowLevelMap cluster_bases_product(Hatrix::RowLevelMap& W1, Hatrix::RowLevelMap& W2,
					  int64_t height) {
  Hatrix::RowLevelMap P;
  for(int64_t level = height; level > 0; level--) {
    int64_t num_nodes = pow(2, level);
    for(int64_t node = 0; node < num_nodes; node++) {
      if(level == height) { //Leaf
	P.insert(node, level, Hatrix::matmul(W1(node, level), W2(node, level), true));
      }
      else { //Non-leaf
	Hatrix::Matrix P_node(W1(node, level).cols, W2(node, level).cols);
	auto W1transfer_splits = W1(node, level).split(2, 1);
	auto W2transfer_splits = W2(node, level).split(2, 1);

	int64_t child_level = level + 1;
	int64_t child1 = 2*node;
	int64_t child2 = 2*node + 1;
	Hatrix::matmul(W1transfer_splits[0], Hatrix::matmul(P(child1, child_level), W2transfer_splits[0]),
		       P_node, true, false, 1., 1.);
	Hatrix::matmul(W1transfer_splits[1], Hatrix::matmul(P(child2, child_level), W2transfer_splits[1]),
		       P_node, true, false, 1., 1.);
	P.insert(node, level, std::move(P_node));
      }
    }
  }
  return P;
}

//Precompute S updates from U^T x A x V
Hatrix::RowColLevelMap forward_transform(Hatrix::RowLevelMap& U, Hatrix::RowLevelMap& Pu,
					 Hatrix::HSS& A,
					 Hatrix::RowLevelMap& V, Hatrix::RowLevelMap& Pv) {
  Hatrix::RowColLevelMap S;
  for(int64_t level = A.height; level > 0; level--) {
    int64_t num_nodes = pow(2, level);
    for(int node = 0; node < num_nodes; node++) {
      //Diagonal
      if(level == A.height) { //Leaf
	S.insert(node, node, level,
		 Hatrix::matmul(Hatrix::matmul(U(node, level), A.D(node, node, level), true),
				V(node, level)));
      }
      else {
	int64_t child_level = level + 1;
	int64_t child1 = 2*node;
	int64_t child2 = 2*node + 1;
	Hatrix::Matrix S_diag(A.rank, A.rank);
	auto Utransfer_splits = U(node, level).split(2, 1);
	auto Vtransfer_splits = V(node, level).split(2, 1);
	
	Hatrix::Matrix tmp1(A.rank, A.rank);
	Hatrix::matmul(S(child1, child1, child_level), Vtransfer_splits[0], tmp1,
		       false, false, 1., 1.);
	Hatrix::matmul(S(child1, child2, child_level), Vtransfer_splits[1], tmp1,
		       false, false, 1., 1.);
	Hatrix::matmul(Utransfer_splits[0], tmp1, S_diag, true, false, 1., 1.);

	Hatrix::Matrix tmp2(A.rank, A.rank);
	Hatrix::matmul(S(child2, child1, child_level), Vtransfer_splits[0], tmp2,
		       false, false, 1., 1.);
	Hatrix::matmul(S(child2, child2, child_level), Vtransfer_splits[1], tmp2,
		       false, false, 1., 1.);
	Hatrix::matmul(Utransfer_splits[1], tmp2, S_diag, true, false, 1., 1.);
	S.insert(node, node, level, std::move(S_diag));
      }
      //Off-diagonal
      int64_t sibling = node % 2 == 0 ? node + 1 : node - 1;
      S.insert(node, sibling, level,
	       Hatrix::matmul(Hatrix::matmul(Pu(node, level), A.S(node, sibling, level)),
			      Pv(sibling, level)));
    }
  }
  return S;
}

void backward_transform(Hatrix::HSS& C, Hatrix::RowColLevelMap& Sc,
			Hatrix::RowLevelMap& U, Hatrix::RowLevelMap& Pu,
			Hatrix::RowLevelMap& V, Hatrix::RowLevelMap& Pv) {
  for(int64_t level = 1; level <= C.height; level++) {
    int num_nodes = pow(2, level);
    for(int64_t node = 0; node < num_nodes; node++) {
      //Diagonal      
      if(level == C.height) { //Leaf
	Hatrix::matmul(U(node, level) * Sc(node, node, level), V(node, level),
		       C.D(node, node, level), false, true, 1., 1.);
      }
      else { //Non-leaf
	int64_t child_level = level + 1;
	int64_t child1 = 2*node;
	int64_t child2 = 2*node + 1;
	auto Utransfer_splits = U(node, level).split(2, 1);
	auto Vtransfer_splits = V(node, level).split(2, 1);
	//Add parent contribution to children
	Hatrix::matmul(Utransfer_splits[0] * Sc(node, node, level),
		       Vtransfer_splits[0], Sc(child1, child1, child_level),
		       false, true, 1., 1.);
	Hatrix::matmul(Utransfer_splits[0] * Sc(node, node, level),
		       Vtransfer_splits[1], Sc(child1, child2, child_level),
		       false, true, 1., 1.);
	Hatrix::matmul(Utransfer_splits[1] * Sc(node, node, level),
		       Vtransfer_splits[0], Sc(child2, child1, child_level),
		       false, true, 1., 1.);
	Hatrix::matmul(Utransfer_splits[1] * Sc(node, node, level),
		       Vtransfer_splits[1], Sc(child2, child2, child_level),
		       false, true, 1., 1.);
      }
      //Off-diagonal
      int64_t sibling = node % 2 == 0 ? node + 1 : node - 1;
      Hatrix::matmul(Pu(node, level), Sc(node, sibling, level) * Pv(sibling, level),
		     C.S(node, sibling, level), false, false, 1., 1.);
    }
  }
}
							 

void matmul(Hatrix::HSS& A, Hatrix::HSS& B, Hatrix::HSS& C,
	    double alpha, double beta) {
  auto Pca = cluster_bases_product(C.U, A.U, C.height);
  auto Pab = cluster_bases_product(A.V, B.U, C.height);
  auto Pbc = cluster_bases_product(B.V, C.V, C.height);
  auto Sa = forward_transform(C.U, Pca, A, B.U, Pab);
  auto Sb = forward_transform(A.V, Pab, B, C.V, Pbc);
  Hatrix::RowColLevelMap Sc;

  for(int64_t level = C.height; level > 0; level--) {
    int64_t num_nodes = pow(2, level);
    for(int64_t node = 0; node < num_nodes; node++) {
      int64_t sibling = node % 2 == 0 ? node + 1 : node - 1;
      //Diagonal
      if(level == C.height) { //Leaf
	Hatrix::matmul(A.D(node, node, level), B.D(node, node, level),
		       C.D(node, node, level), false, false, 1., 1.);
      }
      Sc.insert(node, node, level,
		A.S(node, sibling, level) * Pab(sibling, level) *
		B.S(sibling, node, level));
      //Off-diagonal
      Hatrix::matmul(Sa(node, node, level),
		     B.S(node, sibling, level) * Pbc(sibling, level),
		     C.S(node, sibling, level), false, false, 1., 1.);
      Hatrix::matmul(Pca(node, level) * A.S(node, sibling, level),
		     Sb(sibling, sibling, level),
		     C.S(node, sibling, level), false, false, 1., 1.);
      Sc.insert(node, sibling, level,
		Hatrix::Matrix(A.U(node, level).cols, B.V(sibling, level).cols));
    }
  }
  backward_transform(C, Sc, A.U, Pca, B.V, Pbc);
}

double cond(Hatrix::Matrix A) {
  int64_t s_dim = A.min_dim();
  Hatrix::Matrix U(A.rows, s_dim);
  Hatrix::Matrix S(s_dim, s_dim);
  Hatrix::Matrix V(s_dim, A.cols);
  Hatrix::svd(A, U, S, V);
  return S(0, 0)/S(s_dim-1, s_dim-1);
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 8;
  int64_t height = argc > 3 ? atoi(argv[3]) : 2;
  double pv;

  Hatrix::Context::init();
  randvec_t A_randpts, B_randpts, C_randpts;
  A_randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D
  B_randpts.push_back(equally_spaced_vector(N, 1.0, 2.0)); // 1D
  C_randpts.push_back(equally_spaced_vector(N, 3.0, 4.0)); // 1D

  pv = 1e-3 * (1.0/N);
  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(A_randpts, N, N, 0, 0, pv);
  Hatrix::HSS A(A_randpts, pv, N, rank, height);
  double cond_A = cond(A_dense);

  Hatrix::Matrix B_dense = Hatrix::generate_laplacend_matrix(B_randpts, N, N, 0, 0, pv);
  Hatrix::HSS B(B_randpts, pv, N, rank, height);
  double cond_B = cond(B_dense);

  pv = 1e-3;
  Hatrix::Matrix C_dense = Hatrix::generate_laplacend_matrix(C_randpts, N, N, 0, 0, pv);
  Hatrix::HSS C(C_randpts, pv, N, rank, height);
  double cond_C = cond(C_dense);
  
  matmul(A, B, C, 1, 1);
  Hatrix::matmul(A_dense, B_dense, C_dense, false, false, 1, 1);

  double construction_error = A.construction_rel_error(A_randpts);
  double matmul_error = C.rel_error(C_dense);
  std::cout <<"N=" <<N <<", rank=" <<rank <<", height=" <<height <<"\n";
  std::cout <<"cond(A)=" <<cond_A <<", cond(B)=" <<cond_B <<", cond(C)=" <<cond_C <<"\n";
  std::cout <<"construction error=" <<construction_error <<", matmul error=" <<matmul_error <<"\n";

  Hatrix::Context::finalize();
  return 0;
}
