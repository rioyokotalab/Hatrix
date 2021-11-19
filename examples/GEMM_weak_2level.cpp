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

void matmul(Hatrix::HSS& A, Hatrix::HSS& B, Hatrix::HSS& C,
	    double alpha, double beta) {
  Hatrix::RowLevelMap Pca = cluster_bases_product(C.U, A.U, C.height);
  Hatrix::RowLevelMap Pab = cluster_bases_product(A.V, B.U, C.height);
  Hatrix::RowLevelMap Pbc = cluster_bases_product(B.V, C.V, C.height);

  //C_1/0,0 += A_1/0,0 * B_1/0,0 + A_1/0,1 * B_1/1,0
  Hatrix::Matrix W = A.S(0, 1, 1) * Pab(1, 1) * B.S(1, 0, 1);
  Hatrix::Matrix A_Ubig0 = A.Ubig(0, 1);
  auto A_Ubig0_splits = A_Ubig0.split(2, 1);
  auto A_Utransfer0_splits = A.U(0, 1).split(2, 1);
  Hatrix::Matrix B_Vbig0 = B.Vbig(0, 1);
  auto B_Vbig0_splits = B_Vbig0.split(2, 1);
  auto B_Vtransfer0_splits = B.V(0, 1).split(2, 1);
  
  //Recurse into C_2/0,0
  C.D(0, 0, 2) *= beta;
  Hatrix::matmul(A.D(0, 0, 2), B.D(0, 0, 2),
		 C.D(0, 0, 2), false, false, alpha);
  Hatrix::matmul(Hatrix::matmul(A.U(0, 2), A.S(0, 1, 2) * Pab(1, 2) * B.S(1, 0, 2)),
		 B.V(0, 2), C.D(0, 0, 2), false, true, alpha);
  Hatrix::matmul(A_Ubig0_splits[0], Hatrix::matmul(W, B_Vbig0_splits[0], false, true),
		 C.D(0, 0, 2), false, false, alpha);
  
  //Recurse into C_2/0,1
  C.S(0, 1, 2) *= beta;
  Hatrix::matmul(Hatrix::matmul(C.U(0, 2), A.D(0, 0, 2), true) * B.U(0, 2),
		 B.S(0, 1, 2) * Pbc(1, 2),
		 C.S(0, 1, 2), false, false, alpha);
  Hatrix::matmul(Pca(0, 2) * A.S(0, 1, 2),
		 Hatrix::matmul(A.V(1, 2), B.D(1, 1, 2), true) * C.V(1, 2),
		 C.S(0, 1, 2), false, false, alpha);
  Hatrix::matmul(Pca(0, 2) * A_Utransfer0_splits[0] * W,
		 Hatrix::matmul(B_Vtransfer0_splits[1], Pbc(1, 2), true),
		 C.S(0, 1, 2), false, false, alpha);
  //Recurse into C_2/1,0
  C.S(1, 0, 2) *= beta;
  Hatrix::matmul(Pca(1, 2) * A.S(1, 0, 2),
		 Hatrix::matmul(A.V(0, 2), B.D(0, 0, 2), true) * C.V(0, 2),
		 C.S(1, 0, 2), false, false, alpha);
  Hatrix::matmul(Hatrix::matmul(C.U(1, 2), A.D(1, 1, 2), true) * B.U(1, 2),
		 B.S(1, 0, 2) * Pbc(0, 2),
		 C.S(1, 0, 2), false, false, alpha);
  Hatrix::matmul(Pca(1, 2) * A_Utransfer0_splits[1] * W,
		 Hatrix::matmul(B_Vtransfer0_splits[0], Pbc(0, 2), true),
		 C.S(1, 0, 2), false, false, alpha);
  //Recurse into C_2/1,1
  C.D(1, 1, 2) *= beta;
  Hatrix::matmul(Hatrix::matmul(A.U(1, 2), A.S(1, 0, 2) * Pab(0, 2)),
		 Hatrix::matmul(B.S(0, 1, 2), B.V(1, 2), false, true),
		 C.D(1, 1, 2), false, false, alpha);
  Hatrix::matmul(A.D(1, 1, 2), B.D(1, 1, 2),
		 C.D(1, 1, 2), false, false, alpha);
  Hatrix::matmul(A_Ubig0_splits[1], Hatrix::matmul(W, B_Vbig0_splits[1], false, true),
		 C.D(1, 1, 2), false, false, alpha);

  //C_1/0,1 += A_1/0,0 * B_1/0,1
  C.S(0, 1, 1) *= beta;
  Hatrix::Matrix Shat_A00(C.rank, B.rank);
  Hatrix::Matrix C_Ubig0 = C.Ubig(0, 1);
  auto C_Ubig0_splits = C_Ubig0.split(2, 1);
  Hatrix::Matrix B_Ubig0 = B.Ubig(0, 1);
  auto B_Ubig0_splits = B_Ubig0.split(2, 1);
  Hatrix::matmul(Hatrix::matmul(C_Ubig0_splits[0], A.D(0, 0, 2), true),
		 B_Ubig0_splits[0], Shat_A00);
  Hatrix::matmul(Hatrix::matmul(C_Ubig0_splits[1], A.U(1, 2) * A.S(1, 0, 2), true),
		 Hatrix::matmul(A.V(0, 2), B_Ubig0_splits[0], true), Shat_A00);
  Hatrix::matmul(Hatrix::matmul(C_Ubig0_splits[0], A.U(0, 2) * A.S(0, 1, 2), true),
		 Hatrix::matmul(A.V(1, 2), B_Ubig0_splits[1], true), Shat_A00);
  Hatrix::matmul(Hatrix::matmul(C_Ubig0_splits[1], A.D(1, 1, 2), true),
		 B_Ubig0_splits[1], Shat_A00);
  Hatrix::matmul(Shat_A00, B.S(0, 1, 1) * Pbc(1, 1),
		 C.S(0, 1, 1), false, false, alpha);
  //C_1/0,1 += A_1/0,1 * B_1/1,1
  Hatrix::Matrix Shat_B11(A.rank, C.rank);
  Hatrix::Matrix A_Vbig1 = A.Vbig(1, 1);
  auto A_Vbig1_splits = A_Vbig1.split(2, 1);
  Hatrix::Matrix C_Vbig1 = C.Vbig(1, 1);
  auto C_Vbig1_splits = C_Vbig1.split(2, 1);
  Hatrix::matmul(Hatrix::matmul(A_Vbig1_splits[0], B.D(2, 2, 2), true),
		 C_Vbig1_splits[0], Shat_B11);
  Hatrix::matmul(Hatrix::matmul(A_Vbig1_splits[1], B.U(3, 2) * B.S(3, 2, 2), true),
		 Hatrix::matmul(B.V(2, 2), C_Vbig1_splits[0], true), Shat_B11);
  Hatrix::matmul(Hatrix::matmul(A_Vbig1_splits[0], B.U(2, 2) * B.S(2, 3, 2), true),
		 Hatrix::matmul(B.V(3, 2), C_Vbig1_splits[1], true), Shat_B11);
  Hatrix::matmul(Hatrix::matmul(A_Vbig1_splits[1], B.D(3, 3, 2), true),
		 C_Vbig1_splits[1], Shat_B11);
  Hatrix::matmul(Pca(0, 1) * A.S(0, 1, 1), Shat_B11,
		 C.S(0, 1, 1), false, false, alpha);

  //C_1/1,0 += A_1/1,0 * B_1/0,0
  C.S(1, 0, 1) *= beta;
  Hatrix::Matrix Shat_B00(A.rank, C.rank);
  Hatrix::Matrix A_Vbig0 = A.Vbig(0, 1);
  auto A_Vbig0_splits = A_Vbig0.split(2, 1);
  Hatrix::Matrix C_Vbig0 = C.Vbig(0, 1);
  auto C_Vbig0_splits = C_Vbig0.split(2, 1);
  Hatrix::matmul(Hatrix::matmul(A_Vbig0_splits[0], B.D(0, 0, 2), true),
		 C_Vbig0_splits[0], Shat_B00);
  Hatrix::matmul(Hatrix::matmul(A_Vbig0_splits[1], B.U(1, 2) * B.S(1, 0, 2), true),
		 Hatrix::matmul(B.V(0, 2), C_Vbig0_splits[0], true), Shat_B00);
  Hatrix::matmul(Hatrix::matmul(A_Vbig0_splits[0], B.U(0, 2) * B.S(0, 1, 2), true),
		 Hatrix::matmul(B.V(1, 2), C_Vbig0_splits[1], true), Shat_B00);
  Hatrix::matmul(Hatrix::matmul(A_Vbig0_splits[1], B.D(1, 1, 2), true),
		 C_Vbig0_splits[1], Shat_B00);
  Hatrix::matmul(Pca(1, 1) * A.S(1, 0, 1), Shat_B00,
		 C.S(1, 0, 1), false, false, alpha);
  //C_1/1,0 += A_1/1,1 * B_1/1,0
  Hatrix::Matrix Shat_A11(C.rank, B.rank);
  Hatrix::Matrix C_Ubig1 = C.Ubig(1, 1);
  auto C_Ubig1_splits = C_Ubig1.split(2, 1);
  Hatrix::Matrix B_Ubig1 = B.Ubig(1, 1);
  auto B_Ubig1_splits = B_Ubig1.split(2, 1);
  Hatrix::matmul(Hatrix::matmul(C_Ubig1_splits[0], A.D(2, 2, 2), true),
		 B_Ubig1_splits[0], Shat_A11);
  Hatrix::matmul(Hatrix::matmul(C_Ubig1_splits[1], A.U(3, 2) * A.S(3, 2, 2), true),
		 Hatrix::matmul(A.V(2, 2), B_Ubig1_splits[0], true), Shat_A11);
  Hatrix::matmul(Hatrix::matmul(C_Ubig1_splits[0], A.U(2, 2) * A.S(2, 3, 2), true),
		 Hatrix::matmul(A.V(3, 2), B_Ubig1_splits[1], true), Shat_A11);
  Hatrix::matmul(Hatrix::matmul(C_Ubig1_splits[1], A.D(3, 3, 2), true),
		 B_Ubig1_splits[1], Shat_A11);
  Hatrix::matmul(Shat_A11, B.S(1, 0, 1) * Pbc(0, 1),
		 C.S(1, 0, 1), false, false, alpha);

  //C_1/1,1 += A_1/1,0 * B_1/0,1 + A_1/1,1 * B_1/1,1
  Hatrix::Matrix Y = A.S(1, 0, 1) * Pab(0, 1) * B.S(0, 1, 1);
  Hatrix::Matrix A_Ubig1 = A.Ubig(1, 1);
  auto A_Ubig1_splits = A_Ubig1.split(2, 1);
  auto A_Utransfer1_splits = A.U(1, 1).split(2, 1);
  Hatrix::Matrix B_Vbig1 = B.Vbig(1, 1);
  auto B_Vbig1_splits = B_Vbig1.split(2, 1);
  auto B_Vtransfer1_splits = B.V(1, 1).split(2, 1);
  //Recurse into C_2/2,2
  C.D(2, 2, 2) *= beta;
  Hatrix::matmul(A.D(2, 2, 2), B.D(2, 2, 2),
		 C.D(2, 2, 2), false, false, alpha);
  Hatrix::matmul(Hatrix::matmul(A.U(2, 2), A.S(2, 3, 2) * Pab(3, 2) * B.S(3, 2, 2)),
		 B.V(2, 2), C.D(2, 2, 2), false, true, alpha);
  Hatrix::matmul(A_Ubig1_splits[0], Hatrix::matmul(Y, B_Vbig1_splits[0], false, true),
		 C.D(2, 2, 2), false, false, alpha);

  //Recurse into C_2/2,3
  C.S(2, 3, 2) *= beta;
  Hatrix::matmul(Hatrix::matmul(C.U(2, 2), A.D(2, 2, 2), true) * B.U(2, 2),
		 B.S(2, 3, 2) * Pbc(3, 2),
		 C.S(2, 3, 2), false, false, alpha);
  Hatrix::matmul(Pca(2, 2) * A.S(2, 3, 2),
		 Hatrix::matmul(A.V(3, 2), B.D(3, 3, 2), true) * C.V(3, 2),
		 C.S(2, 3, 2), false, false, alpha);
  Hatrix::matmul(Pca(2, 2) * A_Utransfer1_splits[0] * Y,
		 Hatrix::matmul(B_Vtransfer1_splits[1], Pbc(3, 2), true),
		 C.S(2, 3, 2), false, false, alpha);

  //Recurse into C_2/3,2
  C.S(3, 2, 2) *= beta;
  Hatrix::matmul(Pca(3, 2) * A.S(3, 2, 2),
		 Hatrix::matmul(A.V(2, 2), B.D(2, 2, 2), true) * C.V(2, 2),
		 C.S(3, 2, 2), false, false, alpha);
  Hatrix::matmul(Hatrix::matmul(C.U(3, 2), A.D(3, 3, 2), true) * B.U(3, 2),
		 B.S(3, 2, 2) * Pbc(2, 2),
		 C.S(3, 2, 2), false, false, alpha);
  Hatrix::matmul(Pca(3, 2) * A_Utransfer1_splits[1] * Y,
		 Hatrix::matmul(B_Vtransfer1_splits[0], Pbc(2, 2), true),
		 C.S(3, 2, 2), false, false, alpha);

  //Recurse into C_2/3,3
  C.D(3, 3, 2) *= beta;
  Hatrix::matmul(Hatrix::matmul(A.U(3, 2), A.S(3, 2, 2) * Pab(2, 2)),
		 Hatrix::matmul(B.S(2, 3, 2), B.V(3, 2), false, true),
		 C.D(3, 3, 2), false, false, alpha);
  Hatrix::matmul(A.D(3, 3, 2), B.D(3, 3, 2),
		 C.D(3, 3, 2), false, false, alpha);
  Hatrix::matmul(A_Ubig1_splits[1], Hatrix::matmul(Y, B_Vbig1_splits[1], false, true),
		 C.D(3, 3, 2), false, false, alpha);
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 8;
  int64_t height = 2;
  double pv;

  if (N % int(pow(2, height)) != 0 || rank > int(N / pow(2, height))) {
    std::cout << N << " % " << pow(2, height) << " != 0 || rank > leaf(" << int(N / pow(2, height))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t A_randpts, B_randpts, C_randpts;
  A_randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D
  B_randpts.push_back(equally_spaced_vector(N, 1.0, 2.0)); // 1D
  C_randpts.push_back(equally_spaced_vector(N, 3.0, 4.0)); // 1D
  pv = 1e-3 * (1.0/N); //Make diagonal value increasing proportional to N

  Hatrix::HSS A(A_randpts, pv, N, rank, height);
  Hatrix::HSS B(B_randpts, pv, N, rank, height);
  Hatrix::HSS C(C_randpts, pv, N, rank, height);
  
  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(A_randpts, N, N, 0, 0, pv);
  Hatrix::Matrix B_dense = Hatrix::generate_laplacend_matrix(B_randpts, N, N, 0, 0, pv);
  Hatrix::Matrix C_dense = Hatrix::generate_laplacend_matrix(C_randpts, N, N, 0, 0, pv);  
  
  matmul(A, B, C, 1, 1);
  Hatrix::matmul(A_dense, B_dense, C_dense, false, false, 1, 1);

  double construction_error = A.construction_rel_error(A_randpts);
  double matmul_error = C.rel_error(C_dense);
  std::cout <<"N=" <<N <<", rank=" <<rank <<", height=" <<height;
  std::cout <<", construction error=" <<construction_error <<", matmul error=" <<matmul_error <<"\n";

  Hatrix::Context::finalize();
  return 0;
}
