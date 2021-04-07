#include "Hatrix/Hatrix.h"
using namespace Hatrix;

#include <iostream>


bool init_test() {
  int block_size = 16;
  Matrix A(block_size, block_size);

  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    if (A(i, j) != 0) return false;
  }
  return true;
}

bool copy_test() {
  int block_size = 16;
  Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Matrix A_copy(A);

  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    if (A(i, j) != A_copy(i, j)) return false;
  }
  return true;
}

bool shrink_test() {
  int block_size = 16;
  Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Matrix A_copy(A);

  int shrunk_size = 8;
  A.shrink(shrunk_size, shrunk_size);

  // Check result
  for (int i=0; i<shrunk_size; ++i) for (int j=0; j<shrunk_size; ++j) {
    if (A(i, j) != A_copy(i, j)) return false;
  }
  return true;
}

int main() {
  bool all_succeed = true;

  if (!init_test()) {
    std::cout << "Init test failed!\n";
    all_succeed = false;
  }

  if (!copy_test()) {
    std::cout << "Copy test failed!\n";
    all_succeed = false;
  }

  if (!shrink_test()) {
    std::cout << "Shrink test failed!\n";
    all_succeed = false;
  }
  return all_succeed ? 0 : 1;
}
