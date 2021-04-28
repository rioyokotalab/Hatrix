#include "Hatrix/util/matrix_generators.h"

#include <cmath>
#include <cstdint>
using std::int64_t;
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <stdexcept>


namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Matrix out(rows, cols);
  for (int64_t i=0; i<rows; ++i) {
    for (int64_t j=0; j<cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols) {
  // TODO: Might want more sophisticated method, specify rate of decay of
  // singular values etc...
  Matrix out(rows, cols);
  for (int64_t i=0; i<rows; ++i) {
    for (int64_t j=0; j<cols; ++j) {
      out(i, j) = 1.0 / std::abs(i - j + out.max_dim());
    }
  }
  return out;
}

Matrix generate_from_csv(std::string filename, char delimiter, int64_t rows, int64_t cols) {
  std::vector<std::vector<double>> data;
  std::ifstream datafile(filename);
  if (!datafile.is_open())
    throw std::runtime_error("Could not open file \'"+filename+"\'");

  std::string line;
  double val;
  int64_t len, n = 0;
  while(std::getline(datafile, line)){
    std::stringstream ss(line);
    std::vector<double> row;
    while(ss >> val){
      row.push_back(val);
      if(ss.peek() == delimiter)
        ss.ignore();
    } 
    len = row.size();
    if (len > n)
      n = len;
    data.push_back(row);
  }
  datafile.close();

  int64_t m = data.size();
  
  Matrix out(rows > 0 ? rows : m, cols > 0 ? cols : n);
  rows = out.rows < m ? out.rows : m;
  //only fill values that are actually there
  for (int64_t i=0; i<rows; ++i){
    std::vector<double> row = data.at(i);
    cols = row.size() < out.cols ? row.size() : out.cols; 
    for (int64_t j=0; j<cols; ++j){
      out(i, j) = row.at(j);
    }
  }
  return out;
}

} // namespace Hatrix
