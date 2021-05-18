#pragma once
#include "Hatrix/classes/Matrix.h"

#include <cstdint>
using std::int64_t;
#include <tuple>
#include <unordered_map>


namespace std {

template<>
struct hash<std::tuple<int64_t, int64_t>> {
  size_t operator()(const std::tuple<int64_t, int64_t>& pair) const {
    int64_t first, second;
    std::tie(first, second) = pair;
    size_t first_hash = hash<int64_t>()(first);
    first_hash ^= (
      hash<int64_t>()(second) + 0x9e3779b97f4a7c17
       + (first_hash << 6) + (first_hash >> 2)
    );
    return first_hash;
  }
};

} // namespace std


namespace Hatrix {

class RowMap {
 private:
  std::unordered_map<int64_t, Matrix> map;
 public:
  Matrix& operator[](int64_t key);
  const Matrix& operator[](int64_t key) const;

  Matrix& operator()(int64_t key);
  const Matrix& operator()(int64_t key) const;

  void insert(int64_t key, Matrix&& matrix);
};
typedef RowMap ColMap;

class RowColMap {
 private:
  std::unordered_map<std::tuple<int64_t, int64_t>, Matrix> map;
 public:
  Matrix& operator()(int64_t row, int64_t col);
  const Matrix& operator()(int64_t row, int64_t col) const;

  Matrix& operator[](const std::tuple<int64_t, int64_t>& key);
  const Matrix& operator[](const std::tuple<int64_t, int64_t>& key) const;

  void insert(int64_t row, int64_t col, Matrix&& matrix);
  void insert(const std::tuple<int64_t, int64_t> key, Matrix&& matrix);
};

} // namespace Hatrix

