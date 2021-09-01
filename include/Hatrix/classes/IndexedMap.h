#pragma once
#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "Hatrix/classes/Matrix.h"

namespace std {

template <>
struct hash<std::tuple<int64_t, int64_t>> {
  size_t operator()(const std::tuple<int64_t, int64_t>& pair) const {
    int64_t first, second;
    std::tie(first, second) = pair;
    size_t first_hash = hash<int64_t>()(first);
    first_hash ^= (hash<int64_t>()(second) + 0x9e3779b97f4a7c17 +
                   (first_hash << 6) + (first_hash >> 2));
    return first_hash;
  }
};

template <>
struct hash<std::tuple<int64_t, int64_t, int64_t>> {
  size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& pair) const {
    int64_t first, second, third;
    std::tie(first, second, third) = pair;
    size_t first_hash = hash<int64_t>()(first);
    size_t second_hash = first_hash ^ (hash<int64_t>()(second) + 0x9e3779b97f4a7c17 +
                                       (first_hash << 6) + (first_hash >> 2));
    size_t third_hash = second_hash ^ (hash<int64_t>()(third) + 0x9e3779b97f4a7c17 +
                                       (second_hash << 6) + (second_hash >> 2));
    return third_hash;
  }
};

}  // namespace std

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

  Matrix extract(int64_t key);
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
  void insert(const std::tuple<int64_t, int64_t>& key, Matrix&& matrix);

  Matrix extract(int64_t row, int64_t col);
  Matrix extract(const std::tuple<int64_t, int64_t>& key);
};

// RowLevel and ColLevel also use a <int, int> tuple which is same as RowCol
using RowLevelMap = RowColMap;
using ColLevelMap = RowColMap;

class RowColLevelMap {
private:
  std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, Matrix> map;

public:
  Matrix& operator()(int64_t row, int64_t col, int64_t level);
  const Matrix& operator()(int64_t row, int64_t col, int64_t level) const;

  Matrix& operator[](const std::tuple<int64_t, int64_t, int64_t>& key);
  const Matrix& operator[](const std::tuple<int64_t, int64_t, int64_t>& key) const;

  void insert(int64_t row, int64_t col, int64_t level, Matrix&& matrix);
  void insert(const std::tuple<int64_t, int64_t, int64_t>& key, Matrix&& matrix);

  Matrix extract(int64_t row, int64_t col, int64_t level);
  Matrix extract(const std::tuple<int64_t, int64_t, int64_t>& key);
};


}  // namespace Hatrix
