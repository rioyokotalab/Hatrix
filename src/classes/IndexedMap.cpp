#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {

Matrix& RowMap::operator[](int64_t key) { return map.at(key); }
const Matrix& RowMap::operator[](int64_t key) const { return map.at(key); }

Matrix& RowMap::operator()(int64_t key) { return (*this)[key]; }
const Matrix& RowMap::operator()(int64_t key) const { return (*this)[key]; }

void RowMap::insert(int64_t key, Matrix&& matrix) {
  map.insert({key, std::move(matrix)});
}

Matrix RowMap::extract(int64_t key) {
  Matrix out = std::move(map[key]);
  map.erase(key);
  return out;
}

Matrix& RowColMap::operator()(int64_t row, int64_t col) {
  return map.at({row, col});
}
const Matrix& RowColMap::operator()(int64_t row, int64_t col) const {
  return map.at({row, col});
}

Matrix& RowColMap::operator[](const std::tuple<int64_t, int64_t>& key) {
  return map.at(key);
}
const Matrix& RowColMap::operator[](
    const std::tuple<int64_t, int64_t>& key) const {
  return map.at(key);
}

void RowColMap::insert(int64_t row, int64_t col, Matrix&& matrix) {
  map.insert({{row, col}, std::move(matrix)});
}
void RowColMap::insert(const std::tuple<int64_t, int64_t>& key,
                       Matrix&& matrix) {
  map.insert({key, std::move(matrix)});
}

Matrix RowColMap::extract(int64_t row, int64_t col) {
  Matrix out = std::move(map[{row, col}]);
  map.erase({row, col});
  return out;
}
Matrix RowColMap::extract(const std::tuple<int64_t, int64_t>& key) {
  Matrix out = std::move(map[key]);
  map.erase(key);
  return out;
}

Matrix& RowColLevelMap::operator[](std::tuple<int64_t, int64_t, int64_t>& key) {
  return map.at(key);
}
const Matrix& RowColLevelMap::operator[](std::tuple<int64_t, int64_t, int64_t>& key) {
  return map.at(key);
}

Matrix& RowColLevelMap::operator()(int64_t row, int64_t col, int64_t level) {
  return (*this)[row, col, level];
}
const Matrix& RowColLevelMap::operator()(int64_t row, int64_t col, int64_t level) {
  return (*this)[row, col, level];
}

void RowColLevelMap::insert(int64_t row, int64_t col, int64_t level, Matrix&& matrix) {
  map.insert({row, col, level}, matrix);
}
void RowColLevelMap::insert(const std::tuple<int64_t, int64_t, int64_t>& key, Matrix&& matrix) {
  map.insert(key, matrix);
}

Matrix RowColLevelMap::extract(int64_t row, int64_t col, int64_t level) {
  Matrix out = std::move(map[{row, col, level}]);
  map.erase({row, col, level});
  return out;
}
Matrix RowColLevelMap::extract(const std::tuple<int64_t, int64_t, int64_t>& key) {
  Matrix out = std::move(map[key]);
  map.erase(key);
  return out;
}

}  // namespace Hatrix
