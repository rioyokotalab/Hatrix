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


template<class T>
T& RowColMap<T>::operator()(int64_t row, int64_t col) {
  return map.at({row, col});
}
template<class T>
const T& RowColMap<T>::operator()(int64_t row, int64_t col) const {
  return map.at({row, col});
}

template<class T>
T& RowColMap<T>::operator[](const std::tuple<int64_t, int64_t>& key) {
  return map.at(key);
}
template<class T>
const T& RowColMap<T>::operator[](
    const std::tuple<int64_t, int64_t>& key) const {
  return map.at(key);
}

template<class T>
void RowColMap<T>::insert(int64_t row, int64_t col, T&& value) {
  map.insert({{row, col}, std::move(value)});
}
template<class T>
void RowColMap<T>::insert(const std::tuple<int64_t, int64_t>& key,
                       T&& value) {
  map.insert({key, std::move(value)});
}

template<class T>
T RowColMap<T>::extract(int64_t row, int64_t col) {
  T out = std::move(map[{row, col}]);
  map.erase({row, col});
  return out;
}
template<class T>
T RowColMap<T>::extract(const std::tuple<int64_t, int64_t>& key) {
  T out = std::move(map[key]);
  map.erase(key);
  return out;
}

// explicit instatiation
template class RowColMap<bool>;
template class RowColMap<Matrix>;

Matrix& RowColLevelMap::operator[](const std::tuple<int64_t, int64_t, int64_t>& key) {
  return map.at(key);
}
const Matrix& RowColLevelMap::operator[](const std::tuple<int64_t, int64_t, int64_t>& key) const {
  return map.at(key);
}

Matrix& RowColLevelMap::operator()(int64_t row, int64_t col, int64_t level) {
  return (*this)[{row, col, level}];
}
const Matrix& RowColLevelMap::operator()(int64_t row, int64_t col, int64_t level) const {
  return (*this)[{row, col, level}];
}

void RowColLevelMap::insert(int64_t row, int64_t col, int64_t level, Matrix&& matrix) {
  map.insert({{row, col, level}, std::move(matrix)});
}
void RowColLevelMap::insert(const std::tuple<int64_t, int64_t, int64_t>& key, Matrix&& matrix) {
  map.insert({key, std::move(matrix)});
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
