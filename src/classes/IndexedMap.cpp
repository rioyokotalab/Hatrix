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

template<class T>
bool RowColMap<T>::exists(int64_t row, int64_t col) const {
  return map.count({row, col}) == 0 ? false : true;
}

template<class T>
void RowColMap<T>::erase(int64_t row, int64_t col) {
  map.erase({row, col});
}

template<class T>
void RowColMap<T>::erase_all() {
  map.erase(map.begin(), map.end());
}

// explicit instatiation
template class RowColMap<bool>;
template class RowColMap<Matrix>;

template<class T>
T& RowColLevelMap<T>::operator[](const std::tuple<int64_t, int64_t, int64_t>& key) {
  return map.at(key);
}
template<class T>
const T& RowColLevelMap<T>::operator[](const std::tuple<int64_t, int64_t, int64_t>& key) const {
  return map.at(key);
}

template <class T>
T& RowColLevelMap<T>::operator()(int64_t row, int64_t col, int64_t level) {
  return (*this)[{row, col, level}];
}
template <class T>
const T& RowColLevelMap<T>::operator()(int64_t row, int64_t col, int64_t level) const {
  return (*this)[{row, col, level}];
}

template <class T>
void RowColLevelMap<T>::insert(int64_t row, int64_t col, int64_t level, T&& matrix) {
  map.insert({{row, col, level}, std::move(matrix)});
}
template <class T>
void RowColLevelMap<T>::insert(const std::tuple<int64_t, int64_t, int64_t>& key, T&& matrix) {
  map.insert({key, std::move(matrix)});
}

template <class T>
T RowColLevelMap<T>::extract(int64_t row, int64_t col, int64_t level) {
  T out = std::move(map[{row, col, level}]);
  map.erase({row, col, level});
  return out;
}

template <class T>
T RowColLevelMap<T>::extract(const std::tuple<int64_t, int64_t, int64_t>& key) {
  T out = std::move(map[key]);
  map.erase(key);
  return out;
}

template <class T>
bool RowColLevelMap<T>::exists(int64_t row, int64_t col, int64_t level) const {
  return map.count({row, col, level}) == 0 ? false : true;
}

// explicit instatiation
template class RowColLevelMap<bool>;
template class RowColLevelMap<Matrix>;


}  // namespace Hatrix
