#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {

template <class T>
T& RowMap<T>::operator[](int64_t key) { return map.at(key); }

template <class T>
const T& RowMap<T>::operator[](int64_t key) const { return map.at(key); }

template <class T>
T& RowMap<T>::operator()(int64_t key) { return (*this)[key]; }
template <class T>
const T& RowMap<T>::operator()(int64_t key) const { return (*this)[key]; }

template <class T>
void RowMap<T>::insert(int64_t key, T&& matrix) {
  if ((*this).exists(key)) {
    std::cout << "RowMap<T>::insert() -> Element at <" << key << "> exists and cannot be inserted." << std::endl;
    abort();
  }
  map.insert({key, std::forward<T>(matrix)});
}

template <class T>
T RowMap<T>::extract(int64_t key) {
  T out = std::move(map[key]);
  map.erase(key);
  return out;
}

template <class T>
bool RowMap<T>::exists(int64_t key) const {
  return map.count(key) == 0 ? false : true;
}

template <class T>
void RowMap<T>::erase(int64_t key) {
  map.erase({key});
}

template<class T>
void RowMap<T>::erase_all() {
  map.erase(map.begin(), map.end());
}


template class RowMap<Matrix>;
template class RowMap<std::vector<int64_t>>;
template class RowMap<std::set<int64_t>>;

template<class T>
void RowColMap<T>::deep_copy(const RowColMap<T>& A) {
  for (const auto& e : A.map) {
    T copy(e.second);
    this->insert(e.first, std::move(copy));
  }
}

template<>
void RowColMap<Hatrix::Matrix>::deep_copy(const RowColMap<Hatrix::Matrix>& A) {
  for (const auto& e : A.map) {
    Hatrix::Matrix copy(e.second, true);
    this->insert(e.first, std::move(copy));
  }
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
  (*this).insert({row, col}, std::forward<T>(value));
}
template<class T>
void RowColMap<T>::insert(const std::tuple<int64_t, int64_t>& key,
                          T&& value) {
  if ((*this).exists(key)) {
    std::cout << "Element at <" << std::get<0>(key) << "," << std::get<1>(key)
              << "> exists and cannot be inserted." << std::endl;
    abort();
  }
  map.insert({key, std::forward<T>(value)});
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
bool RowColMap<T>::exists(const std::tuple<int64_t, int64_t>& key) const {
  return map.count(key) == 0 ? false : true;
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
template class RowColMap<Hatrix::Matrix>;
template class RowColMap<int64_t>;
template class RowColMap<int>;
template class RowColMap<std::vector<int64_t>>;
template class RowColMap<std::vector<int>>;

template<class T>
void RowColLevelMap<T>::deep_copy(const RowColLevelMap<T>& A) {
  for (const auto& e : A.map) {
    T copy(e.second);
    this->insert(e.first, std::move(copy));
  }
}

template<>
void RowColLevelMap<Hatrix::Matrix>::deep_copy(const RowColLevelMap<Hatrix::Matrix>& A) {
  for (const auto& e : A.map) {
    Hatrix::Matrix copy(e.second, true);
    this->insert(e.first, std::move(copy));
  }
}

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
  map.insert({{row, col, level}, std::forward<T>(matrix)});
}
template <class T>
void RowColLevelMap<T>::insert(const std::tuple<int64_t, int64_t, int64_t>& key, T&& matrix) {
  map.insert({key, std::forward<T>(matrix)});
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
void RowColLevelMap<T>::erase(int64_t row, int64_t col, int64_t level) {
  map.erase({row, col, level});
}

template <class T>
void RowColLevelMap<T>::erase_all() {
  map.erase(map.begin(), map.end());
}

template <class T>
bool RowColLevelMap<T>::exists(int64_t row, int64_t col, int64_t level) const {
  return map.count({row, col, level}) == 0 ? false : true;
}

// explicit instatiation
template class RowColLevelMap<bool>;
template class RowColLevelMap<int>;
template class RowColLevelMap<Matrix>;


}  // namespace Hatrix
