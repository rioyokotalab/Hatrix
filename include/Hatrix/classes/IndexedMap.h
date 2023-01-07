#pragma once
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <iostream>

#include "Hatrix/classes/Matrix.h"

namespace std {

// TODO: why use these values for the hashing? Write comment.
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

template <class T>
class RowMap {
 private:
  std::unordered_map<int64_t, T> map;

public:
  T& operator[](int64_t key);
  const T& operator[](int64_t key) const;

  T& operator()(int64_t key);
  const T& operator()(int64_t key) const;

  void insert(int64_t key, T&& matrix);

  T extract(int64_t key);

  bool exists(int64_t key) const;

  void erase(int64_t key);
};
template<typename T> using ColMap   = RowMap<T>;
template<typename T> using LevelMap = RowMap<T>;
using RankMap = RowMap<std::vector<int64_t>>;

template <class T>
class RowColMap {
 private:
  std::unordered_map<std::tuple<int64_t, int64_t>, T> map;

 public:
  T& operator()(int64_t row, int64_t col);
  const T& operator()(int64_t row, int64_t col) const;

  T& operator[](const std::tuple<int64_t, int64_t>& key);
  const T& operator[](const std::tuple<int64_t, int64_t>& key) const;

  void insert(int64_t row, int64_t col, T&& value);
  void insert(const std::tuple<int64_t, int64_t>& key, T&& value);

  T extract(int64_t row, int64_t col);
  T extract(const std::tuple<int64_t, int64_t>& key);

  // Check if given <row,col> tuple exists.
  bool exists(int64_t row, int64_t col) const;

  // Check if given <row, col> typle exists in this map.
  bool exists(const std::tuple<int64_t, int64_t>& key) const;

  // Erase given (row, col) from the map.
  void erase(int64_t row, int64_t col);

  // Destructively clear all keys in the map.
  void erase_all();
};

// RowLevel and ColLevel also use a <int, int> tuple which is same as RowCol
using RowLevelMap = RowColMap<Matrix>;
using ColLevelMap = RowColMap<Matrix>;

template <class T>
class RowColLevelMap {
private:
  std::unordered_map<std::tuple<int64_t, int64_t, int64_t>, T> map;

public:
  T& operator()(int64_t row, int64_t col, int64_t level);
  const T& operator()(int64_t row, int64_t col, int64_t level) const;

  T& operator[](const std::tuple<int64_t, int64_t, int64_t>& key);
  const T& operator[](const std::tuple<int64_t, int64_t, int64_t>& key) const;

  void insert(int64_t row, int64_t col, int64_t level, T&& matrix);
  void insert(const std::tuple<int64_t, int64_t, int64_t>& key, T&& matrix);

  T extract(int64_t row, int64_t col, int64_t level);
  T extract(const std::tuple<int64_t, int64_t, int64_t>& key);

  bool exists(int64_t row, int64_t col, int64_t level) const;

  void erase(int64_t row, int64_t col, int64_t level);

  // Destructively clear all keys in the map.
  void erase_all();
};

}  // namespace Hatrix
