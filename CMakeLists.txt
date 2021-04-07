cmake_minimum_required(VERSION 3.9)

project(NewStructure
  LANGUAGES CXX
  VERSION 1.0
)

find_package(CBLAS REQUIRED)
find_package(LAPACKE REQUIRED)

add_library(Hatrix
  src/classes/Matrix.cpp
  src/functions/blas.cpp
  src/functions/lapack.cpp
)
target_link_libraries(Hatrix cblas lapacke)

target_include_directories(Hatrix
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(gemm_test test/gemm.cpp)
target_link_libraries(gemm_test Hatrix)
