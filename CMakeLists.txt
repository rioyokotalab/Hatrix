cmake_minimum_required(VERSION 3.9)

project(NewStructure
  LANGUAGES CXX
  VERSION 1.0
)

option(Hatrix_BUILD_TESTS "Build tests of the library." ON)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(Hatrix
  src/classes/Matrix.cpp
  src/functions/blas.cpp
  src/functions/lapack.cpp
)

target_link_libraries(Hatrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_include_directories(Hatrix
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $ENV{MKLROOT}/include
)

if (${Hatrix_BUILD_TESTS})
  enable_testing()
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  include(find_or_download)
  find_or_download(GTest)
  add_subdirectory(test)
endif()
