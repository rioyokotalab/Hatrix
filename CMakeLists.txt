cmake_minimum_required(VERSION 3.9)

project(NewStructure
  LANGUAGES CXX
  VERSION 1.0
)

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

add_executable(Matrix_test test/Matrix.cpp)
target_link_libraries(Matrix_test Hatrix)

add_executable(gemm_test test/gemm.cpp)
target_link_libraries(gemm_test Hatrix)

add_executable(getrf_test test/getrf.cpp)
target_link_libraries(getrf_test Hatrix)

add_executable(qr_test test/qr.cpp)
target_link_libraries(qr_test Hatrix)
