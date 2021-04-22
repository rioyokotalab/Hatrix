cmake_minimum_required(VERSION 3.9)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(policies)
set_policies()

project(NewStructure
  LANGUAGES CXX
  VERSION 1.0
)

option(Hatrix_BUILD_TESTS "Build tests of the library." ON)
option(Hatrix_CPU_ONLY "Build Hatrix without GPU usage." OFF)

if(Hatrix_CPU_ONLY)
  message(STATUS "Will not compile code for GPU.")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDA)
    message(STATUS "CUDA ${CUDA_VERSION} detected - compiling GPU code.")
    set(USE_CUDA ON)
  else()
    message(STATUS "CUDA not detected - building for CPU only.")
  endif()
endif()

if(USE_CUDA)
  add_executable(CUDA_hello_world src/hello_world.cu)
  set_target_properties(CUDA_hello_world PROPERTIES CUDA_ARCHITECTURES 60 70 75)
  target_link_libraries(CUDA_hello_world ${CUDA_CUBLAS_LIBRARIES})
endif()

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(Hatrix
  src/classes/Matrix.cpp
  src/functions/blas.cpp
  src/functions/lapack.cpp
  src/util/matrix_generators.cpp
  src/util/norms.cpp
)
target_compile_features(Hatrix PUBLIC cxx_std_17)

target_link_libraries(Hatrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

target_include_directories(Hatrix
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $ENV{MKLROOT}/include
)

add_subdirectory(examples)

if (${Hatrix_BUILD_TESTS})
  enable_testing()
  include(find_or_download)
  find_or_download(GTest)
  add_subdirectory(test)
endif()
