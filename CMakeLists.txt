cmake_minimum_required(VERSION 3.9)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(policies)
include(host_specific_settings)
set_policies()
host_specific_settings()

project(Hatrix
  VERSION 1.0
  LANGUAGES CXX
)

# Use Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(Hatrix_BUILD_TESTS "Build tests of the library." ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
  add_compile_options(-g -O3)
endif()

add_library(Hatrix
  src/util/matrix_generators.cpp
  src/util/greens_functions.cpp
  src/util/helpers.cpp
  src/util/profiling.cpp
  src/util/timer.cpp
)
target_compile_features(Hatrix PUBLIC cxx_std_17)

target_include_directories(Hatrix
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
  ${BLAS_INCLUDE_DIR}
  ${LAPACKE_INCLUDE_DIR}
)

option(USE_PAPI "Use PAPI to get CPU performance counters" OFF)
if(USE_PAPI)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PAPI REQUIRED papi)
  target_compile_definitions(Hatrix PUBLIC HATRIX_ENABLE_PAPI)
  target_compile_options(Hatrix PUBLIC -g)
  target_include_directories(Hatrix PRIVATE ${PAPI_INCLUDE_DIRS})
  target_link_libraries(Hatrix ${PAPI_LINK_LIBRARIES})
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Warning: PAPI may not work with Release build type that uses -DNDEBUG macro")
    message("Please use \"Profile\" or \"Debug\" build type instead")
  endif()
endif()

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_sources(Hatrix PRIVATE
  src/classes/IndexedMap.cpp
  src/classes/Matrix.cpp
  src/classes/Particle.cpp
  src/classes/Cell.cpp
  src/classes/Domain.cpp
  src/classes/SymmetricSharedBasisMatrix.cpp
  src/functions/arithmetics.cpp
  src/functions/blas.cpp
  src/functions/lapack.cpp
)
target_link_libraries(Hatrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

if (LAPACKE_FOUND)
  message("Linking LAPACKE: ${LAPACKE_LIBRARY}")
  target_link_libraries(Hatrix ${LAPACKE_LIBRARY})
endif()

option(USE_OPENMP "Multithreading with OpenMP" ON)
if(USE_OPENMP)
  if (APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
  endif()

  find_package(OpenMP REQUIRED)
  target_link_libraries(Hatrix OpenMP::OpenMP_CXX)
endif()

if (APPLE)
find_package(GSL REQUIRED)    # See below (2)
target_link_libraries(Hatrix GSL::gsl)
endif()

include(find_or_download)
# JSON writer
option(USE_JSON "Enable matrix dump to JSON file" OFF)
if(USE_JSON)
  find_or_download(nlohmann_json)
  add_definitions(-DUSE_JSON)
endif()

add_subdirectory(examples)

if(${Hatrix_BUILD_TESTS})
  enable_testing()
  find_or_download(GTest)
  add_subdirectory(test)
endif()
