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
option(Hatrix_CPU_ONLY "Build Hatrix without GPU usage." ON)

if(Hatrix_CPU_ONLY)
  message(STATUS "Will not compile code for GPU.")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA ${CUDA_VERSION} detected - compiling GPU code.")
    set(USE_CUDA ON)
  else()
    message(STATUS "CUDA not detected - building for CPU only.")
  endif()
endif()

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
  src/util/helpers.cpp
  src/util/profiling.cpp
  src/util/timer.cpp
)
target_compile_features(Hatrix PUBLIC cxx_std_11)

target_include_directories(Hatrix
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${BLAS_INCLUDE_DIR}
)

option(USE_OPENMP "Multithreading with OpenMP" ON)
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

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

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  target_sources(Hatrix PRIVATE
    src/classes/Matrix.cu
    src/functions/arithmetics.cpp
    src/functions/blas.cu
    src/functions/lapack.cu
    src/util/context.cu
  )
  set_target_properties(Hatrix PROPERTIES CUDA_ARCHITECTURES 60 70 75)
  target_link_libraries(Hatrix cublas cusolver curand)
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_sources(Hatrix PRIVATE
    src/classes/IndexedMap.cpp
    src/classes/Matrix.cpp
    src/functions/arithmetics.cpp
    src/functions/blas.cpp
    src/functions/lapack.cpp
    src/util/context.cpp
  )
  target_link_libraries(Hatrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

find_package(GSL REQUIRED)    # See below (2)
target_link_libraries(Hatrix GSL::gsl)

include(find_or_download)
# JSON writer
find_or_download(nlohmann_json)

add_subdirectory(examples)

if(${Hatrix_BUILD_TESTS})
  enable_testing()
  find_or_download(GTest)
  add_subdirectory(test)
endif()
