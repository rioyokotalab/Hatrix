
cmake_minimum_required (VERSION 3.9)

project ("nbd" LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(BLAS)
find_package(LAPACK)

add_library(nbd STATIC 
  nbd.h
  kernel.cpp kernel.h
  build_tree.cpp build_tree.h 
  aca.cpp aca.h
  h2mv.cpp h2mv.h
  jacobi.cpp jacobi.h
  test_util.cpp test_util.h
  timer.cpp timer.h
)
target_include_directories(nbd PRIVATE "./" )
target_link_libraries(nbd blas lapacke)

add_executable(symmh2_example
  symmh2_example.cpp
)
target_link_libraries(symmh2_example nbd)
target_include_directories(symmh2_example PRIVATE "./" )

add_executable(h2_example
  h2_example.cpp
)
target_link_libraries(h2_example nbd)
target_include_directories(h2_example PRIVATE "./" )

add_executable(lra_example
  lra_example.cpp
)
target_link_libraries(lra_example nbd)
target_include_directories(lra_example PRIVATE "./" )

add_executable(irs_example
  irs_example.cpp
)
target_link_libraries(irs_example nbd)
target_include_directories(irs_example PRIVATE "./" )
