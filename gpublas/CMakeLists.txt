
cmake_minimum_required (VERSION 3.9)

project ("gpu" LANGUAGES CXX CUDA)

add_library(gpu STATIC handle.cu handle.h funcs.cpp funcs.h)
target_include_directories(gpu PRIVATE "./")
target_link_libraries(gpu PRIVATE cublas cusolver curand)
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 60 70 75)

add_executable(gpu_test test.cpp)
target_include_directories(gpu_test PRIVATE "./")
target_link_libraries(gpu_test PRIVATE gpu)
set_property(TARGET gpu_test PROPERTY CUDA_ARCHITECTURES 60 70 75)