include(GoogleTest)

list(APPEND GTEST_TESTS
  Matrix
  arithmetics
  matmul
  triangular_matmul
  lu
  qr
  block_dense_lu
  svd
  matrix_generators
  norms
  errors
  scale
)

foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}.cpp)
  target_link_libraries(${TEST}_test Hatrix GTest::gtest_main)
  gtest_add_tests(TARGET ${TEST}_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()
