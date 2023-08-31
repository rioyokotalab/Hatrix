include(GoogleTest)

list(APPEND GTEST_TESTS
  Matrix
  Dense
  arithmetics
  matmul
  triangular_matmul
  lu
  ldl
  qr
  rq
  svd
  norms
  scale
  id
)

foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}.cpp)
  target_link_libraries(${TEST}_test Hatrix GTest::gtest_main)
  gtest_add_tests(TARGET ${TEST}_test)
endforeach()
