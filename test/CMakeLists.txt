include(GoogleTest)

list(APPEND GTEST_TESTS Matrix arithmetics matmul triangular_matmul lu qr svd norms scale)

foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}.cpp)
  target_link_libraries(${TEST}_test Hatrix_gpu GTest::gtest_main)
  gtest_add_tests(TARGET ${TEST}_test)
endforeach()
