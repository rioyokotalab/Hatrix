include(GoogleTest)

list(APPEND GTEST_TESTS Matrix matmul lu qr block_dense_lu svd)

foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}.cpp)
  target_link_libraries(${TEST}_test Hatrix GTest::gtest_main)
  gtest_add_tests(TARGET ${TEST}_test)
endforeach()
