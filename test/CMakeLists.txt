include(GoogleTest)

if(USE_CUDA)
  list(APPEND GTEST_TESTS
    Matrix
    arithmetics
    matmul
    lu
  )
else()
  list(APPEND GTEST_TESTS
    Matrix
    arithmetics
    matmul
    triangular_matmul
    lu
    qr
    svd
    norms
    scale
  )
endif()

foreach(TEST ${GTEST_TESTS})
  add_executable(${TEST}_test ${TEST}.cpp)
  target_link_libraries(${TEST}_test Hatrix GTest::gtest_main)
  gtest_add_tests(TARGET ${TEST}_test)
endforeach()
