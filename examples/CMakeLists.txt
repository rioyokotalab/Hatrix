if(USE_CUDA)
  list(APPEND HATRIX_EXAMPLES
    NxN_BlockDense_LU
  )
else()
  list(APPEND HATRIX_EXAMPLES
    2x2_BLR_LU
    2x2_BlockDense_LU
    2x2_BlockDense_QR
    NxN_BlockDense_LU
    NxN_BLR_LU
    NxN_BLR_strong_LU
  )
endif()

foreach(EXAMPLE ${HATRIX_EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} Hatrix)
endforeach()
