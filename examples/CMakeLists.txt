if(USE_CUDA)
  list(APPEND HATRIX_EXAMPLES
    NxN_BlockDense_LU
  )
else()
  list(APPEND HATRIX_EXAMPLES
    2x2_BLR_weak_GEMM
    2x2_BLR_LU
    2x2_BLR_weak_QR
    2x2_BlockDense_LU
    2x2_BlockDense_QR
    NxN_BLR_weak_GEMM
    NxN_BLR_strong_GEMM
    NxN_BlockDense_LU
    NxN_BLR_LU
    NxN_BLR_weak_QR
    NxN_BLR_strong_LU
    HSS_2level_construct
    HSS_Nlevel_construct
    HSS_Nlevel_construct_chandrasekaran
    HSS_Nlevel_construct_miro
    GEMM_weak_2level
    GEMM_weak_Nlevel
    UMV_weak_2level
    UMV_weak_1level
    UMV_weak_Nlevel
    BLR2_construct
    UMV_strong_1level
    H2_matrix_2level_construct
    LDL_weak_1level
    BLR2_strong_far_dense_construct
    LDL_weak_Nlevel
    H2_strong_N_level
    UMV_strong_1level_debug
    UMV_strong_chained_product
    UMV_strong_H2_3level_hss_chain_product
    UMV_strong_H2_3level_chain_product
    UMV_strong_H2_Nlevel
    UMV_strong_1level_better_index
    UMV_BLR2_far_dense
    H2_far_dense_construct
    UMV_H2_far_dense
    LDL_strong_1level
    LDL_strong_Nlevel
    H2_fixed_acc_construct_svd
    H2_fixed_acc_construct_qr
  )
endif()

foreach(EXAMPLE ${HATRIX_EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} Hatrix)
endforeach()
