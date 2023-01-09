add_subdirectory(common)
include_directories(common)

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
    LDL_strong_1level
    LDL_strong_Nlevel
    Construction_H2_Nlevel
    LDL_SymmetricH2_Nlevel
    Eigen_SymmetricH2_Nlevel
    UMV_H2_Nlevel
    Construction_SymmetricH2_Nlevel_with_sampling
    ULV_SymmetricH2_Nlevel_nodependency
    LDL_SymmetricH2_Nlevel_nodependency
    Eigen_SymmetricH2_Nlevel_nodependency
    Eigen_SymmetricH2_Nlevel_nodependency_comp_fill_in
    Construction_ELSES
    LDL_ELSES_nodependency
    Eigen_ELSES
    Eigen_ELSES_nodependency
  )
endif()

foreach(EXAMPLE ${HATRIX_EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} Hatrix nlohmann_json::nlohmann_json)
endforeach()

find_package(MPI REQUIRED)
add_executable(SymmH2_ULV_SC22 SymmH2_ULV_SC22.cpp)
target_compile_options(SymmH2_ULV_SC22 PRIVATE -DUSE_MKL -D_PROF)
target_link_libraries(SymmH2_ULV_SC22 MPI::MPI_CXX Hatrix)
