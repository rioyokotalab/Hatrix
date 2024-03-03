add_subdirectory(common)
include_directories(common)

list(APPEND HATRIX_EXAMPLES
  BLR2_weak_CON
  BLR2_strong_CON
  H2_weak_CON_2lev
  H2_weak_CON
  H2_strong_CON
  # H2_strong_CON_sym_ID
  Dense_LU_2x2
  # Dense_QR_2x2
  # Dense_EV
  Dense_LU_NxN
  BLR2_weak_LU_2x2_noumv
  BLR2_weak_LU_noumv
  BLR2_strong_LU_noumv
  # BLR2_weak_LDL
  # BLR2_strong_LDL
  # BLR2_strong_LU
  BLR2_weak_LU
  # H2_weak_LDL
  H2_weak_LU
  H2_weak_LU_2lev
  # H2_strong_LDL
  # H2_strong_LDL_nodep
  # H2_strong_PO_nodep
  # H2_strong_PO
  # H2_strong_LU
  # BLR2_weak_MM_2x2
  # BLR2_strong_MM
  # BLR2_weak_MM
  # H2_weak_MM_2lev
  # H2_weak_MM
  # H2_strong_EV_sym
  # H2_strong_EV_sym_nodep
)

foreach(EXAMPLE ${HATRIX_EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} Hatrix)
  if(USE_JSON)
    target_link_libraries(${EXAMPLE} nlohmann_json::nlohmann_json)
  endif()
endforeach()

# find_package(MPI)
# if(MPI_FOUND)
#   add_executable(H2_strong_PO_nodep_mpi H2_strong_PO_nodep_mpi.cpp)
#   target_link_libraries(H2_strong_PO_nodep_mpi MPI::MPI_CXX Hatrix)

#   add_executable(H2_strong_EV_sym_mpi_ref H2_strong_EV_sym_mpi_ref.cpp)
#   target_link_libraries(H2_strong_EV_sym_mpi_ref Hatrix MPI::MPI_CXX)

#   add_executable(H2_strong_EV_sym_mpi H2_strong_EV_sym_mpi.cpp)
#   target_link_libraries(H2_strong_EV_sym_mpi Hatrix MPI::MPI_CXX)

#   # Requires ScaLAPACK
#   add_executable(Dense_EV_sym_mpi Dense_EV_sym_mpi.cpp)
#   target_link_libraries(Dense_EV_sym_mpi Hatrix MPI::MPI_CXX ${SCALAPACK_LIBS})
# endif()
