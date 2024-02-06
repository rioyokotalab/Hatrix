add_subdirectory(include)
include_directories(include)

list(APPEND HATRIX_EXAMPLES
  BLR2_weak_CON
  BLR2_strong_CON
  H2_weak_CON_2lev
  H2_weak_CON
  H2_strong_CON_2lev
  H2_strong_CON
  H2_strong_CON_sym
  H2_strong_CON_sym_ID
  Dense_LU_2x2
  Dense_QR_2x2
  Dense_EV
  Dense_LU_NxN
  BLR2_weak_LU_2x2_noumv
  BLR2_weak_LU_noumv
  BLR2_strong_LU_noumv
  BLR2_weak_LDL
  BLR2_strong_LDL
  BLR2_strong_LU
  BLR2_weak_LU
  H2_weak_LDL
  H2_weak_LU
  H2_weak_LU_2lev
  H2_strong_LDL
  H2_strong_LDL_nodep
  H2_strong_PO_nodep
  H2_strong_PO
  H2_strong_LU
  BLR2_weak_MM_2x2
  BLR2_strong_MM
  BLR2_weak_MM
  H2_weak_MM_2lev
  H2_weak_MM
  H2_strong_EV_sym
  H2_strong_EV_sym_nodep
)

foreach(EXAMPLE ${HATRIX_EXAMPLES})
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} Hatrix)
  target_link_libraries(${EXAMPLE} ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)
  if(USE_JSON)
    target_link_libraries(${EXAMPLE} nlohmann_json::nlohmann_json)
  endif()
endforeach()

find_package(MPI)
if(MPI_FOUND)
  add_executable(H2_strong_PO_nodep_mpi H2_strong_PO_nodep_mpi.cpp)
  target_compile_options(H2_strong_PO_nodep_mpi PRIVATE -DUSE_MKL -D_PROF)
  target_link_libraries(H2_strong_PO_nodep_mpi MPI::MPI_CXX Hatrix)
  target_link_libraries(H2_strong_PO_nodep_mpi ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)

  add_executable(H2_strong_EV_sym_mpi_ref H2_strong_EV_sym_mpi_ref.cpp)
  target_link_libraries(H2_strong_EV_sym_mpi_ref Hatrix MPI::MPI_CXX)
  target_link_libraries(H2_strong_EV_sym_mpi_ref ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)

  add_executable(H2_strong_EV_sym_mpi H2_strong_EV_sym_mpi.cpp)
  target_link_libraries(H2_strong_EV_sym_mpi Hatrix MPI::MPI_CXX)
  target_link_libraries(H2_strong_EV_sym_mpi ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)

  # Requires ScaLAPACK
  add_executable(Dense_EV_sym_mpi Dense_EV_sym_mpi.cpp)
  target_link_libraries(Dense_EV_sym_mpi Hatrix MPI::MPI_CXX ${SCALAPACK_LIBS})
  target_link_libraries(Dense_EV_sym_mpi ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)
endif()


# H2-Construction with ELSES wrapper
add_executable(H2_strong_CON_sym_elses H2_strong_CON_sym_elses.cpp)
target_link_libraries(H2_strong_CON_sym_elses Hatrix)
target_link_libraries(H2_strong_CON_sym_elses ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)
add_executable(H2_strong_CON_sym_ID_elses H2_strong_CON_sym_ID_elses.cpp)
target_link_libraries(H2_strong_CON_sym_ID_elses Hatrix)
target_link_libraries(H2_strong_CON_sym_ID_elses ${ELSES_ROOT}/src/src.a ${ELSES_ROOT}/xmlf90-1.2g-elses/macros/lib/libflib.a)
