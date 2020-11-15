option(Hpg_ENABLE_SERIAL "Enable Serial device" OFF)
option(Hpg_ENABLE_OPENMP "Enable OpenMP device" OFF)
option(Hpg_ENABLE_CUDA "Enable CUDA device" OFF)

# find FFTW
#
if(Hpg_ENABLE_SERIAL OR Hpg_ENABLE_OPENMP)
  # find FFTW
  # TODO: might be good to choose precision according to config variable
  set(FFTW_COMPONENTS)
  if(Hpg_ENABLE_SERIAL)
    list(APPEND FFTW_COMPONENTS DOUBLE_LIB)
  endif()
  if(Hpg_ENABLE_OPENMP)
    list(APPEND FFTW_COMPONENTS DOUBLE_OPENMP_LIB)
  endif()
  find_package(FFTW REQUIRED COMPONENTS ${FFTW_COMPONENTS})
endif()

# find Kokkos
#
find_package(Kokkos
  REQUIRED COMPONENTS kokkos kokkoscore
  OPTIONAL_COMPONENTS CUDA)
if(Hpg_ENABLE_SERIAL)
  list(FIND Kokkos_DEVICES SERIAL index)
  if(index GREATER -1)
    list(APPEND enabled_devices SERIAL)
  else()
    message(FATAL_ERROR "Serial device not enabled in Kokkos")
  endif()
endif()
if(Hpg_ENABLE_OPENMP)
  list(FIND Kokkos_DEVICES OPENMP index)
  if(index GREATER -1)
    list(APPEND enabled_devices OPENMP)
  else()
    message(FATAL_ERROR "OpenMP device not enabled in Kokkos")
  endif()
endif()
if(Hpg_ENABLE_CUDA)
  list(FIND Kokkos_DEVICES CUDA index)
  if(index GREATER -1)
    list(APPEND enabled_devices CUDA)
  else()
    message(FATAL_ERROR "CUDA device not enabled in Kokkos")
  endif()
endif()
list(LENGTH enabled_devices num_enabled_devices)
if(${num_enabled_devices} EQUAL 0)
  message(FATAL_ERROR "No enabled devices")
endif()
kokkos_check(DEVICES ${enabled_devices})

# libhpg
#
add_library(hpg hpg.cc)
if(Hpg_ENABLE_SERIAL)
  target_compile_definitions(hpg PUBLIC HPG_ENABLE_SERIAL)
  if(NOT Hpg_ENABLE_OPENMP)
    target_link_libraries(hpg PRIVATE FFTW::Double)
  endif()
endif()
if(Hpg_ENABLE_OPENMP)
  target_compile_definitions(hpg PUBLIC HPG_ENABLE_OPENMP)
  target_link_libraries(hpg PRIVATE FFTW::DoubleOpenMP)
endif()
if(Hpg_ENABLE_CUDA)
  target_compile_definitions(hpg PUBLIC HPG_ENABLE_CUDA)
  target_compile_options(hpg PRIVATE --expt-relaxed-constexpr)
  target_link_libraries(hpg PRIVATE cufft)
endif()
target_link_libraries(hpg PRIVATE Kokkos::kokkoscore)
target_include_directories(hpg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GenerateExportHeader)
generate_export_header(hpg)

install(TARGETS hpg EXPORT HPGConfig)
install(FILES ${HPG_INCLUDE}
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/hpg)
install(EXPORT HPGConfig DESTINATION share/hpg/cmake)
export(TARGETS hpg FILE HPGConfig.cmake)
