option(Hpg_ENABLE_ALL "Enable all devices enabled by Kokkos installation" ON)
option(Hpg_ENABLE_SERIAL "Enable Serial device" ${Hpg_ENABLE_ALL})
option(Hpg_ENABLE_OPENMP "Enable OpenMP device" ${Hpg_ENABLE_ALL})
option(Hpg_ENABLE_CUDA "Enable CUDA device" ${Hpg_ENABLE_ALL})
set(Hpg_API 11 CACHE STRING "HPG API C++ standard")

# validate value of Hpg_API
#
set(HPG_API_OPTIONS 11 17)
set_property(CACHE Hpg_API PROPERTY STRINGS HPG_API_OPTIONS)
list(FIND HPG_API_OPTIONS ${Hpg_API} VALID_HPG_API)
if(VALID_HPG_API LESS 0)
  message(FATAL_ERROR "Invalid Hpg_API; supported options are ${HPG_API_OPTIONS}")
endif()

# find Kokkos
#
set(HAVE_SERIAL -1)
set(HAVE_OPENMP -1)
set(HAVE_CUDA -1)
find_package(Kokkos
  REQUIRED COMPONENTS kokkos kokkoscore
  OPTIONAL_COMPONENTS CUDA)
if(Hpg_ENABLE_SERIAL)
  list(FIND Kokkos_DEVICES SERIAL HAVE_SERIAL)
  if(HAVE_SERIAL GREATER -1)
    list(APPEND ENABLED_DEVICES SERIAL)
  elseif(NOT Hpg_ENABLE_ALL)
    message(FATAL_ERROR "Serial device not enabled in Kokkos")
  endif()
endif()
if(Hpg_ENABLE_OPENMP)
  list(FIND Kokkos_DEVICES OPENMP HAVE_OPENMP)
  if(HAVE_OPENMP GREATER -1)
    list(APPEND ENABLED_DEVICES OPENMP)
  elseif(NOT Hpg_ENABLE_ALL)
    message(FATAL_ERROR "OpenMP device not enabled in Kokkos")
  endif()
endif()
if(Hpg_ENABLE_CUDA)
  list(FIND Kokkos_DEVICES CUDA HAVE_CUDA)
  if(HAVE_CUDA GREATER -1)
    list(APPEND ENABLED_DEVICES CUDA)
  elseif(NOT Hpg_ENABLE_ALL)
    message(FATAL_ERROR "CUDA device not enabled in Kokkos")
  endif()
endif()
list(LENGTH ENABLED_DEVICES num_enabled_devices)
if(${num_enabled_devices} EQUAL 0)
  message(FATAL_ERROR "No enabled devices")
endif()
kokkos_check(DEVICES ${ENABLED_DEVICES})

# find FFTW
#
if(HAVE_SERIAL GREATER -1 OR  HAVE_OPENMP GREATER -1)
  # find FFTW
  # TODO: might be good to choose precision according to config variable
  set(FFTW_COMPONENTS)
  if(HAVE_SERIAL GREATER -1)
    list(APPEND FFTW_COMPONENTS DOUBLE_LIB)
  endif()
  if(HAVE_OPENMP GREATER -1)
    list(APPEND FFTW_COMPONENTS DOUBLE_OPENMP_LIB)
  endif()
  find_package(FFTW REQUIRED COMPONENTS ${FFTW_COMPONENTS})
endif()

# libhpg
#
add_library(hpg hpg.cc)
if(HAVE_SERIAL GREATER -1)
  set(HPG_ENABLE_SERIAL ${Hpg_ENABLE_SERIAL})
  if(HAVE_OPENMP LESS 0)
    target_link_libraries(hpg PRIVATE FFTW::Double)
  endif()
endif()
if(HAVE_OPENMP GREATER -1)
  set(HPG_ENABLE_OPENMP ${Hpg_ENABLE_OPENMP})
  target_link_libraries(hpg PRIVATE FFTW::DoubleOpenMP)
endif()
if(HAVE_CUDA GREATER -1)
  set(HPG_ENABLE_CUDA ${Hpg_ENABLE_CUDA})
  target_compile_options(hpg PRIVATE --expt-relaxed-constexpr)
  target_link_libraries(hpg PRIVATE cufft)
endif()

# use configuration file to save some variables that affect API
configure_file(hpg_config.hpp.in hpg_config.hpp)

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
