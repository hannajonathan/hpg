# Copyright 2021 Associated Universities, Inc. Washington DC, USA.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
include(GNUInstallDirs)

option(Hpg_ENABLE_ALL "Enable all devices enabled by Kokkos installation" ON)
option(Hpg_ENABLE_SERIAL "Enable Serial device" ${Hpg_ENABLE_ALL})
option(Hpg_ENABLE_OPENMP "Enable OpenMP device" ${Hpg_ENABLE_ALL})
option(Hpg_ENABLE_CUDA "Enable CUDA device" ${Hpg_ENABLE_ALL})
option(Hpg_ENABLE_MPI "Enable MPI runtime" OFF)
set(Hpg_API 11 CACHE STRING "HPG API C++ standard")
option(Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  "Enable experimental kernel versions" OFF)
mark_as_advanced(Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS)
set(Hpg_MAX_NUM_CF_GROUPS
  1000 CACHE STRING "Maximum number of CF groups")

# validate value of Hpg_API
#
set(HPG_API_OPTIONS 11 17)
set_property(CACHE Hpg_API PROPERTY STRINGS HPG_API_OPTIONS)
list(FIND HPG_API_OPTIONS ${Hpg_API} VALID_HPG_API)
if(VALID_HPG_API LESS 0)
  message(FATAL_ERROR
    "Invalid Hpg_API; supported options are ${HPG_API_OPTIONS}")
endif()

set(HAVE_SERIAL -1)
set(HAVE_OPENMP -1)
set(HAVE_CUDA -1)

# find Kokkos
#
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
if(num_enabled_devices EQUAL 0)
  message(FATAL_ERROR "No enabled devices")
endif()
# TODO: eventually, we'll be able to build without an enabled host device, but
# until then...
if((HAVE_SERIAL EQUAL -1) AND (HAVE_OPENMP EQUAL -1))
  message(FATAL_ERROR "No host devices enabled")
endif()
kokkos_check(DEVICES ${ENABLED_DEVICES})

# find MPI
#
if(Hpg_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  if(${MPI_CXX_VERSION_MAJOR} VERSION_LESS 3)
    message(FATAL_ERROR "HPG requires MPI 3 or later")
  endif()
endif()

# find FFTW
#
if(HAVE_SERIAL GREATER -1 OR HAVE_OPENMP GREATER -1)
  # find FFTW
  # TODO: might be good to choose precision according to config variable
  set(FFTW_COMPONENTS)
  if(HAVE_SERIAL GREATER -1)
    list(APPEND FFTW_COMPONENTS Double)
  endif()
  if(HAVE_OPENMP GREATER -1)
    list(APPEND FFTW_COMPONENTS DoubleOpenMP)
  endif()
  find_package(FFTW REQUIRED COMPONENTS ${FFTW_COMPONENTS})
endif()

# libhpg_version
#
configure_file(hpg_version.cpp.in hpg_version.cpp @ONLY)
add_library(hpg_version STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/hpg_version.cpp)
target_include_directories(hpg_version PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
set_target_properties(hpg_version PROPERTIES
  EXPORT_NAME version
  OUTPUT_NAME hpgversion)
add_library(hpg::version ALIAS hpg_version)

# libhpg
#
add_library(hpg_hpg hpg.cc hpg_impl.cc hpg_runtime.cc hpg_error.cc)
if(Hpg_ENABLE_MPI)
  target_sources(hpg_hpg PRIVATE hpg_mpi.cc)
endif()
set_target_properties(hpg_hpg PROPERTIES
  INTERFACE_ENABLE_SERIAL ${Hpg_ENABLE_SERIAL}
  INTERFACE_ENABLE_OPENMP ${Hpg_ENABLE_OPENMP}
  INTERFACE_ENABLE_CUDA ${Hpg_ENABLE_CUDA}
  INTERFACE_ENABLE_MPI ${Hpg_ENABLE_MPI})
set_property(TARGET hpg_hpg APPEND PROPERTY
  COMPATIBLE_INTERFACE_BOOL ENABLE_SERIAL)
set_property(TARGET hpg_hpg APPEND PROPERTY
  COMPATIBLE_INTERFACE_BOOL ENABLE_OPENMP)
set_property(TARGET hpg_hpg APPEND PROPERTY
  COMPATIBLE_INTERFACE_BOOL ENABLE_CUDA)
set_property(TARGET hpg_hpg APPEND PROPERTY
  COMPATIBLE_INTERFACE_BOOL ENABLE_MPI)
target_link_libraries(hpg_hpg
  PRIVATE hpg::version)
add_library(hpg::hpg ALIAS hpg_hpg)
if(Hpg_ENABLE_MPI)
  set(HPG_ENABLE_MPI ${Hpg_ENABLE_MPI})
  target_link_libraries(hpg_hpg PUBLIC MPI::MPI_CXX)
endif()
if(HAVE_SERIAL GREATER -1)
  set(HPG_ENABLE_SERIAL ${Hpg_ENABLE_SERIAL})
  if(HAVE_OPENMP LESS 0)
    target_link_libraries(hpg_hpg PRIVATE FFTW::Double)
  endif()
endif()
if(HAVE_OPENMP GREATER -1)
  set(HPG_ENABLE_OPENMP ${Hpg_ENABLE_OPENMP})
  target_link_libraries(hpg_hpg PRIVATE FFTW::DoubleOpenMP)
endif()
if(HAVE_CUDA GREATER -1)
  set(HPG_ENABLE_CUDA ${Hpg_ENABLE_CUDA})
  target_compile_options(hpg_hpg PRIVATE --expt-relaxed-constexpr)
  target_link_libraries(hpg_hpg PRIVATE cufft)
endif()
if(Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS)
  set(HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ${Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS})
endif()
target_compile_options(hpg_hpg PRIVATE -Wno-unknown-pragmas)
target_link_libraries(hpg_hpg PRIVATE Kokkos::kokkoscore)

# set the HPG_ENABLE_* variables in the parent scope based on
# Hpg_ENABLE_* and what we found in Kokkos
set(HPG_ENABLE_SERIAL ${HPG_ENABLE_SERIAL} PARENT_SCOPE)
set(HPG_ENABLE_OPENMP ${HPG_ENABLE_OPENMP} PARENT_SCOPE)
set(HPG_ENABLE_CUDA ${HPG_ENABLE_CUDA} PARENT_SCOPE)
set(HPG_ENABLE_MPI ${HPG_ENABLE_MPI} PARENT_SCOPE)

# use configuration file to save some variables that affect API
configure_file(hpg_config.hpp.in hpg_config.hpp)

set(hpg_PUBLIC_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_impl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_core.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_runtime.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_indexing.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_rval.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_error.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpg_version.hpp
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/hpg_config.hpp
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/hpg_export.h)
if(Hpg_ENABLE_MPI)
  list(APPEND hpg_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/hpg_mpi.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hpg_mpi_runtime.hpp)
endif()
set_target_properties(hpg_hpg PROPERTIES
  VERSION ${hpg_VERSION}
  SOVERSION ${hpg_VERSION_MAJOR}
  EXPORT_NAME hpg
  OUTPUT_NAME hpg
  PUBLIC_HEADER "${hpg_PUBLIC_HEADERS}")
target_include_directories(hpg_hpg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GenerateExportHeader)
generate_export_header(hpg_hpg BASE_NAME hpg)

install(TARGETS hpg_hpg hpg_version
  EXPORT hpgTargets
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hpg")

# minigridder
#
if(HPG_ENABLE_OPENMP)
  add_executable(minigridder minigridder.cc)
  if(HPG_ENABLE_CUDA)
    target_compile_options(minigridder PRIVATE --expt-relaxed-constexpr)
  endif()
  target_link_libraries(minigridder hpg::hpg Kokkos::kokkoscore)

  install(TARGETS minigridder
    EXPORT hpgTargets)
endif()
