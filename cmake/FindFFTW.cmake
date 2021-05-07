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
#[=======================================================================[.rst:
FindFFTW
-------

Finds the FFTW3 library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``FFTW::Float``
Serial, single precision library

``FFTW::FloatThreads``
Threaded, single precision library

``FFTW::FloatOpenMP``
OpenMP, single precision library

``FFTW::FloatMPI``
MPI, single precision library

``FFTW::Double``
Serial, double precision library

``FFTW::DoubleThreads``
Threaded, double precision library

``FFTW::DoubleOpenMP``
OpenMP, double precision library

``FFTW::DoubleMPI``
MPI, double precision library

``FFTW::LongDouble``
Serial, long-double precision library

``FFTW::LongDoubleThreads``
Threaded, long-double precision library

``FFTW::LongDoubleOpenMP``
OpenMP, long-double precision library

``FFTW::LongDoubleMPI``
MPI, long-double precision library

``FFTW::Quad``
Serial, quad precision library

``FFTW::QuadThreads``
Threaded, quad precision library

``FFTW::QuadOpenMP``
OpenMP, quad precision library

``FFTW::QuadMPI``
MPI, quad precision library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``FFTW_FOUND``
True if the system has the FFTW3 library.
``FFTW_VERSION``
The version of the FFTW3 library which was found.
``FFTW_INCLUDE_DIRS``
Include directories needed to use FFTW3.
``FFTW_LIBRARIES``
Libraries needed to link to FFTW3.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``FFTW_ROOT_DIR``
Root directory of FFTW3 installation.

``FFTW_INCLUDE_DIR``
The directory containing ``fftw3.h``.

``FFTW_Float_LIBRARY``
The path to the fftw3f library.
``FFTW_FloatThreads_LIBRARY``
The path to the fftw3f_threads library.
``FFTW_FloatOpenMP_LIBRARY``
The path to the fftw3f_omp library.
``FFTW_FloatMPI_LIBRARY``
The path to the fftw3f_mpi library.

``FFTW_Double_LIBRARY``
The path to the fftw3 library.
``FFTW_DoubleThreads_LIBRARY``
The path to the fftw3_threads library.
``FFTW_DoubleOpenMP_LIBRARY``
The path to the fftw3_omp library.
``FFTW_DoubleMPI_LIBRARY``
The path to the fftw3_mpi library.

``FFTW_LongDouble_LIBRARY``
The path to the fftw3l library.
``FFTW_LongDoubleThreads_LIBRARY``
The path to the fftw3l_threads library.
``FFTW_LongDoubleOpenMP_LIBRARY``
The path to the fftw3l_omp library.
``FFTW_LongDoubleMPI_LIBRARY``
The path to the fftw3l_mpi library.

``FFTW_Quad_LIBRARY``
The path to the fftw3q library.
``FFTW_QuadThreads_LIBRARY``
The path to the fftw3q_threads library.
``FFTW_QuadOpenMP_LIBRARY``
The path to the fftw3q_omp library.
``FFTW_QuadMPI_LIBRARY``
The path to the fftw3q_mpi library.

#]=======================================================================]

include(CMakeFindDependencyMacro)
include(FindPackageHandleStandardArgs)

set(${CMAKE_FIND_PACKAGE_NAME}_comps
  ${${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS})

set(FFTW_ROOT_DIR "" CACHE PATH "fftw3 root directory")

# determine root path from pkg-config
find_dependency(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT FFTW_ROOT_DIR)
  pkg_check_modules(PKG_FFTW QUIET fftw3)
  if(PKG_FFTW_FOUND)
    set(FFTW_VERSION ${PKG_FFTW_VERSION})
  endif()
endif()

# find a specific component library at FFTW_ROOT_DIR
macro(FFTW_find_library_at_root _comp)
  list(FIND ${CMAKE_FIND_PACKAGE_NAME}_comps ${_comp} _idx)
  if(NOT ${_idx} EQUAL -1)
    set(_names ${ARGN})
    find_library(
      FFTW_${_comp}_LIBRARY
      NAMES ${_names}
      PATHS ${FFTW_ROOT_DIR}
      PATH_SUFFIXES lib lib64
      NO_DEFAULT_PATH)
    if(NOT FFTW_wisdom AND FFTW_${_comp}_LIBRARY)
      list(FILTER _names INCLUDE REGEX "^fftw3.*")
      list(GET _names 0 _prefix)
      string(REGEX REPLACE
        "(fftw3)([flq]?)(.*)" "fftw\\2-wisdom" _fftw_wisdom ${_prefix})
      find_program(FFTW_wisdom ${_fftw_wisdom} "${FFTW_ROOT_DIR}/bin")
    endif()
  endif()
endmacro()

# find a specific component library at PKG_FFTW_LIBRARY_DIRS
macro(FFTW_find_library_at_pkg _comp)
  list(FIND ${CMAKE_FIND_PACKAGE_NAME}_comps ${_comp} _idx)
  if(NOT ${_idx} EQUAL -1)
    find_library(
      FFTW_${_comp}_LIBRARY
      NAMES ${ARGN}
      PATHS ${PKG_FFTW_LIBRARY_DIRS}
      NO_DEFAULT_PATH)
  endif()
endmacro()

if(FFTW_ROOT_DIR)

  # find double libs
  FFTW_find_library_at_root(Double fftw3 libfftw3-3)
  FFTW_find_library_at_root(DoubleThreads fftw3_threads)
  FFTW_find_library_at_root(DoubleOpenMP fftw3_omp)
  FFTW_find_library_at_root(DoubleMPI fftw3_mpi)

  # find float libs
  FFTW_find_library_at_root(Float fftw3f libfftw3f-3)
  FFTW_find_library_at_root(FloatThreads fftw3f_threads)
  FFTW_find_library_at_root(FloatOpenMP fftw3f_omp)
  FFTW_find_library_at_root(FloatMPI fftw3f_mpi)

  # find long double libs
  FFTW_find_library_at_root(LongDouble fftw3l libfftw3l-3)
  FFTW_find_library_at_root(LongDoubleThreads fftw3l_threads)
  FFTW_find_library_at_root(LongDoubleOpenMP fftw3l_omp)
  FFTW_find_library_at_root(LongDoubleMPI fftw3l_mpi)

  # find quad libs
  FFTW_find_library_at_root(Quad fftw3q libfftw3q-3)
  FFTW_find_library_at_root(QuadThreads fftw3q_threads)
  FFTW_find_library_at_root(QuadOpenMP fftw3q_omp)
  FFTW_find_library_at_root(QuadMPI fftw3q_mpi)

  # find include dir
  find_path(
    FFTW_INCLUDE_DIR
    NAMES fftw3.h
    PATHS ${FFTW_ROOT_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

  # run wisdom command with -V flag to get the version number
  if(FFTW_wisdom)
    execute_process(
      COMMAND "${FFTW_wisdom}" -V
      RESULT_VARIABLE _wisdom_result
      OUTPUT_VARIABLE _wisdom_out)
    if(_wisdom_result)
      message(FATAL_ERROR "${FFTW_wisdom} failed: ${_wisdom_result}")
    endif()
    string(REGEX REPLACE
      "(.*FFTW version )([0-9]\.[0-9]\.[0-9])(.*)" "\\2"
      FFTW_VERSION ${_wisdom_out})
  endif()

else()

  # find double libs
  FFTW_find_library_at_pkg(Double fftw3)
  FFTW_find_library_at_pkg(DoubleThreads fftw3_threads)
  FFTW_find_library_at_pkg(DoubleOpenMP fftw3_omp)
  FFTW_find_library_at_pkg(DoubleMPI fftw3_mpi)

  # find float libs
  FFTW_find_library_at_pkg(Float fftw3f)
  FFTW_find_library_at_pkg(FloatThreads fftw3f_threads)
  FFTW_find_library_at_pkg(FloatOpenMP fftw3f_omp)
  FFTW_find_library_at_pkg(FloatMPI fftw3f_mpi)

  # find long double libs
  FFTW_find_library_at_pkg(LongDouble fftw3l)
  FFTW_find_library_at_pkg(LongDoubleThreads fftw3l_threads)
  FFTW_find_library_at_pkg(LongDoubleOpenMP fftw3l_omp)
  FFTW_find_library_at_pkg(LongDoubleMPI fftw3l_mpi)

  # find quad libs
  FFTW_find_library_at_pkg(Quad fftw3q)
  FFTW_find_library_at_pkg(QuadThreads fftw3q_threads)
  FFTW_find_library_at_pkg(QuadOpenMP fftw3q_omp)
  FFTW_find_library_at_pkg(QuadMPI fftw3q_mpi)

  # find include dir
  find_path(
    FFTW_INCLUDE_DIR
    NAMES fftw3.h
    PATHS ${PKG_FFTW_INCLUDE_DIRS})
endif()

# set path result variables
if(FFTW_INCLUDE_DIR)
  set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
endif()
foreach(_comp IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_comps)
  if(FFTW_${_comp}_LIBRARY)
    set(FFTW_${_comp}_FOUND TRUE)
    list(APPEND FFTW_LIBRARIES "${FFTW_${_comp}_LIBRARY}")
  endif()
endforeach()

# handle checking components
find_package_handle_standard_args(FFTW
  VERSION_VAR FFTW_VERSION
  HANDLE_COMPONENTS)

# create the imported targets
if(FFTW_FOUND)
  mark_as_advanced(FFTW_ROOT_DIR FFTW_INCLUDE_DIR)
  foreach(_comp IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_comps)
    if(FFTW_${_comp}_FOUND)
      mark_as_advanced(FFTW_${_comp}_LIBRARY)
      if(NOT TARGET FFTW::${_comp})
        add_library(FFTW::${_comp} UNKNOWN IMPORTED)
        set_target_properties(FFTW::${_comp} PROPERTIES
          IMPORTED_LOCATION  "${FFTW_${_comp}_LIBRARY}")
        target_include_directories(FFTW::${_comp} INTERFACE
          "${FFTW_INCLUDE_DIR}")
      endif()
    endif()
  endforeach()
endif()
