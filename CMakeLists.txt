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
cmake_minimum_required(VERSION 3.18)
include(hpgVersionDetails.cmake)
project(hpg
  VERSION ${hpg_VER}
  DESCRIPTION "High performance (de-)gridding kernel implementation library"
  HOMEPAGE_URL https://gitlab.nrao.edu/mpokorny/hpg)
enable_testing()
enable_language(CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)

# Set a default build type if none was specified
set(default_build_type Release)
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type Debug)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set a default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND NOT WIN32
    AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_INSTALL_PREFIX "/opt/nrao.edu/${PROJECT_NAME}" CACHE PATH "..." FORCE)
endif()

# Project wide setup
set(MIN_CXX_STANDARD 20)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${MIN_CXX_STANDARD})
endif()
if(${CMAKE_CXX_STANDARD} LESS ${MIN_CXX_STANDARD})
  message(FATAL_ERROR
    "C++ language standard must not be less than ${MIN_CXX_STANDARD}")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Hpg_BUILD_DOCS "Build Doxygen documentation" ON)
option(Hpg_BUILD_TESTS "Build unit tests" ON)

# the following are set according to Hpg_ENABLE_* cache variables and
# the devices enabled by Kokkos (in the src directory)
set(HPG_ENABLE_SERIAL OFF)
set(HPG_ENABLE_OPENMP OFF)
set(HPG_ENABLE_CUDA OFF)

add_subdirectory(dependencies)

add_subdirectory(src)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(${Hpg_BUILD_DOCS})
    add_subdirectory(doc)
  endif()
  if(${Hpg_BUILD_TESTS})
    add_subdirectory(tests)
  endif()
endif()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(hpgConfigVersion.cmake
  VERSION ${hpg_VER}
  COMPATIBILITY SameMajorVersion)
set(cmakeModulesDir "${CMAKE_INSTALL_LIBDIR}/cmake/hpg")
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/hpgConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hpg"
  PATH_VARS cmakeModulesDir
  NO_SET_AND_CHECK_MACRO)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hpgConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/hpgConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hpg")
install(EXPORT hpgTargets
  NAMESPACE hpg::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hpg")
