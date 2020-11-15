cmake_minimum_required(VERSION 3.14)
project(hpg)

enable_language(CXX)

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

set(MIN_CXX_STANDARD 17)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${MIN_CXX_STANDARD})
endif()
if(${CMAKE_CXX_STANDARD} LESS ${MIN_CXX_STANDARD})
  message(FATAL_ERROR
    "C++ language standard must not be less than ${MIN_CXX_STANDARD}")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(Hpg_ENABLE_SERIAL "Enable Serial device" OFF)
option(Hpg_ENABLE_OPENMP "Enable OpenMP device" OFF)
option(Hpg_ENABLE_CUDA "Enable CUDA device" OFF)

# get findFFTW
#
if(Hpg_ENABLE_SERIAL OR Hpg_ENABLE_OPENMP)
  configure_file(downloadFindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
  if(result)
    message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
  else()
    message("CMake step for findFFTW completed (${result}).")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
  if(result)
    message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
  endif()

  set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")

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

# get googletest
#
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# Now simply link against gtest or gtest_main as needed. Eg
# add_executable(example example.cpp)
# target_link_libraries(example gtest_main)
# add_test(NAME example_test COMMAND example)

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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GenerateExportHeader)
generate_export_header(hpg)

install(TARGETS hpg EXPORT HPGConfig)
install(FILES ${HPG_INCLUDE}
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}/hpg)
install(EXPORT HPGConfig DESTINATION share/hpg/cmake)
export(TARGETS hpg FILE HPGConfig.cmake)

# basic test application
#
add_executable(test test.cc)
target_link_libraries(test hpg)

# documentation
#
find_package(Doxygen)
doxygen_add_docs(
  doc
  hpg.hpp hpg_impl.hpp hpg.cc
  COMMENT "Generate documentation")
