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
file(COPY gridone.h5 DESTINATION .)
add_executable(GridderStateUnitTests GridderStateUnitTests.cc)
target_link_libraries(GridderStateUnitTests
  hpg::hpg
  hdf5::hdf5_cpp
  hdf5::hdf5
  gtest)
add_test(NAME GridderStateUnitTests COMMAND GridderStateUnitTests)
set_tests_properties(GridderStateUnitTests PROPERTIES TIMEOUT 90)

set(_HPG_EXP_GRIDDING_KERNEL_VERSION 0)
if(Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    AND NOT ${_HPG_EXP_GRIDDING_KERNEL_VERSION} EQUAL 0)
  add_executable(ExpGridderStateUnitTests GridderStateUnitTests.cc)
  target_link_libraries(ExpGridderStateUnitTests hpg::hpg gtest)
  target_compile_definitions(ExpGridderStateUnitTests PRIVATE
    HPG_GRIDDING_KERNEL_VERSION=${_HPG_EXP_GRIDDING_KERNEL_VERSION}
    HPG_DELTA_EXPERIMENTAL_ONLY)
  add_test(NAME ExpGridderStateUnitTests COMMAND ExpGridderStateUnitTests)
  set_tests_properties(ExpGridderStateUnitTests PROPERTIES TIMEOUT 90)
endif()

add_executable(GridderUnitTests GridderUnitTests.cc)
target_link_libraries(GridderUnitTests hpg::hpg gtest)
add_test(NAME GridderUnitTests COMMAND GridderUnitTests)

add_executable(CFIndexerUnitTests CFIndexerUnitTests.cc)
target_link_libraries(CFIndexerUnitTests hpg::hpg gtest_main)
add_test(NAME CFIndexerUnitTests COMMAND CFIndexerUnitTests)

add_executable(GridArraysUnitTests GridArraysUnitTests.cc)
target_link_libraries(GridArraysUnitTests hpg::hpg gtest)
add_test(NAME GridArraysUnitTests COMMAND GridArraysUnitTests)

add_executable(DeviceCFArrayUnitTests DeviceCFArrayUnitTests.cc)
target_link_libraries(DeviceCFArrayUnitTests hpg::hpg gtest)
add_test(NAME DeviceCFArrayUnitTests COMMAND DeviceCFArrayUnitTests)
set_tests_properties(DeviceCFArrayUnitTests PROPERTIES TIMEOUT 120)

# if(Hpg_ENABLE_MPI)
#   add_executable(MPIRuntimeUnitTests MPIRuntimeUnitTests.cc)
#   target_link_libraries(MPIRuntimeUnitTests hpg::hpg gtest_main)
#   add_test(NAME MPIRuntimeUnitTests COMMAND MPIRuntimeUnitTests)
# endif()

# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg::hpg)
