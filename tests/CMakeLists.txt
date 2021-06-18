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
add_executable(GridderStateUnitTests GridderStateUnitTests.cc)
target_link_libraries(GridderStateUnitTests hpg::hpg gtest)
add_test(NAME GridderStateUnitTests COMMAND GridderStateUnitTests)
set_tests_properties(GridderStateUnitTests PROPERTIES TIMEOUT 45)

if(Hpg_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS)
  add_executable(ExpGridderStateUnitTests GridderStateUnitTests.cc)
  target_link_libraries(ExpGridderStateUnitTests hpg::hpg gtest)
  target_compile_definitions(ExpGridderStateUnitTests PRIVATE
    HPG_GRIDDING_KERNEL_VERSION=1
    HPG_DELTA_EXPERIMENTAL_ONLY)
  add_test(NAME ExpGridderStateUnitTests COMMAND ExpGridderStateUnitTests)
  set_tests_properties(ExpGridderStateUnitTests PROPERTIES TIMEOUT 45)
endif()

add_executable(GridderUnitTests GridderUnitTests.cc)
target_link_libraries(GridderUnitTests hpg::hpg gtest)
add_test(NAME GridderUnitTests COMMAND GridderUnitTests)

add_executable(CFIndexerUnitTests CFIndexerUnitTests.cc)
target_link_libraries(CFIndexerUnitTests hpg::hpg gtest_main)
add_test(NAME CFIndexerUnitTests COMMAND CFIndexerUnitTests)

add_executable(GridArraysUnitTests GridArraysUnitTests.cc)
target_link_libraries(GridArraysUnitTests hpg::hpg gtest_main)
add_test(NAME GridArraysUnitTests COMMAND GridArraysUnitTests)

add_executable(DeviceCFArrayUnitTests DeviceCFArrayUnitTests.cc)
target_link_libraries(DeviceCFArrayUnitTests hpg::hpg gtest_main)
add_test(NAME DeviceCFArrayUnitTests COMMAND DeviceCFArrayUnitTests)
set_tests_properties(DeviceCFArrayUnitTests PROPERTIES TIMEOUT 60)

# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg::hpg)
