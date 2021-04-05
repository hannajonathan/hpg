# Copyright 2021 Associated Universities, Inc. Washington DC, USA.
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

# documentation
#
find_package(Doxygen)
set(DOXYGEN_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/../src)
# convince doxygen to document c++14-or-later functions
set(DOXYGEN_PREDEFINED __cplusplus=201402)
doxygen_add_docs(
  doc
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  COMMENT "Generate documentation")
