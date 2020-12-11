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
