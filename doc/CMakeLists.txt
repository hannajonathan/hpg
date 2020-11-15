# documentation
#
find_package(Doxygen)
doxygen_add_docs(
  doc
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  COMMENT "Generate documentation")
