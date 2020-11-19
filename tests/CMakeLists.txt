add_subdirectory(dependencies)

add_executable(GridderStateUnitTests GridderStateUnitTests.cc)
target_link_libraries(GridderStateUnitTests hpg gtest)
add_test(NAME GridderStateUnitTests COMMAND GridderStateUnitTests)

add_executable(GridderUnitTests GridderUnitTests.cc)
target_link_libraries(GridderUnitTests hpg gtest)
add_test(NAME GridderUnitTests COMMAND GridderUnitTests)

# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg)
