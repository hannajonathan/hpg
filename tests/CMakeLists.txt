add_executable(GridderStateUnitTests GridderStateUnitTests.cc)
target_link_libraries(GridderStateUnitTests hpg gtest)
add_test(NAME GridderStateUnitTests COMMAND GridderStateUnitTests)

add_executable(GridderUnitTests GridderUnitTests.cc)
target_link_libraries(GridderUnitTests hpg gtest)
add_test(NAME GridderUnitTests COMMAND GridderUnitTests)

add_executable(CFIndexerUnitTests CFIndexerUnitTests.cc)
target_link_libraries(CFIndexerUnitTests hpg gtest_main)
add_test(NAME CFIndexerUnitTests COMMAND CFIndexerUnitTests)

# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg)
