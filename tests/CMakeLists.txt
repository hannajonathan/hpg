# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg)
# target_link_libraries(test gtest_main)
add_test(NAME test0 COMMAND test0)
