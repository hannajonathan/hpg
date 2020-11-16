add_executable(utGridderState utGridderState.cc)
target_link_libraries(utGridderState hpg gtest)
add_test(NAME utGridderState COMMAND utGridderState)

# basic test application
#
add_executable(test0 test.cc)
target_link_libraries(test0 hpg)
