add_executable(tests test.cpp)
target_link_libraries(tests PUBLIC gtest gtest_main pthread matrix neuralnetwork dataset)

include(GoogleTest)
gtest_discover_tests(tests)