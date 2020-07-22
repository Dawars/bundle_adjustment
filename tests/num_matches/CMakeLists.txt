cmake_minimum_required(VERSION 3.1...3.16)

add_executable(test_num_matches num_matches_test.cpp)

target_compile_features(test_num_matches PRIVATE cxx_std_17)

target_link_libraries(test_num_matches PRIVATE fmt::fmt)
target_link_libraries(test_num_matches PRIVATE BundleAdjustment)
