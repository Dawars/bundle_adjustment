cmake_minimum_required(VERSION 3.1...3.16)

add_executable(test_visualize_match visualize_match_test.cpp)

target_compile_features(test_visualize_match PRIVATE cxx_std_17)

target_link_libraries(test_visualize_match PRIVATE fmt::fmt)
target_link_libraries(test_visualize_match PRIVATE BundleAdjustment)
