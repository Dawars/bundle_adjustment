cmake_minimum_required(VERSION 3.1...3.16)

add_executable(test_get_color get_color_test.cpp)

target_compile_features(test_get_color PRIVATE cxx_std_17)

target_link_libraries(test_get_color PRIVATE fmt::fmt)
target_link_libraries(test_get_color PRIVATE BundleAdjustment)
