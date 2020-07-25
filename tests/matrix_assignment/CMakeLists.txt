cmake_minimum_required(VERSION 3.1...3.16)

add_executable(test_matrix_assignment matrix_assignment_test.cpp)

target_compile_features(test_matrix_assignment PRIVATE cxx_std_17)

target_link_libraries(test_matrix_assignment PRIVATE fmt::fmt)
target_link_libraries(test_matrix_assignment PRIVATE BundleAdjustment)
