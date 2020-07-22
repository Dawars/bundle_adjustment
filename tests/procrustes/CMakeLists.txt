cmake_minimum_required(VERSION 3.1...3.16)

add_executable(test_procrustes procrustes_test.cpp)

target_compile_features(test_procrustes PRIVATE cxx_std_17)

target_link_libraries(test_procrustes PRIVATE fmt::fmt)
target_link_libraries(test_procrustes PRIVATE BundleAdjustment)
