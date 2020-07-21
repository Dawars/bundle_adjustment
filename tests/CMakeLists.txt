cmake_minimum_required(VERSION 3.1...3.16)

add_executable(tests procrustes_test.cpp)

target_compile_features(tests PRIVATE cxx_std_17)

target_link_libraries(tests PRIVATE fmt::fmt)
target_link_libraries(tests PRIVATE BundleAdjustment)
