cmake_minimum_required(VERSION 3.1...3.16)

add_executable(app main.cpp)

 target_link_libraries(app PRIVATE BundleAdjustment)
