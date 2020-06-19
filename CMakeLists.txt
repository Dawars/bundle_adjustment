cmake_minimum_required(VERSION 3.1...3.16)
project(Harris VERSION 0.1 DESCRIPTION "Harris corner detection lib" LANGUAGES CXX)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

add_subdirectory(libs/Sophus)
add_subdirectory(libs/ceres-solver)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(download_libtorch)
#include(ExternalProject)
#add_subdirectory(libs/pytorch)

add_subdirectory(src)
add_subdirectory(app)

