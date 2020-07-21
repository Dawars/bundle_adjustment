cmake_minimum_required(VERSION 3.1...3.16)
project(BundleAdjustment VERSION 0.1 DESCRIPTION "BundleAdjustment project for 3D Reconstruction" LANGUAGES CXX)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

add_subdirectory(libs/fmt)
#add_subdirectory(libs/Sophus)
add_subdirectory(libs/ceres-solver)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#include(download_libtorch)
#include(ExternalProject)
#add_subdirectory(libs/pytorch)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests)

