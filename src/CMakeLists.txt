cmake_minimum_required(VERSION 3.1...3.16)

# Optionally glob, but only for CMake 3.12 or later:
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Harris_SOURCE_DIR}/include/harris/*.h")
#set(HEADER_LIST "${Harris_SOURCE_DIR}/include/harris/harris.h")

add_library(Harris ${HEADER_LIST} harris.cpp BalDataloader.cpp BundleAdjustment.cpp BAConstraint.cpp)

target_include_directories(Harris PUBLIC ../include)

target_include_directories(Harris PRIVATE ../libs/eigen)


if(OpenMP_CXX_FOUND)
    target_link_libraries(Harris PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(Harris PUBLIC Ceres::ceres)
#target_link_libraries(Harris PRIVATE opencv_core sophus)
#target_link_libraries(Harris PRIVATE "${TORCH_LIBRARIES}")