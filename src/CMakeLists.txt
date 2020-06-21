cmake_minimum_required(VERSION 3.1...3.16)

# Optionally glob, but only for CMake 3.12 or later:
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${BundleAdjustment_SOURCE_DIR}/include/bundleadjust/*.h")
#set(HEADER_LIST "${BundleAdjustment_SOURCE_DIR}/include/harris/harris.h")

add_library(BundleAdjustment ${HEADER_LIST} BalDataloader.cpp BundleAdjustment.cpp BAConstraint.cpp)

target_include_directories(BundleAdjustment PUBLIC ../include)

target_include_directories(BundleAdjustment PRIVATE ../libs/eigen)


if(OpenMP_CXX_FOUND)
    target_link_libraries(BundleAdjustment PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(BundleAdjustment PUBLIC Ceres::ceres)
#target_link_libraries(BundleAdjustment PRIVATE opencv_core sophus)
#target_link_libraries(BundleAdjustment PRIVATE "${TORCH_LIBRARIES}")