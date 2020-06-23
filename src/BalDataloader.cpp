//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>
#include <tuple>

#include "./bundleadjust/BalDataloader.h"

BalDataloader::BalDataloader(std::string path) {
//    <num_cameras> <num_points> <num_observations>
//    <camera_index_1> <point_index_1> <x_1> <y_1>
//    ...
//    <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
//    <camera_1>
//    ...
//    <camera_num_cameras>
//    <point_1>
//    ...
//    <point_num_points>

    std::ifstream file(path);

    file >> num_camera >> num_points >> num_observations;

    obs_cam.resize(num_observations);
    obs_point.resize(num_observations);
    observations.resize(num_observations);

    for (int i = 0; i < num_observations; ++i) {
        int cam_index, point_index;
        float x, y; // 2D image coordinates
        file >> cam_index >> point_index >> x >> y;

        observations[i] = std::pair<float, float>(x, y);
        obs_cam[i] = cam_index;
        obs_point[i] = point_index;
    }

    cameras.resize(num_camera);

    for (int i = 0; i < num_camera; ++i) {
//        R3,t3,f,k1 and k2
        float R[3], T[3], f, k1, k2;
        file >> R[0] >> R[1] >> R[2] >> T[0] >> T[1] >> T[2] >> f >> k1 >> k2;

        cameras[i] = {R[0], R[1], R[2], T[0], T[1], T[2], f, k1, k2};
    }

    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        // 3D ground truth - multiple lines
        float x, y, z;
        file >> x >> y >> z;
        points[i] = {x, y, z};
    }
}
