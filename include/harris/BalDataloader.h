//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <vector>
#include <set>

struct Camera {
    float R[3];
    float t[3];
    float f, k1, k2;
};

/**
 * Loader for the Bal dataset
 * https://grail.cs.washington.edu/projects/bal/
 */
class BalDataloader {
public:
    BalDataloader(std::string path);

public:
    int num_camera, num_points, num_observations;

    std::vector<std::pair<float, float>> observations; // 2d points
    std::vector<size_t> obs_cam; //  ith 2d point on jth camera
    std::vector<size_t> obs_point; //  ith 2d point corresponds to jth 3d point
    std::vector<Camera> cameras;
    std::vector<std::tuple<float, float, float>> points;

};
