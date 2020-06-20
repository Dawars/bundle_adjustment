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

private:
    int num_camera, num_points, num_observations;

    std::vector<std::pair<float, float>> observations; // 2d points
    std::vector<std::set<int>> cam_obs; //  ith camera jth 2d point // todo maybe reverse
    std::vector<Camera> cameras;
    std::vector<std::tuple<float, float, float>> points;

};
