//
// Created by Komorowicz David on 2020. 06. 25..
//

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include "bundleadjust/BalDataloader.h"

class KinectDataloader {
public:
    KinectDataloader(const std::string &datasetDir);

private:
    std::vector<std::pair<float, float>> observations; // 2d points
    std::vector<size_t> obs_cam; //  ith 2d point on jth camera
    std::vector<size_t> obs_point; //  ith 2d point corresponds to jth 3d point
    std::vector<Camera> cameras;
    std::vector<std::tuple<float, float, float>> points;
};

