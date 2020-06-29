//
// Created by Komorowicz David on 2020. 06. 29..
//

#pragma once

#include "bundleadjust/FeatureDetector.h"

class SiftDetector : FeatureDetector {
public:
    std::vector<cv::KeyPoint> getFeatures(cv::Mat &color, cv::Mat &gray, std::unordered_map<std::string, float> params = {}) override;
};

