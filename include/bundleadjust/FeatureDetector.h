//
// Created by Komorowicz David on 2020. 06. 25..
//

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

class FeatureDetector {
public:
    virtual std::vector<cv::KeyPoint> getFeatures(cv::Mat &color, cv::Mat &gray, std::unordered_map<std::string, float> params = {}) = 0;

    virtual ~FeatureDetector() { };
};

