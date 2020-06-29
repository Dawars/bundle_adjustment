//
// Created by Komorowicz David on 2020. 06. 25..
//

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class FeatureDetector {
public:
    std::vector<Eigen::Vector2f> getFeatures(cv::Mat &color, cv::Mat &gray);

    cv::Mat getDescriptors(cv::Mat &color, cv::Mat &gray, std::vector<Eigen::Vector2f> &features);

};

