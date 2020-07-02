#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>

#include "bundleadjust/FeatureDetector.h"

class OnlinePointMatcher {
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
    std::vector<std::vector<cv::KeyPoint>> keypoints; // list of keypoints for every frame processed so far
//    std::vector<std::vector<cv::Vec3b>> keypointColor; // todo for photometric loss
    std::vector<cv::Mat> descriptors;

    size_t currentId = 0;

    std::vector<int> obs_cam; //  ith 2d point on jth camera
    std::vector<int> obs_point; //  ith 2d point corresponds to jth 3d point
    std::vector<std::tuple<float, float, float>> points;

public:
    OnlinePointMatcher();
    void extractKeypoints(const cv::Mat currentFrame);
    void matchKeypoints();
};