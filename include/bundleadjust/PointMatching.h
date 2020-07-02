#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>

#include "bundleadjust/FeatureDetector.h"

class OnlinePointMatcher {
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    std::vector<std::vector<cv::KeyPoint>> keypoints; // list of keypoints for every frame processed so far
    std::vector<cv::Mat> descriptors;

    std::vector<int> obs_cam; //  ith 2d point on jth camera
    std::vector<int> obs_point; //  ith 2d point corresponds to jth 3d point

public:
    std::vector<Mat> images;
    std::vector<DMatch> matches;
    Mat current_frame;
    std::vector<std::string> image_paths;

    OnlinePointMatcher();

    void extractKeypoints(const cv::Mat currentFrame);

    void matchKeypoints();

    void configure_matcher(const Ptr<FeatureDetector> detector, const Ptr<DescriptorExtractor> extractor,
                           const Ptr<DescriptorMatcher> matcher);

};