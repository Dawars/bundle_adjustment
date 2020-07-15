#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "bundleadjust/FeatureDetector.h"

class OnlinePointMatcher {
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    std::unordered_map<std::string, float> params;

     // list of keypoints for every frame processed so far
    std::vector<cv::Mat> descriptors;

    int numPoints3d = 0; // 3d points


public:
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<int> obs_cam; //  ith 2d point on jth camera
    std::vector<int> obs_point; //  ith 2d point corresponds to jth 3d point
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;


    OnlinePointMatcher(const cv::Ptr<cv::FeatureDetector> detector,
                       const cv::Ptr<cv::DescriptorExtractor> extractor,
                       const cv::Ptr<cv::DescriptorMatcher>  matcher,
                       std::unordered_map<std::string, float> params);

    void extractKeypoints(const cv::Mat currentFrame);

    void matchKeypoints(std::vector<cv::Mat> & depthImages, Eigen::Matrix3f & intrinsics);

    std::vector<cv::Point2f> getObservations() const;

    int getObsCam(int index) const;
    int getObsPoint(int index) const;
    int getNumPoints() const;
    int getNumObservations() const;
    int getNumFrames() const;
    std::vector<std::vector<cv::KeyPoint>> getKeyPoints() const;
    std::vector<int> getObsCam() const;
    std::vector<int> getObsPoint() const;
};