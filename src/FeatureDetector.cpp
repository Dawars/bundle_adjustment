//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/FeatureDetector.h"

std::vector<Eigen::Vector2f> FeatureDetector::getFeatures(cv::Mat &color, cv::Mat& depth) {
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;

    cv::Mat intensity;
    cvtColor(color, intensity, cv::COLOR_RGB2GRAY);

    cv::Mat dst = cv::Mat(intensity.size(), CV_8UC1);

//    https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
    cv::cornerHarris(intensity, dst, blockSize, apertureSize, k);

    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    std::vector<Eigen::Vector2f> featurePoints;
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int) dst_norm.at<float>(i, j) > thresh) {
                featurePoints.push_back({i,j});
            }
        }
    }
    // todo subpixel
    // todo skip blurry images no features
    return featurePoints;
}

cv::Mat FeatureDetector::getDescriptors(cv::Mat &color, cv::Mat &gray, std::vector<Eigen::Vector2f> &features) {


    return cv::Mat();
}
