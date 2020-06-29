//
// Created by Komorowicz David on 2020. 06. 29..
//

#include "bundleadjust/ShiTomasiDetector.h"

std::vector<cv::KeyPoint> ShiTomasiDetector::getFeatures(cv::Mat &color, cv::Mat &depth, std::unordered_map<std::string, float> params) {

    cv::Mat intensity;
    cvtColor(color, intensity, cv::COLOR_RGB2GRAY);

    std::vector<cv::Point2f> corners;

//    https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
    cv::goodFeaturesToTrack(intensity, corners, 25, 0.01, 10);

    std::vector<cv::KeyPoint> featurePoints;
    cv::KeyPoint::convert(corners, featurePoints, 10.0);
    return featurePoints;
}