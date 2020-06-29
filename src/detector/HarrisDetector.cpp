//
// Created by Komorowicz David on 2020. 06. 29..
//

#include "bundleadjust/HarrisDetector.h"

std::vector<cv::KeyPoint> HarrisDetector::getFeatures(cv::Mat &color, cv::Mat& depth, std::unordered_map<std::string, float> params) {
    int blockSize = params["blockSize"];
    int apertureSize = params["apertureSize"];
    double k = params["k"];
    int thresh = params["thresh"];

    cv::Mat intensity;
    cvtColor(color, intensity, cv::COLOR_RGB2GRAY);

    cv::Mat dst = cv::Mat(intensity.size(), CV_8UC1);

//    https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
    cv::cornerHarris(intensity, dst, blockSize, apertureSize, k);

    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    std::vector<cv::Point2f> corners;
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int) dst_norm.at<float>(i, j) > thresh) {
                corners.push_back(cv::Point2f(i,j));
            }
        }
    }
    // todo subpixel

    std::vector<cv::KeyPoint> featurePoints;
    cv::KeyPoint::convert(corners, featurePoints, 10.0);

    return featurePoints;
}