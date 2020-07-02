// //
// // Created by Komorowicz David on 2020. 06. 29..
// //

// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"

// #include "bundleadjust/SiftDetector.h"

// std::vector<cv::KeyPoint> SiftDetector::getFeatures(cv::Mat &color, cv::Mat &depth, std::unordered_map<std::string, float> params) {
//     cv::Mat intensity;
//     cvtColor(color, intensity, cv::COLOR_RGB2GRAY);

//     std::vector<cv::Point2f> corners;

// //    https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/features2D

//     cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
//     //cv::Ptr<Feature2D> f2d = cv2::xfeatures2d::SURF::create();
//     //cv::Ptr<Feature2D> f2d = cv2::ORB::create();

//     std::vector<cv::KeyPoint> keypoints;
//     f2d->detect(intensity, keypoints);

//     return keypoints;
// }