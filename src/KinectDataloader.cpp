//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/PointMatching.h"
#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/HarrisDetector.h"
#include "bundleadjust/ShiTomasiDetector.h"
#include "bundleadjust/SiftDetector.h"
#include "VirtualSensor.h"


#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;

void visualize(cv::Mat image, std::vector<cv::KeyPoint> featurePoints) {

    std::string name{"Detected corners"};
    cv::Mat out(image.size(), image.type());
    cv::drawKeypoints(image, featurePoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::namedWindow(name);
    cv::imshow(name, out);
    cv::waitKey(0);
}

KinectDataloader::KinectDataloader(const std::string &datasetDir) {
    std::unordered_map<std::string, float> params = {
            {"blockSize",    2.0},
            {"apertureSize", 3.0},
            {"k",            0.04},
            {"thresh",       200}
    };

//    SiftDetector detector;
//    HarrisDetector detector;
//    ShiTomasiDetector detector;

    auto detector = SIFT::create(); // TODO: Extend and use FeatureDetector wrapper
    auto extractor = SIFT::create();
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    OnlinePointMatcher correspondenceFinder{detector, extractor, matcher,
                                            {{"ratioThreshold", 0.7},}};

    VirtualSensor sensor{};
    sensor.Init(datasetDir);

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColor();
        auto depth = sensor.GetDepth();

        // Feature Detection and Matching
        // https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
//        auto featurePoints = detector.getFeatures(color, depth, params);

//        visualize(color, featurePoints);

        correspondenceFinder.extractKeypoints(color);
        // TODO: save depth and color values at feature points

    }

    correspondenceFinder.matchKeypoints();

    // TODO: depth test

    // TODO: visualize matches, images need to be stored in memory


    // TODO: Save out data from matcher
}
