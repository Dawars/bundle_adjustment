//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/HarrisDetector.h"
#include "bundleadjust/ShiTomasiDetector.h"
#include "bundleadjust/SiftDetector.h"
#include "VirtualSensor.h"

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

        SiftDetector detector;
//    HarrisDetector detector;
//    ShiTomasiDetector detector;

    VirtualSensor sensor;
    sensor.Init(datasetDir);

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColor();
        auto depth = sensor.GetDepth();

        // Feature Detection and Matching
        // https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
        auto featurePoints = detector.getFeatures(color, depth, params);

        visualize(color, featurePoints);

    }
}
