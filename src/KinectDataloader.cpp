//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/FeatureDetector.h"
#include "VirtualSensor.h"

void visualize(cv::Mat image, std::vector<Eigen::Vector2f> featurePoints){

    std::string name{"Harris corners"};

    for(auto &point : featurePoints){
        cv::circle(image, cv::Point(point(0), point(1)), 5, cv::Scalar(0), 2, 8, 0);
    }
    cv::namedWindow(name);
    cv::imshow(name, image);
    cv::waitKey(0);
}

KinectDataloader::KinectDataloader(const std::string &datasetDir) {
    FeatureDetector detector;

    VirtualSensor sensor;
    sensor.Init(datasetDir);

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColorRGBX();
        auto depth = sensor.GetDepth();

        // Feature Detection and Matching
        // https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
        auto featurePoints = detector.getFeatures(color, depth);

        visualize(color, featurePoints);

    }
}
