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
//    std::unordered_map<std::string, float> params = {
//            {"blockSize",    2.0},
//            {"apertureSize", 3.0},
//            {"k",            0.04},
//            {"thresh",       200}
//    };
//
//    SiftDetector detector;
//    HarrisDetector detector;
//    ShiTomasiDetector detector;

    auto detector = SIFT::create(); // TODO: Extend and use FeatureDetector wrapper
    auto extractor = SIFT::create();
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    correspondenceFinder = new OnlinePointMatcher{detector, extractor, matcher, {{"ratioThreshold", 0.7}}};

    VirtualSensor sensor{};
    sensor.Init(datasetDir);

    auto intrinsics = sensor.GetColorIntrinsics();
    this->intrinsics[0] = intrinsics(0, 0);
    this->intrinsics[0] = intrinsics(1, 1);
    this->intrinsics[0] = intrinsics(0, 2);
    this->intrinsics[0] = intrinsics(1, 2);
    this->intrinsics[0] = 0;
    this->intrinsics[0] = 0;

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColor();
        auto depth = sensor.GetDepth();
        // todo filter depth map
        colorImages.push_back(color);
        depthImages.push_back(depth);

        correspondenceFinder->extractKeypoints(color);
    }

    correspondenceFinder->matchKeypoints();

    // TODO: depth test

    // TODO: visualize matches

}

KinectDataloader::~KinectDataloader() {
    delete this->correspondenceFinder;
}

int KinectDataloader::getObsCam(int index) const {
    return this->correspondenceFinder->getObsCam(index);
}

int KinectDataloader::getObsPoint(int index) const {
    return this->correspondenceFinder->getObsPoint(index);
}

int KinectDataloader::getNumPoints() const {
    return this->correspondenceFinder->getNumPoints();
}

std::vector<cv::Point2f> KinectDataloader::getObservations() const {
    return this->correspondenceFinder->getObservations();
}

int KinectDataloader::getNumObservations() const {
    return this->correspondenceFinder->getNumPoints();
}

int KinectDataloader::getNumFrames() const {
    return this->correspondenceFinder->getNumFrames();
}

bool KinectDataloader::isColorAvailable() const {
    return true;
}

bool KinectDataloader::isDepthAvailable() const {
    return true;
}

cv::Mat KinectDataloader::getColor(int frameId) const {
    return this->colorImages[frameId];
}

cv::Mat KinectDataloader::getDepth(int frameId) const {
    return this->depthImages[frameId];
}

void KinectDataloader::initialize(double *R, double *T, double *intrinsics, double *X) {
    for (int i = 0; i < this->getNumFrames(); ++i) {
        R[3 * i + 0] = 0; // todo init from procrutes
        R[3 * i + 1] = 0;
        R[3 * i + 2] = 0;
        T[3 * i + 0] = 0;
        T[3 * i + 1] = 0;
        T[3 * i + 2] = 0;

        for (int j = 0; j < 6; ++j) {
            intrinsics[3 * i + j] = this->intrinsics[j];
        }
    }

    for (int i = 0; i < this->getNumPoints(); ++i) {
        // todo init from procrutes

//        X[3*i + 0] = u;
//        X[3*i + 1] = v;
//        X[3*i + 2] = depthImages[frameId].at<double>(u, v);

    }
}
