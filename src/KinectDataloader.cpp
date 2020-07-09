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

void visualizeMatch(cv::Mat img1, cv::Mat img2, OnlinePointMatcher *matcher) {
    std::vector<cv::DMatch> matches;
    auto kpts = matcher->getKeyPoints();
    std::vector<cv::Point2f> pt1;
    std::vector<cv::Point2f> pt2;
    std::vector<int> idx1, idx2;
    for (int i = 0; i < kpts[0].size(); ++i) {
        if (matcher->getObsPoint(i) != -1) {
            int offset = kpts[0].size();
            for (int j = 0; j < kpts[1].size(); ++j) {
                if (matcher->getObsPoint(i) == matcher->getObsPoint(offset + j)) {
                    pt1.push_back(kpts[0][i].pt);
                    pt2.push_back(kpts[1][j].pt);
                    idx1.push_back(i);
                    idx2.push_back(j);
                }
            }
        }
    }
    cv::Mat fundamental_matrix =
            findHomography(pt1, pt2, cv::FM_RANSAC);

    double eps = 1e1;
    for (int i = 0; i < pt1.size(); ++i) {
        cv::Point3d p1 = {pt1[i].x, pt1[i].y, 1};
        cv::Point3d p2 = {pt2[i].x, pt2[i].y, 1};
        cv::Mat mp1(p1);
        cv::Mat mp2(p2);
        cv::Mat mp3 = fundamental_matrix * mp1;
        cv::Point3d p3(mp3);
        p3.x /= p3.z;
        p3.y /= p3.z;
        if (cv::norm(p2-p3) < eps) {
            matches.push_back(cv::DMatch(idx1[i], idx2[i], 1.));
        }
    }

    cv::Mat out;
    cv::drawMatches(img1, kpts[0], img2, kpts[1], matches, out);

    std::string name("name");
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

    correspondenceFinder = new OnlinePointMatcher{detector, extractor, matcher, {{"ratioThreshold", 0.7},
                                                                                 {"ransacEps", 1e1}}};

    VirtualSensor sensor{};
    sensor.Init(datasetDir);

    auto intrinsics = sensor.GetColorIntrinsics();
    this->intrinsics[0] = intrinsics(0, 0);
    this->intrinsics[1] = intrinsics(1, 1);
    this->intrinsics[2] = intrinsics(0, 2);
    this->intrinsics[3] = intrinsics(1, 2);
    this->intrinsics[4] = 0;
    this->intrinsics[5] = 0;

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColor();
        auto depth = sensor.GetDepth();
        // todo filter depth map
        colorImages.push_back(color);
        depthImages.push_back(depth);

        correspondenceFinder->extractKeypoints(color);
    }

    correspondenceFinder->matchKeypoints();
//    visualizeMatch(color1, color2, correspondenceFinder);
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
            intrinsics[6 * i + j] = this->intrinsics[j];
        }
    }

    for (int i = 0; i < this->getNumPoints(); ++i) {
        // todo init from procrutes

//        X[3*i + 0] = u;
//        X[3*i + 1] = v;
//        X[3*i + 2] = depthImages[frameId].at<double>(u, v);

    }
}
