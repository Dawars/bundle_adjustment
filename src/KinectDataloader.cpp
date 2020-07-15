//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/PointMatching.h"
#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/HarrisDetector.h"
#include "bundleadjust/ShiTomasiDetector.h"
#include "bundleadjust/SiftDetector.h"
#include "bundleadjust/ProcrustesAligner.h"

#include "VirtualSensor.h"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <ceres/rotation.h>

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

    auto detector = cv::SIFT::create(); // TODO: Extend and use FeatureDetector wrapper
    auto extractor = cv::SIFT::create();
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    correspondenceFinder = new OnlinePointMatcher{detector, extractor, matcher, {{"ratioThreshold", 0.5},
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

    Eigen::Matrix3f depthIntrinsicsInv = sensor.GetDepthIntrinsics().inverse();
    correspondenceFinder->matchKeypoints(depthImages, depthIntrinsicsInv);
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

    const int origin_frame = this->getNumFrames() / 2; // One should be able to choose if they know the optimal frame
    std::vector<Eigen::Vector3f> source_points;
    std::vector<int> source_points_indices;

    // Get all the key points from the origin frame coords (x,y,z) and point index
    for(int i=0; i<this->correspondenceFinder->getNumObservations(); i++) {
        if(this->correspondenceFinder->obs_cam[i] == origin_frame && this->correspondenceFinder->obs_point[i] != -1) {
            Eigen::Vector3f p;
            p << correspondenceFinder->x[i], correspondenceFinder->y[i], correspondenceFinder->z[i];
            source_points.push_back(p);
            source_points_indices.push_back(correspondenceFinder->obs_point[i]);
        }
    }


    for (int i=0; i<this->getNumFrames(); i++) {
        if(i == origin_frame) {
            R[3 * i + 0] = 0;
            R[3 * i + 1] = 0;
            R[3 * i + 2] = 0;
            T[3 * i + 0] = 0;
            T[3 * i + 1] = 0;
            T[3 * i + 2] = 0;
        } else {
            std::vector<Eigen::Vector3f> target_points;
            std::vector<int> target_points_indices;

            // get all the key points from the target frame
            for(int j=0; j<this->correspondenceFinder->getNumObservations(); j++) {
                if(this->correspondenceFinder->obs_cam[j] == i && this->correspondenceFinder->obs_point[j] != -1) {
                    Eigen::Vector3f p;
                    p << correspondenceFinder->x[j], correspondenceFinder->y[j], correspondenceFinder->z[j];
                    target_points.push_back(p);
                    target_points_indices.push_back(correspondenceFinder->obs_point[j]);
                }
            }
            // remove all the key points that aren't in both vectors and exclude points with negative infinity depth
            std::vector<Eigen::Vector3f> matching_source_points;
            std::vector<Eigen::Vector3f> matching_target_points;
            for(int j=0; j<source_points.size(); j++) {
                for(int k=0; k<target_points.size(); k++) {
                    if(source_points_indices[j] == target_points_indices[k]) {
                        if(std::isinf(source_points[j](2))) break;
                        if(std::isinf(target_points[k](2))) break;

                        matching_source_points.push_back(source_points[j]);
                        matching_target_points.push_back(target_points[k]);

                        //std::cout << source_points[j](0) << " " << source_points[j](1) << " " << source_points[j](2) << "\n\n";

                        break;
                    }
                }
            }

            ProcrustesAligner aligner;
	        Eigen::Matrix4f estimatedPose = aligner.estimatePose(matching_target_points, matching_source_points);
            Eigen::Matrix3f rotation_matrix;
            rotation_matrix << estimatedPose(0,0), estimatedPose(0,1), estimatedPose(0,2),
                               estimatedPose(1,0), estimatedPose(1,1), estimatedPose(1,2),
                               estimatedPose(2,0), estimatedPose(2,1), estimatedPose(2,2);
            Eigen::AngleAxis<float> r = Eigen::AngleAxis<float>(rotation_matrix);

            //std::cout << estimatedPose << "\n\n";


            R[3 * i + 0] = r.axis()(0);
            R[3 * i + 1] = r.axis()(1);
            R[3 * i + 2] = r.axis()(2);
            T[3 * i + 0] = estimatedPose(0,3);
            T[3 * i + 1] = estimatedPose(1,3);
            T[3 * i + 2] = estimatedPose(2,3);
        }
    }

    for (int i = 0; i < this->getNumPoints(); ++i) {
        // todo init from procrutes

        X[3*i + 0] = 0;
        X[3*i + 1] = 0;
        X[3*i + 2] = 1;

    }
}
