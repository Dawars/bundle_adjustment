//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <vector>
#include <set>
#include <opencv2/opencv.hpp>

#include "bundleadjust/Dataloader.h"

struct Camera {
    float R[3];
    float t[3];
    float f, k1, k2;
//
//    Eigen::Vector2f undistort(Eigen::Vector2f observation) const {
////        r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
//
//        float p2 = observation.squaredNorm();
//        float p4 = p2 * p2;
//        return 1 + k1 * p2 + k2 * p4;
//    }

//    Eigen::Matrix3f projectionMatrix() const {
//
//        // camera space, axis goes through "center of image"
//        Eigen::Matrix3f proj;
//        proj << f, 0., 0.,
//                0., f, 0.
//                0., 0., 1.;
//        return proj;
//    }
};

/**
 * Loader for the Bal dataset
 * https://grail.cs.washington.edu/projects/bal/
 */
class BalDataloader : public Dataloader {
public:
    BalDataloader(std::string path);
    ~BalDataloader() override;

    int num_camera, num_points, num_observations;

    std::vector<std::pair<float, float>> observations; // 2d points
    std::vector<size_t> obs_cam; //  ith 2d point on jth camera
    std::vector<size_t> obs_point; //  ith 2d point corresponds to jth 3d point
    std::vector<Camera> cameras;
    std::vector<std::tuple<float, float, float>> points;

    inline int getObsCam(int index) const override;
    inline int getObsPoint(int index) const override;
    inline int getNumPoints() const override;
    inline std::vector<cv::Point2f> getObservations() const override;
    inline int getNumObservations() const override;
    inline int getNumFrames() const override;

    inline bool isColorAvailable() const override;
    inline bool isDepthAvailable() const override;

    cv::Mat getColor(int frameId) const override;
    cv::Mat getDepth(int frameId) const override;

    virtual void initialize(double* R, double* T, double* intrinsics, double* X) override;

};
