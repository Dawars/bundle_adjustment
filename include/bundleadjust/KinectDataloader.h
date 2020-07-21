//
// Created by Komorowicz David on 2020. 06. 25..
//

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include "bundleadjust/Dataloader.h"
#include "bundleadjust/PointMatching.h"

class KinectDataloader : public Dataloader {
public:
    KinectDataloader(const std::string &datasetDir);
    virtual ~KinectDataloader() override;

    std::vector<cv::Mat> colorImages;
    std::vector<cv::Mat> depthImages;

    Eigen::Vector3d getPointColor(int point_index) const override;

    inline int getObsCam(int index) const override;
    inline int getObsPoint(int index) const override;
    inline int getNumPoints() const override;
    inline std::vector<cv::Point2f> getObservations() const override;
    inline int getNumObservations() const override;
    inline int getNumFrames() const override;
    

    inline bool isColorAvailable() const override;
    inline bool isDepthAvailable() const override;

    inline cv::Mat getColor(int frameId) const override;
    inline cv::Mat getDepth(int frameId) const override;

    void initialize(double* R, double* T, double* intrinsics, double* X) override;

private:
//    std::vector<Camera> cameras;
    void setupPointDepth();
    OnlinePointMatcher* correspondenceFinder;
    Eigen::Matrix3f intrinsics; // fx, fy, ox, oy
    std::vector<double> x, y, z;

    std::vector<Eigen::Matrix4f> estimatedPoses;
};

