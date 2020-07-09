//
// Created by Komorowicz David on 2020. 06. 25..
//

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include <Eigen/Dense>

#include "bundleadjust/Dataloader.h"
#include "bundleadjust/PointMatching.h"

class KinectDataloader : public Dataloader {
public:
    KinectDataloader(const std::string &datasetDir, bool initGroundTruth = false);
    virtual ~KinectDataloader() override;

    std::vector<cv::Mat> colorImages;
    std::vector<cv::Mat> depthImages;

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
    bool initGroundTruth;
    OnlinePointMatcher* correspondenceFinder;
    double intrinsics[6]; // fx, fy, ox, oy, k1, k2
    std::vector<Eigen::Matrix4f> trajectories;
};

