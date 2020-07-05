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

    int getObsCam(int index) const override;
    int getObsPoint(int index) const override;
    int getNumPoints() const override;
    std::vector<cv::Point2f> getObservations() const override;
    int getNumObservations() const override;
    int getNumFrames() const override;

    bool isColorAvailable() const override;
    bool isDepthAvailable() const override;

    cv::Mat getColor(int frameId) const override;
    cv::Mat getDepth(int frameId) const override;

    void initialize(double* R, double* T, double* intrinsics, double* X) override;

private:
//    std::vector<Camera> cameras;
    OnlinePointMatcher* correspondenceFinder;
};

