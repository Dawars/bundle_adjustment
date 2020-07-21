//
// Created by Komorowicz David on 2020. 07. 05..
//

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class Dataloader {
public:
    virtual ~Dataloader() = 0;

    inline virtual int getObsCam(int index) const = 0;
    inline virtual int getObsPoint(int index) const = 0;
    inline virtual int getNumPoints() const = 0;
    inline virtual std::vector<cv::Point2f> getObservations() const = 0;
    inline virtual int getNumObservations() const = 0;
    inline virtual Eigen::Vector3d getPointColor(int index) const = 0;
    inline virtual int getNumFrames() const = 0;
    inline virtual bool isColorAvailable() const = 0;
    inline virtual bool isDepthAvailable() const = 0;
    inline virtual cv::Mat getColor(int frameId) const = 0;
    inline virtual cv::Mat getDepth(int frameId) const = 0;
    virtual void initialize(double* R, double* T, double* intrinsics, double* X) = 0;
};

inline Dataloader::~Dataloader(){}