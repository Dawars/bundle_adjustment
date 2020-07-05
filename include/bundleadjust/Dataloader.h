//
// Created by Komorowicz David on 2020. 07. 05..
//

#pragma once

#include <opencv2/opencv.hpp>

class Dataloader {
public:
    virtual ~Dataloader() = 0;

    virtual int getObsCam(int index) const = 0;
    virtual int getObsPoint(int index) const = 0;
    virtual int getNumPoints() const = 0;
    virtual std::vector<cv::Point2f> getObservations() const = 0;
    virtual int getNumObservations() const = 0;
    virtual int getNumFrames() const = 0;
    virtual bool isColorAvailable() const = 0;
    virtual bool isDepthAvailable() const = 0;
    virtual cv::Mat getColor(int frameId) const = 0;
    virtual cv::Mat getDepth(int frameId) const = 0;
    virtual void initialize(double* R, double* T, double* intrinsics, double* X) = 0;
};

inline Dataloader::~Dataloader(){}