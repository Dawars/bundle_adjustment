//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <ceres/ceres.h>

#include "BalDataloader.h"

class BundleAdjustment {
public:
    BundleAdjustment(const BalDataloader& dataset);
    virtual ~BundleAdjustment();

    void createProblem();
    void solve();

    double* getRotation(size_t cameraIndex);
    double* getTranslation(size_t cameraIndex);
    double* getPoint(size_t pointIndex);

    void projectFrom3D(int cam_id);
    void writeMesh(std::string filename);
private:
    ceres::Problem problem;

    BalDataloader dataset;
    double *R;
    double *T;
    double *X;
    void reset();
    void configureSolver(ceres::Solver::Options& options);
};

