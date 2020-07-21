//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <ceres/ceres.h>

#include "Dataloader.h"

class BundleAdjustment {
public:
    BundleAdjustment(Dataloader* dataset, ceres::Solver::Options options);
    virtual ~BundleAdjustment();

    void createProblem();
    void solve();

    double* getRotation(size_t cameraIndex);
    double* getTranslation(size_t cameraIndex);
    double* getIntrinsics(size_t cameraIndex);
    double* getPoint(size_t pointIndex);

    void projectFrom3D(int cam_id);
    void writeMesh(std::string filename);
    void WriteToPLYFile(std::string& filename);
private:
    Dataloader* dataset;
    ceres::Solver::Options options;
    ceres::Problem problem;

    double *R;
    double *T;
    double *X;
    double *intrinsics;
};

