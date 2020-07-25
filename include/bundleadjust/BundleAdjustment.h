//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <ceres/ceres.h>

#include "Dataloader.h"

class MeshWriterCallback;

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

    void WriteToPLYFile(std::string filename);

    Dataloader* dataset;

private:
    ceres::Solver::Options options;
    ceres::Problem problem;

    double *R;
    double *X;
    double *T;
    double *intrinsics;
    MeshWriterCallback *callback;
};

