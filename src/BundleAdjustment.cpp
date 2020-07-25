//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>

#include <Eigen/Dense>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <fmt/os.h>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BAConstraint.h"
#include "MeshWriterCallback.h"


BundleAdjustment::BundleAdjustment(Dataloader *dataset, ceres::Solver::Options options)
        : dataset(dataset),
          options(options) {

    callback = new MeshWriterCallback(this);

    R = new double[dataset->getNumFrames() * 3];
    T = new double[dataset->getNumFrames() * 3];
    intrinsics = new double[dataset->getNumFrames() * 6];
    X = new double[dataset->getNumPoints() * 3]; // reconstructed 3D points

    dataset->initialize(R, T, intrinsics, X);
}

BundleAdjustment::~BundleAdjustment() {
    delete[] R;
    delete[] T;
    delete[] X;
    delete[] intrinsics;

    delete callback;
}

void BundleAdjustment::createProblem() {
    std::cout << "Creating problem" << std::endl;
    auto observations = dataset->getObservations();

    int invalidObs = 0;
    for (int i = 0; i < dataset->getNumObservations(); ++i) {

        // get camera for observation
        int camIndex = dataset->getObsCam(i);
        int pointIndex = dataset->getObsPoint(i);
        if (pointIndex == -1) {
            invalidObs++; // no 3d point to 2d point
            continue;
        }

        if (std::isnan(observations[i].x)) {
            continue;
        }

        Eigen::Vector3f obs;
        obs << observations[i].x, observations[i].y, 1.f;

        auto cost_function = BAConstraint::create(obs);
        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
                                 getPoint(pointIndex),
                                 getRotation(camIndex),
                                 getTranslation(camIndex),
                                 getIntrinsics(camIndex)
        );

        // adding cam intrinsics as fixed vars
        problem.SetParameterBlockConstant(getIntrinsics(camIndex));

        // todo group params http://ceres-solver.org/nnls_solving.html#parameterblockordering

    }
    std::cout << "Invalid observations: " << invalidObs << " out of " << dataset->getNumObservations() << std::endl;
    std::cout << "Creating problem end" << std::endl;

}

void BundleAdjustment::solve() {
    std::cout << "Solving problem" << std::endl;

    options.update_state_every_iteration = true;
    options.callbacks.push_back(callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << ceres::LinearSolverTypeToString(options.linear_solver_type) << std::endl;
    std::cout << summary.FullReport() << std::endl;

    std::cout << "Solving problem end" << std::endl;
}


void BundleAdjustment::writeMesh(std::string filename) {

    std::ofstream file(filename);
    if (file.is_open()) {
        file << "OFF" << std::endl;
        file << dataset->getNumPoints() << " 0 0" << std::endl;

        for (int i = 0; i < dataset->getNumPoints(); ++i) {
            auto point = getPoint(i);

            file << point[0] << " " << point[1] << " " << point[2] << std::endl;
        }
    }
}

double *BundleAdjustment::getRotation(size_t cameraIndex) {
    return &R[cameraIndex * 3];
}

double *BundleAdjustment::getTranslation(size_t cameraIndex) {
    return &T[cameraIndex * 3];
}

double *BundleAdjustment::getIntrinsics(size_t pointIndex) {
    return &intrinsics[pointIndex * 6];
}

double *BundleAdjustment::getPoint(size_t pointIndex) {
    return &X[pointIndex * 3];
}

void BundleAdjustment::WriteToPLYFile(std::string filename) {
    auto of = fmt::output_file(filename);

    of.print("ply"
    "\nformat ascii 1.0"
    "\nelement vertex {}"
    "\nproperty float x"
    "\nproperty float y"
    "\nproperty float z"
    "\nproperty uchar red"
    "\nproperty uchar green"
    "\nproperty uchar blue"
    "\nend_header\n", this->dataset->getNumPoints() + this->dataset->getNumFrames());

    for (int i = 0; i < this->dataset->getNumFrames(); ++i)  {
        auto cam = getTranslation(i);
        of.print("{0:.4f} {1:.4f} {2:.4f} 0 255 0\n", cam[0], cam[1], cam[2]);
    }

    for (int i = 0; i < this->dataset->getNumPoints(); ++i) {
        auto point = getPoint(i);
        Eigen::Vector3i bgr = this->dataset->getPointColor(i);
        of.print("{0:.4f} {1:.4f} {2:.4f} {3} {4} {5}\n", point[0], point[1], point[2], bgr(0), bgr(1), bgr(2));
    }
    of.close();
}