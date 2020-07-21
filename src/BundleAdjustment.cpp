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


BundleAdjustment::BundleAdjustment(Dataloader *dataset, ceres::Solver::Options options)
        : dataset(dataset),
          options(options) {

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
        if(dataset->isColorAvailable()){
            // todo add photometric loss
        }

        if(dataset->isDepthAvailable()){
            // todo add depth loss
        }
    }
    std::cout << "Invalid observations: " << invalidObs << " out of " << dataset->getNumObservations() << std::endl;
    std::cout << "Creating problem end" << std::endl;

}

void BundleAdjustment::solve() {
    std::cout << "Solving problem" << std::endl;

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
        double Tx = this->T[i*3];
        double Ty = this->T[i*3+1];
        double Tz = this->T[i*3+2];
        of << Tx << ' ' << Ty << ' ' << Tz
        << " 0 255 0" << '\n';
    }

    for (int i = 0; i < this->dataset->getNumPoints(); ++i) {
        for (int j = 0; j < 3; ++j) {
            of << X[i*3+j] << ' ';
        }
        Eigen::Vector3d bgr = this->dataset->getPointColor(i);
        of << ' ' << bgr(2) << ' ' << bgr(1) << ' ' << bgr(0) << "\n";
    }
    of.close();
}