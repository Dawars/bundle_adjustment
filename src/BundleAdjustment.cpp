//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>

#include <Eigen/Dense>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

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
/*
template<typename T>
void transformation(T *R, T *tr, T *src, T *pt, T f, T k1, T k2) {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t(tr);

    T p[3];
    ceres::AngleAxisRotatePoint(R, src, p);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<T> Pcam = Prot + t;

    Eigen::Vector3<T> Pimg = -Pcam / Pcam(2); // z division, minus because of camera model in BAL
    T r2 = Pimg.topRows(2).squaredNorm();
    T distortion = T(1) + r2 * (k1 + r2 * k2);

    Eigen::Vector3<T> pred = f * distortion * Pimg;
    pt[0] = pred.x();
    pt[1] = pred.y();
    pt[2] = pred.z();
}
/*
void BundleAdjustment::projectFrom3D(int cam_id) {
    double error1 = 0.;
    double error2 = 0.;
    for (int i = 0; i < dataset.num_observations; ++i) {
        if (true || dataset.obs_cam[i] == cam_id) {
            auto camera = dataset.cameras[cam_id];
            double R[3] = {camera.R[0], camera.R[1], camera.R[2]};
            double t[3] = {camera.t[0], camera.t[1], camera.t[2]};
            double f = camera.f;
            double k1 = camera.k1;
            double k2 = camera.k2;

            auto[x, y, z] = dataset.points[dataset.obs_point[i]];
            double src[3] = {x, y, z};
            double dst[3];

            transformation(R, t, src, dst, f, k1, k2);

            std::pair<double, double> obs = dataset.observations[i];

            error1 += (obs.first - dst[0]) * (obs.first - dst[0]);
            error2 += (obs.second - dst[1]) * (obs.second - dst[1]);
//            std::cout << dataset.observations[i].first << " " << dst[0] << std::endl;
//            std::cout << dataset.observations[i].second << " " << dst[1] << std::endl << std::endl;
        }
    }
    std::cout << error1 << " " << error2 << std::endl;
}*/

/*

void BundleAdjustment::writeMesh(std::string filename) {

    std::ofstream file(filename);
    if (file.is_open()) {
        //create a 3D camera model with scale factor lambda
        double lambda = .1;
        double camera_init_points[15] = {0.0, 0.0, 0.0,
                                         -0.5, 0.0, 1.0,
                                         0.5, 0.0, 1.0,
                                         0.0, -0.5, 1.0,
                                         0.0, 0.5, 1.0};

        for (int i = 0; i < 15; ++i) {
            camera_init_points[i] *= lambda;
        }

        file << "OFF" << std::endl;
        file << 1 * dataset.num_points + 0 * dataset.num_camera << " " << 0 * dataset.num_camera << " 0" << std::endl;

        for (int i = 0; i < dataset.num_points; ++i) {
            auto [x, y, z] = dataset.points[i];
            file << x << " " << y << " " << z << "\n";
        }

        for (int i = 0; i < dataset.num_camera; ++i) {
            for (int j = 0; j < 5; ++j) {
                double p[3];
                double convert = M_PI / 180.;
                double R[3] = {convert * dataset.cameras[i].R[0], convert * dataset.cameras[i].R[1],
                               convert * dataset.cameras[i].R[2]};
                double T[3] = {dataset.cameras[i].t[0], dataset.cameras[i].t[1], dataset.cameras[i].t[2]};
                file << p[0] << " " << p[1] << " " << p[2] << "\n";
            }
        }

        for (int i = 0; i < dataset.num_camera; ++i) {
            int p[5];
            for (int j = 0; j < 5; ++j) {
                p[j] = 1 * dataset.num_points + i * 5 + j;
            }
            file << "3 " << p[0] << " " << p[1] << " " << p[3] << " 1.0 0.0 0.0\n";
            file << "3 " << p[0] << " " << p[3] << " " << p[2] << " 0.0 1.0 0.0\n";
            file << "3 " << p[0] << " " << p[2] << " " << p[4] << " 0.0 0.0 1.0\n";
            file << "3 " << p[0] << " " << p[4] << " " << p[1] << " 1.0 1.0 0.0\n";
        }
    }
}
*/

