//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>
#include <ceres/rotation.h>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BAConstraint.h"

BundleAdjustment::BundleAdjustment(const BalDataloader &dataset) : dataset(dataset) {
//todo maybe init with data from point matching
    R = new double[dataset.num_camera * 3];
    T = new double[dataset.num_camera * 3];
    X = new double[dataset.num_points * 3]; // reconstructed 3D points

    std::fill_n(R, dataset.num_camera * 3, 0);
    std::fill_n(T, dataset.num_camera * 3, 0);
    std::fill_n(X, dataset.num_points * 3, 1);
}

BundleAdjustment::~BundleAdjustment() {
    delete[] R;
    delete[] T;
    delete[] X;
}

void BundleAdjustment::createProblem() {
    for (int i = 0; i < dataset.num_observations; ++i) {

        // get camera for observation
        size_t camIndex = dataset.obs_cam[i];
        size_t pointIndex = dataset.obs_point[i];

        Eigen::Vector3f obs = {dataset.observations[i].first, dataset.observations[i].second, 1};

        auto cost_function = BAConstraint::create(obs, dataset.cameras[camIndex]);

        problem.AddResidualBlock(cost_function,
                                 nullptr /* squared loss */,
                                 getPoint(pointIndex),
                                 getRotation(camIndex),
                                 getTranslation(camIndex)
        );
    }
}

void BundleAdjustment::solve() {
    ceres::Solver::Options options;
    configureSolver(options);

    // Run the solver (for one iteration).
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

}

void BundleAdjustment::configureSolver(ceres::Solver::Options &options) {
    // Ceres options.
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 10;
    options.num_threads = 4;
}

double *BundleAdjustment::getRotation(size_t cameraIndex) {
    return &R[cameraIndex * 3];
}

double *BundleAdjustment::getTranslation(size_t cameraIndex) {
    return &T[cameraIndex * 3];
}

double *BundleAdjustment::getPoint(size_t pointIndex) {
    return &X[pointIndex * 3];
}


template<typename T>
void transformation(T *R, T *tr, T *src, T *pt) {
    T R_matrix[9];
    ceres::EulerAnglesToRotationMatrix(R, 0, R_matrix);
    T R_inverse[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_inverse[3 * i + j] = R_matrix[3 * j + i];
        }
    }
    for (int i = 0; i < 3; ++i) {
        pt[i] = -tr[i];
        for (int j = 0; j < 3; ++j) {
            pt[i] += R[3 * i + j] * src[j];
        }
    }
}

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
                transformation(&R[0], &T[0], &camera_init_points[3 * j], &p[0]);
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

