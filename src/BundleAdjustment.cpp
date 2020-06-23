//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>

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
                                 getRotation(camIndex),
                                 getTranslation(camIndex),
                                 getPoint(pointIndex)
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

void BundleAdjustment::writeMesh(std::string filename) {

    std::ofstream file(filename);
    if (file.is_open()) {
        file << "OFF" << std::endl;
        file << dataset.num_points << " 0 0" << std::endl;

        for (int i = 0; i < dataset.num_points; ++i) {
            auto point = getPoint(i);

            file << point[0] << " " << point[1] << " " << point[2] << std::endl;
        }
    }
}

