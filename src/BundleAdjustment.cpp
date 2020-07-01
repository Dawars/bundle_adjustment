//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>
#include <ceres/rotation.h>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BAConstraint.h"

BundleAdjustment::BundleAdjustment(const BalDataloader &dataset) : dataset(dataset) {reset();}

BundleAdjustment::~BundleAdjustment() {
    delete[] R;
    delete[] T;
    delete[] X;
}

void BundleAdjustment::reset() {
    //todo maybe init with data from point matching

    R = new double[dataset.num_camera * 3];
    T = new double[dataset.num_camera * 3];
    X = new double[dataset.num_points * 3]; // reconstructed 3D points

    std::fill_n(R, dataset.num_camera * 3, 0);
    std::fill_n(T, dataset.num_camera * 3, 0);
    std::fill_n(X, dataset.num_points * 3, 1);
    for (int i = 0; i < dataset.num_camera; ++i) {
        R[i + i % 3] = dataset.cameras[i / 3].R[i % 3];
        T[i + i % 3] = dataset.cameras[i / 3].t[i % 3];
    }
    for (int i = 0; i < dataset.num_points; ++i) {
        auto[x, y, z] = dataset.points[i];
        X[3 * i] = x;
        X[3 * i + 1] = y;
        X[3 * i + 2] = z;
    }
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
    auto solvers = {ceres::SPARSE_NORMAL_CHOLESKY, ceres::DENSE_SCHUR, ceres::SPARSE_SCHUR, ceres::ITERATIVE_SCHUR, ceres::CGNR};
    for (auto solver : solvers) {
        reset();
        ceres::Solver::Options options;
        configureSolver(options);
        options.linear_solver_type = solver;

        // Run the solver (for one iteration).
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << ceres::LinearSolverTypeToString(solver) << std::endl;
        std::cout << summary.FullReport() << std::endl;
    }
}

void BundleAdjustment::configureSolver(ceres::Solver::Options &options) {
    // Ceres options.
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
//    options.linear_solver_type = ceres::LinearSolverType;//SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 600;
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

