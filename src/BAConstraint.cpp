//
// Created by Komorowicz David on 2020. 06. 20..
//

#include <ceres/rotation.h>
#include <Eigen/Dense>

#include "harris/BAConstraint.h"

BAConstraint::BAConstraint(const Eigen::Vector3f& observation, const Camera& camera) :
        observation{ observation },
        camera{ camera }
{ }

template <typename T>
bool BAConstraint::operator()(const T* const point, const T* const rot, const T* const tr, T* residuals) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > R(rot);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t(tr);

    T p[3];
    ceres::AngleAxisRotatePoint(rot, point, p);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > P(p);

    // todo apply undistort, project
//    Eigen::Matrix3f proj = camera.projectionMatrix();
    auto res = observation.cast<T>() - (P+t);

    residuals[0] = res(0);
    residuals[1] = res(1);

    return true;
}

ceres::CostFunction *BAConstraint::create(const Eigen::Vector3f &observation, const Camera &camera) {
    return new ceres::AutoDiffCostFunction<BAConstraint, 2, 3, 3, 3>(
            new BAConstraint(observation, camera)
    );
}