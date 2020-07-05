//
// Created by Komorowicz David on 2020. 06. 20..
//

#include <ceres/rotation.h>
#include <Eigen/Dense>

#include "bundleadjust/BAConstraint.h"

BAConstraint::BAConstraint(const cv::Point2f &observation) :
        observation{observation} {}

template<typename T>
bool BAConstraint::operator()(const T *const point, const T *const rot, const T *const tr, const T *const intrinsics, T *residuals) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t(tr);

    T f = intrinsics[0];
    T k1 = intrinsics[1];
    T k2 = intrinsics[2];

    T p[3];
    ceres::AngleAxisRotatePoint(rot, point, p);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<T> Pcam = Prot + t;

    Eigen::Vector3<T> Pimg = -Pcam / Pcam(2); // z division, minus because of camera model in BAL
    T r2 = Pimg.topRows(2).squaredNorm();
    T distortion = T(1) + r2 * (k1 + r2 * k2);

    Eigen::Vector3<T> pred = f * distortion * Pimg;
//    Eigen::Vector3<T> res = observation.cast<T>() - pred;

    residuals[0] = T(observation.x) - pred(0);
    residuals[1] = T(observation.y) - pred(1);

    return true;
}

ceres::CostFunction *BAConstraint::create(const cv::Point2f &observation) {
    return new ceres::AutoDiffCostFunction<BAConstraint, 2, 3, 3, 3, 3>(
            new BAConstraint(observation)
    );
}