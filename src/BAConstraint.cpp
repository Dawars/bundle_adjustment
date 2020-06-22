//
// Created by Komorowicz David on 2020. 06. 20..
//

#include <ceres/rotation.h>
#include <Eigen/Dense>

#include "bundleadjust/BAConstraint.h"

BAConstraint::BAConstraint(const Eigen::Vector3f &observation, const Camera &camera) :
        observation{observation},
        camera{camera} {}

template<typename T>
bool BAConstraint::operator()(const T *const point, const T *const rot, const T *const tr, T *residuals) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t(tr);

    T f = T(camera.f);
    T k1 = T(camera.k1);
    T k2 = T(camera.k2);

    T p[3];
    ceres::AngleAxisRotatePoint(rot, point, p);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<T> Pcam = Prot + t;

    // fixme 0 division with this initialization
    Eigen::Vector3<T> Pimg = -Pcam / Pcam(2); // z division, - because of camera model in BAL
    T r2 = Pimg.topRows(2).squaredNorm();
    T distortion = T(1) + r2 * (k1 + r2 * k2);

    Eigen::Vector3<T> pred = f * distortion * Pimg;
    Eigen::Vector3<T> res = observation.cast<T>() - pred;

    residuals[0] = res(0);
    residuals[1] = res(1);

    return true;
}

ceres::CostFunction *BAConstraint::create(const Eigen::Vector3f &observation, const Camera &camera) {
    return new ceres::AutoDiffCostFunction<BAConstraint, 2, 3, 3, 3>(
            new BAConstraint(observation, camera)
    );
}