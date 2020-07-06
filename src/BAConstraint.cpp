//
// Created by Komorowicz David on 2020. 06. 20..
//

#include <ceres/rotation.h>
#include <Eigen/Dense>

#include "bundleadjust/BAConstraint.h"

BAConstraint::BAConstraint(const Eigen::Vector3f &observation) :
        observation{observation} {}

template<typename T>
bool BAConstraint::operator()(const T *const point, const T *const rot, const T *const tr, const T *const intrinsics, T *residuals) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t(tr);

    T fx = intrinsics[0];
    T fy = intrinsics[1];
    T ox = intrinsics[2];
    T oy = intrinsics[3];
    T k1 = intrinsics[4];
    T k2 = intrinsics[5];

    Eigen::Matrix3<T> intr;
    intr << fx, T(0), ox,
            T(0), fy, oy,
            T(0), T(0), T(1);

    T p[3];
    ceres::AngleAxisRotatePoint(rot, point, p);
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<T> Pcam = Prot + t;

    // todo unify camera model, signed
    Eigen::Vector3<T> Pimg = -Pcam / Pcam(2); // z division, minus because of camera model in BAL
    T r2 = Pimg.topRows(2).squaredNorm();
    T d = T(1) + r2 * (k1 + r2 * k2);

    Eigen::Vector3<T> Pproj = d * intr * Pimg;

    Eigen::Vector3<T> res = observation.cast<T>() - Pproj;

    residuals[0] = res(0);
    residuals[1] = res(1);

    return true;
}

ceres::CostFunction *BAConstraint::create(const Eigen::Vector3f &observation) {
    return new ceres::AutoDiffCostFunction<BAConstraint, 2, 3, 3, 3, 6>(
            new BAConstraint(observation)
    );
}