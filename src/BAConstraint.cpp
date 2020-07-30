//
// Created by Komorowicz David on 2020. 06. 20..
//

#include <ceres/rotation.h>
#include <Eigen/Dense>

#include "bundleadjust/BAConstraint.h"

BAConstraint::BAConstraint(const Eigen::Vector3f &observation, const float * estimatedPose) :
        observation{observation} {
             this->pose = Eigen::Map<const Eigen::Matrix4f>(estimatedPose);
        }

template<typename T>
bool BAConstraint::operator()(const T *const point, const T *const rot, const T *const tr, const T *const intrinsics, T *residuals) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(tr);

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

    T R[9];
    ceres::AngleAxisToRotationMatrix<T>(rot, R);


    Eigen::Map<const Eigen::Matrix3<T>>rotation_matrix(R);
    Eigen::Matrix<T, 4, 4> extr;

    extr << T(rotation_matrix(0,0)), T(rotation_matrix(0,1)), T(rotation_matrix(0,2)), T(tr[0]),
            T(rotation_matrix(1,0)), T(rotation_matrix(1,1)), T(rotation_matrix(1,2)), T(tr[1]),
            T(rotation_matrix(2,0)), T(rotation_matrix(2,1)), T(rotation_matrix(2,2)), T(tr[2]),
            T(0), T(0), T(0), T(1);

    const Eigen::Matrix4<T> extrInv = extr.inverse();



    //ceres::AngleAxisRotatePoint(rot, point, p);

    Eigen::Matrix<T, 4, 1> homogenous_world_point;
    homogenous_world_point << point[0],
                              point[1], 
                              point[2], 
                              T(1);
    Eigen::Matrix<T, 4, 1> cam_point = extrInv * homogenous_world_point;



    //Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<T> Pcam;
    Pcam << cam_point(0,0), cam_point(1,0), cam_point(2,0);    

    // todo unify camera model, signed
    Eigen::Vector3<T> Pimg;
    Pimg(0) = Pcam(0) / Pcam(2); // z division, minus because of camera model in BAL
    Pimg(1) = Pcam(1) / Pcam(2);
    T r2 = Pimg.topRows(2).squaredNorm();
    T d = T(1) + r2 * (k1 + r2 * k2);

    Eigen::Vector3<T> Pproj = intr * Pimg;
    Eigen::Vector3<T> res = observation.cast<T>() - Pproj;


    residuals[0] = res(0);
    residuals[1] = res(1);

    return true;
}

ceres::CostFunction *BAConstraint::create(const Eigen::Vector3f &observation, const float * estimatedPose) {
    return new ceres::AutoDiffCostFunction<BAConstraint, 2, 3, 3, 3, 6>(
            new BAConstraint(observation, estimatedPose)
    );
}


// bool BAConstraint::printOp(double * point, double * rot, double * tr, double * intrinsics) const {
//     // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
//     Eigen::Map<const Eigen::Matrix<double, 3, 1> > t(tr);

//     double fx = intrinsics[0];
//     double fy = intrinsics[1];
//     double ox = intrinsics[2];
//     double oy = intrinsics[3];
//     double k1 = intrinsics[4];
//     double k2 = intrinsics[5];

//     Eigen::Matrix3<double> intr;
//     intr << fx, 0, ox,
//             0, fy, oy,
//             0, 0, 1;

//     double p[3];
//     ceres::AngleAxisRotatePoint<double>(rot, point, p);
//     Eigen::Map<const Eigen::Matrix<double, 3, 1> > Prot(p);
//     Eigen::Vector3<double> Pcam = Prot + t;

//     // todo unify camera model, signed


//     Eigen::Vector3<double> Pimg = Pcam / Pcam(2); // z division, minus because of camera model in BAL
//     double r2 = Pimg.topRows(2).squaredNorm();
//     double d = 1 + r2 * (k1 + r2 * k2);

//     Eigen::Vector3<double> Pproj = intr * Pimg;

//     std::cout << "Pproj: " << Pproj(0) << " " << Pproj(1) << "\n";
//     std::cout << "Obs: " << observation(0) << " " << observation(1) << "\n";



//     Eigen::Vector3<double> res = observation.cast<double>() - Pproj;


//     return true;
// }


bool BAConstraint::printOp(double * point, double * rot, double * tr, double * intrinsics, const float * estimatedPose) const {
    // http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
    double fx = intrinsics[0];
    double fy = intrinsics[1];
    double ox = intrinsics[2];
    double oy = intrinsics[3];
    double k1 = intrinsics[4];
    double k2 = intrinsics[5];

    Eigen::Matrix3<double> intr;
    intr << fx, 0, ox,
            0, fy, oy,
            0, 0, 1;

    double p[3];

    double R[9];
    ceres::AngleAxisToRotationMatrix<double>(rot, R);


    Eigen::Matrix<double, 4, 4> extr;
    Eigen::Map<const Eigen::Matrix3<double>>rotation_matrix(R);

    extr << rotation_matrix(0,0), rotation_matrix(0,1), rotation_matrix(0,2), tr[0],
            rotation_matrix(1,0), rotation_matrix(1,1), rotation_matrix(1,2), tr[1],
            rotation_matrix(2,0), rotation_matrix(2,1), rotation_matrix(2,2), tr[2],
            0, 0, 0, 1;
//     extr << R[0], R[1], R[2], tr[0],
//             R[3], R[4], R[5], tr[1],
//             R[6], R[7], R[8], tr[2],
//             0, 0, 0, 1;


//     extr << estimatedPose[0], estimatedPose[1], estimatedPose[2], estimatedPose[3],
//             estimatedPose[4], estimatedPose[5], estimatedPose[6], estimatedPose[7],
//             estimatedPose[8], estimatedPose[9], estimatedPose[10], estimatedPose[0],
//             0, 0, 0, 1;

    const Eigen::Matrix4<double> extrInv = extr.inverse();



    //ceres::AngleAxisRotatePoint(rot, point, p);

    Eigen::Matrix<double, 4, 1> homogenous_world_point;
    homogenous_world_point << point[0],
                              point[1], 
                              point[2], 
                              1;
    Eigen::Matrix<double, 4, 1> cam_point = extrInv * homogenous_world_point;



    //Eigen::Map<const Eigen::Matrix<T, 3, 1> > Prot(p);
    Eigen::Vector3<double> Pcam;
    Pcam << cam_point(0,0), cam_point(1,0), cam_point(2,0);    

    // todo unify camera model, signed
    Eigen::Vector3<double> Pimg;
    Pimg(0) = Pcam(0) / Pcam(2); // z division, minus because of camera model in BAL
    Pimg(1) = Pcam(1) / Pcam(2);
    double r2 = Pimg.topRows(2).squaredNorm();
    double d = 1 + r2 * (k1 + r2 * k2);

    
    Eigen::Vector3<double> Pproj = intr * Pimg;
    std::cout << "Proj: " << Pimg(0) << " " << Pimg(1) << "\n";
    std::cout << "Obs: " << observation(0) << " " << observation(1) << "\n";
    Eigen::Vector3<double> res = observation.cast<double>() - Pproj;



    return true;
}