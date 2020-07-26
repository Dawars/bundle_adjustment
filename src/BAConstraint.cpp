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


    Eigen::Matrix<T, 4, 4> extr;
    extr << T(R[0]), T(R[1]), T(R[2]), T(tr[0]),
            T(R[3]), T(R[4]), T(R[5]), T(tr[1]),
            T(R[6]), T(R[7]), T(R[8]), T(tr[2]),
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
    Eigen::Map<const Eigen::Matrix<double, 3, 1> > t(tr);

    double fx = intrinsics[0];
    double fy = intrinsics[1];
    double ox = intrinsics[2];
    double oy = intrinsics[3];
    double k1 = intrinsics[4];
    double k2 = intrinsics[5];

    Eigen::Matrix3f intr;
    intr << fx, 0, ox,
            0, fy, oy,
            0, 0, 1;



    Eigen::Matrix4f pose = Eigen::Map<const Eigen::Matrix4f>(estimatedPose);
    Eigen::Matrix4f poseInv = pose.inverse();
    Eigen::Vector4f point_hom;
    point_hom << point[0], point[1], point[2], 1;
    Eigen::Vector4f cam_point = poseInv * point_hom;
    Eigen::Vector3f cam3;
    cam3 << cam_point(0), cam_point(1), cam_point(2);

    
    double p[3];
    ceres::AngleAxisRotatePoint(rot, point, p);
    Eigen::Map<const Eigen::Matrix<double, 3, 1> > Prot(p);
    Eigen::Vector3<double> Pcam = Prot + t;

    Eigen::Vector3f imP = intr*cam3;
    //imP = imP / imP(2);

    // todo unify camera model, signed

        
    Eigen::Vector3f Pimg;
    Pcam(0) = Pcam(0)/Pcam(2); // z division, minus because of camera model in BAL
    Pcam(1) = Pcam(1)/Pcam(2);
    double r2 = Pimg.topRows(2).squaredNorm();
    double d = 1 + r2 * (k1 + r2 * k2);

    Eigen::Vector3f Pproj = intr * Pimg;

    imP(0) = imP(0)/imP(2);
    imP(1) = imP(1)/imP(2);
    std::cout << "Pproj: " << imP(0) << " " << imP(1) << "\n";
    std::cout << "Obs: " << observation(0) << " " << observation(1) << "\n";


    Eigen::Vector3f res;
    res<< observation(0) - Pproj(0), observation(1) - Pproj(1), 1;


    return true;
}