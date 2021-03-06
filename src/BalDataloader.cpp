//
// Created by Komorowicz David on 2020. 06. 20..
//
#include <fstream>
#include <tuple>

#include "./bundleadjust/BalDataloader.h"

BalDataloader::BalDataloader(std::string path) {
//    <num_cameras> <num_points> <num_observations>
//    <camera_index_1> <point_index_1> <x_1> <y_1>
//    ...
//    <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
//    <camera_1>
//    ...
//    <camera_num_cameras>
//    <point_1>
//    ...
//    <point_num_points>

    std::ifstream file(path);
    if(!file.is_open()) { throw std::invalid_argument("File could not be opened"); }

    file >> num_camera >> num_points >> num_observations;

    obs_cam.resize(num_observations);
    obs_point.resize(num_observations);
    observations.resize(num_observations);

    for (int i = 0; i < num_observations; ++i) {
        int cam_index, point_index;
        float x, y; // 2D image coordinates
        file >> cam_index >> point_index >> x >> y;

        observations[i] = std::pair<float, float>(x, y);
        obs_cam[i] = cam_index;
        obs_point[i] = point_index;
    }

    cameras.resize(num_camera);

    for (int i = 0; i < num_camera; ++i) {
//        R3,t3,f,k1 and k2
        float R[3], T[3], f, k1, k2;
        file >> R[0] >> R[1] >> R[2] >> T[0] >> T[1] >> T[2] >> f >> k1 >> k2;

        // convert from right handed to left handed system
        // https://stackoverflow.com/questions/31191752/right-handed-euler-angles-xyz-to-left-handed-euler-angles-xyz
        cameras[i] = {R[0], R[1], R[2], T[0], T[1], T[2], f, k1, k2};
    }

    points.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        // 3D ground truth - multiple lines
        float x, y, z;
        file >> x >> y >> z;
        points[i] = {x, y, z};
    }
}


BalDataloader::~BalDataloader() {}


std::vector<cv::Point2f> BalDataloader::getObservations() const {
    std::vector<cv::Point2f> obs;

    for (auto &pt : this->observations) {
        obs.emplace_back(pt.first, pt.second);
    }

    return obs;
}

int BalDataloader::getObsCam(int index) const {
    return this->obs_cam[index];
}

int BalDataloader::getObsPoint(int index) const {
    return this->obs_point[index];
}

int BalDataloader::getNumPoints() const {
    return this->num_points;
}

int BalDataloader::getNumObservations() const {
    return this->num_observations;
}

int BalDataloader::getNumFrames() const {
    return this->num_camera;
}

bool BalDataloader::isColorAvailable() const {
    return false;
}

bool BalDataloader::isDepthAvailable() const {
    return false;
}

cv::Mat BalDataloader::getColor(int frameId) const {
    return cv::Mat();
}

cv::Mat BalDataloader::getDepth(int frameId) const {
    return cv::Mat();
}

// Return a random number sampled from a uniform distribution in the range
// [0,1].
inline double RandDouble() {
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// Marsaglia Polar method for generation standard normal (pseudo)
// random numbers http://en.wikipedia.org/wiki/Marsaglia_polar_method
inline double RandNormal() {
    double x1, x2, w;
    do {
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 || w == 0.0 );

    w = sqrt((-2.0 * log(w)) / w);
    return x1 * w;
}
void BalDataloader::initialize(double *R, double *T, double *intrinsics, double *X) {
    double sigmaCam = 0.01;
    double sigmaX = 2;

    for (int i = 0; i < num_camera; ++i) {
        auto &cam = cameras[i];
        R[3 * i + 0] = cam.R[0] + RandNormal() * sigmaCam;
        R[3 * i + 1] = cam.R[1] + RandNormal() * sigmaCam;
        R[3 * i + 2] = cam.R[2] + RandNormal() * sigmaCam;
        T[3 * i + 0] = cam.t[0] + RandNormal() * sigmaCam;
        T[3 * i + 1] = cam.t[1] + RandNormal() * sigmaCam;
        T[3 * i + 2] = cam.t[2] + RandNormal() * sigmaCam;
        intrinsics[6 * i + 0] = cam.f; // fx
        intrinsics[6 * i + 1] = cam.f; // fy
        intrinsics[6 * i + 2] = 0; // ox
        intrinsics[6 * i + 3] = 0; // oy
        intrinsics[6 * i + 4] = cam.k1; // k1
        intrinsics[6 * i + 5] = cam.k2; // k2
    }

    for (int i = 0; i < num_points; ++i) {
        auto & pt = points[i];
        X[3*i + 0] = std::get<0>(pt) + RandNormal() * sigmaX;
        X[3*i + 1] = std::get<1>(pt) + RandNormal() * sigmaX;
        X[3*i + 2] = std::get<2>(pt) + RandNormal() * sigmaX;
    }
}

Eigen::Vector3i BalDataloader::getPointColor(int point_index) const {
    Eigen::Vector3i color;
    color.setConstant(255);
    return color;
}
