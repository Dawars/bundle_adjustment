//
// Created by Komorowicz David on 2020. 06. 25..
//
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


template<typename T, unsigned int n, unsigned m>
std::istream &operator>>(std::istream &in, Eigen::Matrix<T, n, m> &other) {
    for (unsigned int i = 0; i < other.rows(); i++)
        for (unsigned int j = 0; j < other.cols(); j++)
            in >> other(i, j);
    return in;
}

template<typename T, unsigned int n, unsigned m>
std::ostream &operator<<(std::ostream &out, const Eigen::Matrix<T, n, m> &other) {
    std::fixed(out);
    for (int i = 0; i < other.rows(); i++) {
        out << other(i, 0);
        for (int j = 1; j < other.cols(); j++) {
            out << "\t" << other(i, j);
        }
        out << std::endl;
    }
    return out;
}

template<typename T>
std::istream &operator>>(std::istream &in, Eigen::Quaternion<T> &other) {
    in >> other.x() >> other.y() >> other.z() >> other.w();
    return in;
}

template<typename T>
std::ostream &operator<<(std::ostream &out, const Eigen::Quaternion<T> &other) {
    std::fixed(out);
    out << other.x() << "\t" << other.y() << "\t" << other.z() << "\t" << other.w();
    return out;
}

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor {
public:

    VirtualSensor(int increment = 10);

    ~VirtualSensor();

    bool Init(const std::string &datasetDir);

    bool ProcessNextFrame();

    unsigned int GetCurrentFrameCnt();

    // get current color data
    cv::Mat GetColor() const;

    // get current depth data
    cv::Mat GetDepth() const;

    // color camera info
    Eigen::Matrix3f GetColorIntrinsics();

    Eigen::Matrix4f GetColorExtrinsics();

    unsigned int GetColorImageWidth();

    unsigned int GetColorImageHeight();

    // depth (ir) camera info
    Eigen::Matrix3f GetDepthIntrinsics();

    Eigen::Matrix4f GetDepthExtrinsics();

    unsigned int GetDepthImageWidth();

    unsigned int GetDepthImageHeight();

    // get current trajectory transformation
    Eigen::Matrix4f GetTrajectory();

    std::vector<Eigen::Matrix4f> GetTrajectories();

private:

    bool
    ReadFileList(const std::string &filename, std::vector<std::string> &result, std::vector<double> &timestamps);

    bool ReadTrajectoryFile(const std::string &filename, std::vector<Eigen::Matrix4f> &result,
                            std::vector<double> &timestamps);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // current frame index
    int m_currentIdx;

    int m_increment;

    // frame data
//	float* m_depthFrame;
//	BYTE* m_colorFrame;
    cv::Mat rgbImage;
    cv::Mat depthImage;

    Eigen::Matrix4f m_currentTrajectory;

    // color camera info
    Eigen::Matrix3f m_colorIntrinsics;
    Eigen::Matrix4f m_colorExtrinsics;
    unsigned int m_colorImageWidth;
    unsigned int m_colorImageHeight;

    // depth (ir) camera info
    Eigen::Matrix3f m_depthIntrinsics;
    Eigen::Matrix4f m_depthExtrinsics;
    unsigned int m_depthImageWidth;
    unsigned int m_depthImageHeight;

    // base dir
    std::string m_baseDir;
    // filenamelist depth
    std::vector<std::string> m_filenameDepthImages;
    std::vector<double> m_depthImagesTimeStamps;
    // filenamelist color
    std::vector<std::string> m_filenameColorImages;
    std::vector<double> m_colorImagesTimeStamps;

    // trajectory
    std::vector<Eigen::Matrix4f> m_trajectory;
    std::vector<double> m_trajectoryTimeStamps;

};
